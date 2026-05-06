# Auto-converted from notebook

# %% [cell 1: markdown]
# # ***Libraries***

# %% [cell 2: markdown]
# ## ***Install***

# %% [cell 3: code]
%%capture
%pip install -U transformers
%pip install -U datasets
%pip install -U accelerate
%pip install -U peft
%pip install -U trl
%pip install -U bitsandbytes
%pip install wandb


!pip install openpyxl
!pip install typing_extensions==4.11.0
!pip install tiktoken
!pip install protobuf
!pip install sentencepiece

!pip install -U huggingface_hub
!pip install scikit-learn

# %% [cell 4: markdown]
# ## ***Import***

# %% [cell 5: code]
import transformers
import torch

import os
import gc
import json
import time
import random
import datetime
import pandas as pd
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from datasets import Dataset, load_dataset

# from unsloth import FastLanguageModel

import bitsandbytes as bnb
from trl import SFTTrainer, setup_chat_format, SFTConfig
tqdm.pandas()

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    PeftModel,
    PeftConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# %% [cell 6: code]
from huggingface_hub import login

login(token="Your HuggingFace Token")

# %% [cell 7: code]
import torch
# Set torch dtype and attention implementation
if torch.cuda.get_device_capability()[0] >= 8:
    # !pip install -qqq flash-attn
    torch_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
else:
    torch_dtype = torch.float16
    attn_implementation = "eager"

# %% [cell 8: code]
from transformers import logging

logging.set_verbosity_warning()

# %% [cell 9: markdown]
# # ***Download Model***

# %% [cell 10: markdown]
# You can download any open-weight model you need to finetune. Classes are written to handle any model differences. 

# %% [cell 11: code]
!huggingface-cli download medalpaca/medalpaca-7b --local-dir /workspace/models/medAlpaca7B

# %% [cell 12: markdown]
# # ***Constants***

# %% [cell 13: code]
root_dir = '/workspace/'
data_dir = root_dir + 'data/'
pred_dir = root_dir + 'predictions/'
train_dataset_path = data_dir + 'train.csv'
valid_dataset_path = data_dir + 'validation.csv'
test_dataset_path = data_dir + 'Test_DePiC.xlsx'

# Add your Huggingface token here
hf_access_token = ""

# Label mapping
label_mapping_dict = {0: "Healthy", 1: "AD"}

# System instruction for chat template
classification_system_prompt = """You are an expert cognitive impairment analyst.
Your role is to evaluate spoken language transcripts and classify them based on linguistic patterns.
"""

# Task prompt to finetune and inference for Llama models and Ministral
classification_task_prompts =[ (
    "You are an expert in cognitive health and language analysis. You will analyze a spoken language transcript from a person describing the 'cookie theft' picture. This is not written text but a transcription of spontaneous speech."
    "\nAnalyze the provided transcript and classify it into one of two categories: 'Healthy' for a healthy cognitive state or 'AD' for Alzheimer's disease."
    "\nProvide only the label ('Healthy' or 'AD') as the output. Do not include explanations or additional text."
    "\nText: {text}"
    # "\nLabel:"
)]

# Prompt used for MedAlpaca to handle tokenizer differences.
classification_task_prompt_single = '''You are an expert in cognitive health and language analysis. You will analyze a spoken language transcript from a person describing the 'cookie theft' picture. This is not written text but a transcription of spontaneous speech.
    Analyze the provided transcript and classify it into one of two categories: 'Healthy' for a healthy cognitive state or 'AD' for Alzheimer's disease.
    Provide only the label ('Healthy' or 'AD') as the output. Do not include explanations or additional text.
    '''

# %% [cell 14: markdown]
# # ***Model***

# %% [cell 15: code]
class ModelHandler:
    def __init__(
        self,
        base_model: str,
        load_quantized: int = None, # choose values of [None, 4, 8]
        device_map: str = "auto",
        tokenizer_trust_remote_code: bool = True,
        max_seq_length=1024,
        linear_modules=None, # pass required modules in a list, or None for all
        use_lora: bool = True,
        lora_rank: int = 32,
        lora_dropout: float = 0.05,
    ):
        self.base_model = base_model
        self.load_quantized = load_quantized
        self.device_map = device_map
        self.tokenizer_trust_remote_code = tokenizer_trust_remote_code
        self.tokenizer_pad_token = False
        self.max_seq_length = max_seq_length

        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_dropout = lora_dropout

        self.model = None
        self.tokenizer = None
        self.lora_config = None

        # Determine attention backend and tensor dtype
        self.torch_dtype, self.attn_implementation = self.set_attention_config()

        # Load both model and tokenizer at initialization
        self.load_model_and_tokenizer(linear_modules)

    def find_all_linear_names(self, model):
        """
        Collects names of all Linear4bit layers in the model.
        Used to identify target modules for LoRA injection.
        """
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        # Exclude lm_head for stability in mixed precision (16-bit)
        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def get_linear_modules(self, model):
        """
        Returns target linear modules for LoRA.
        If bitsandbytes is available, detect automatically;
        otherwise, fall back to a standard transformer module set.
        """
        modules = self.find_all_linear_names(model)
        if len(modules) > 1:
            # Works when bitsandbytes is enabled
            return modules
        else:
            return [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

    def set_attention_config(self):
        """
        Sets attention implementation and dtype.
        Placeholder: currently defaults to float16 + eager attention.
        """
        return torch.float16, "eager"

    def load_model_and_tokenizer(self, linear_modules):
        """
        Loads the tokenizer, model, and applies LoRA if configured.
        Handles quantization (4-bit or 8-bit) when requested.
        """

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=self.tokenizer_trust_remote_code, legacy=False
        )
        # Ensure tokenizer has a pad token (fallback to EOS if missing)
        if not self.tokenizer.pad_token:
            self.tokenizer_pad_token = True
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print('****Tokenizer Loaded****')

        # Configure quantization (only 4-bit or 8-bit supported)
        quantization_config = None
        if self.load_quantized in [4, 8]:
            quantization_config = {"load_in_8bit": False, "load_in_4bit": False}
            if self.load_quantized == 4:
                print("loading in 4 bit")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.torch_dtype,
                    bnb_4bit_use_double_quant=True,
                )
            elif self.load_quantized == 8:
                print("loading in 8 bit")
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model with quantization and dtype configs
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            quantization_config=quantization_config
        )

        print('****Model Loaded****')

        # Set LoRA modules (use auto-detection if not provided)
        self.linear_modules = (
            self.get_linear_modules(self.model) if linear_modules is None else linear_modules
        )

        # Apply LoRA adapter if enabled
        if self.linear_modules and self.use_lora:
            self.lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=2 * self.lora_rank,
                lora_dropout=self.lora_dropout,
                target_modules=self.linear_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, self.lora_config)
            print('****LoRA Conf Loaded****')
       
        # Reset generation config defaults for inference
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        torch.cuda.empty_cache()

        # Ensure pad token id is consistent in generation config
        if self.tokenizer_pad_token:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def get_model_and_tokenizer(self):
        """Returns both model and tokenizer."""
        return self.model, self.tokenizer

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model

    def get_peft_config(self):
        return self.lora_config

# %% [cell 16: markdown]
# # ***Dataset***

# %% [cell 17: code]
class PromptConstructor:
    tokenizer_cache = {}

    def __init__(self):
        pass

    @classmethod
    def get_tokenizer(cls, model_name):
        # Check if the tokenizer for the model is already cached
        if model_name not in cls.tokenizer_cache:
            cls.tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name, token=hf_access_token)
        return cls.tokenizer_cache[model_name]

    @staticmethod
    def get_start_of_assistant(tokenizer, chat_messages):
        """
        Get the starting token sequence for the assistant's generation.

        Args:
            tokenizer: A tokenizer object capable of applying a chat template.
            chat_messages: List of dicts containing role and content for each chat message.

        Returns:
            str: The token sequence for the assistant role.

        Raises:
            TokenizationError: If there are issues in producing the token sequence.
        """

        def apply_chat_template(template):
            return tokenizer.apply_chat_template(template, tokenize=False)

        def extract_assistant_tokens(full_template, partial_template):
            full_result = apply_chat_template(full_template)
            partial_result = apply_chat_template(partial_template)

            if partial_result not in full_result:
                raise ValueError(
                    "There is some problem with tokenizer , it may from supporting system role. \n It is better to not use system role and check again to occure error or not")

            return full_result.replace(partial_result, "").split("!!!")[0]

        if chat_messages[0].get("role") == "system":
            full_template = [
                {"role": "system", "content": "---"},
                {"role": "user", "content": "###"},
                {"role": "assistant", "content": "!!!"}
            ]
            partial_template = [
                {"role": "system", "content": "---"},
                {"role": "user", "content": "###"}
            ]
            return extract_assistant_tokens(full_template, partial_template)

        elif chat_messages[0].get("role") == "user":
            full_template = [
                {"role": "user", "content": "###"},
                {"role": "assistant", "content": "!!!"}
            ]
            partial_template = [
                {"role": "user", "content": "###"}
            ]
            return extract_assistant_tokens(full_template, partial_template)

        else:
            raise ValueError(f"Unrecognized role: {chat_messages[0].get('role')}")

    def apply_chat_template(self,
                            messages: str,
                            tokenizer,
                            output_force: str = None):
        # tokenizer = self.get_tokenizer(model_name)
        chat_template = tokenizer.apply_chat_template(messages, tokenize=False)
        if output_force is not None:
            output_starter = PromptConstructor.get_start_of_assistant(tokenizer, messages)
            chat_template += output_starter
            chat_template += output_force
        return chat_template

# %% [cell 18: code]
class DatasetHandler:
    def __init__(
        self,
        path: str,
        dataset_type: str,  # "train", "valid", or "test"
        file_type: str = "csv",
        transcript_column: str = "transcript",
        output_column: str = "text", #used in the trainer class
        label_column: str = "label",
        map_labels: bool = False, #For validation and train, set True
        mapping_dictionary: dict = None,
        prompt_to_use: list = None,
        system_prompt_to_use: str = "",
        tokenizer=None,
    ):
        self.path = path
        self.dataset_type = dataset_type.lower()
        self.file_type = file_type
        self.transcript_column = transcript_column
        self.output_column = output_column
        self.label_column = label_column
        self.map_labels = map_labels
        self.mapping_dictionary = mapping_dictionary
        self.prompt_to_use = prompt_to_use or []
        self.system_prompt_to_use = system_prompt_to_use
        self.tokenizer = tokenizer

        self.dataset = None
        self.load_and_process_dataset()

    def load_dataset(self):
        """
        Load dataset from an Excel or CSV file.
        """
        if self.file_type.lower() == "csv":
            df = pd.read_csv(self.path)
        elif self.file_type.lower() == "excel":
            df = pd.read_excel(self.path)
        else:
            raise ValueError("Invalid file type. Use 'csv' or 'excel'.")

        # Rename transcript column for consistency
        df = df.rename(columns={self.transcript_column: "transcript"})
        return df

    def preprocess_data(self, df):
        """
        Apply random prompt to the 'transcript' column and map 'label' column if required.
        The same prompt is used for all data instances as there is only one specified. 
        """

        def apply_prompt(row):
            if self.prompt_to_use:
                prompt = random.choice(self.prompt_to_use)
                row["instruction"] = prompt.format(text=row["transcript"])
            if self.map_labels and self.mapping_dictionary:
                row[self.label_column] = self.mapping_dictionary.get(row[self.label_column], row[self.label_column])
            return row

        return df.apply(apply_prompt, axis=1)

    def format_chat_template(self, row):
        if self.tokenizer.chat_template:
            """
            Apply chat template formatting based on dataset type.
            """
            messages = [
                # {"role": "system", "content": self.system_prompt_to_use},
                        {"role": "user", "content": row["instruction"]}]
    
            if self.dataset_type != "test":
                add_generation_prompt = False #results no difference in Llama Chat format (assistant's answer comes immediately after the user's)
                messages.append({"role": "assistant", "content": "\nLabel: '"+str(row[self.label_column])+"'"})
                row[self.output_column] = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt) if self.tokenizer else messages
            else:
                # This is written to force models to provide a relevant answer for the prompt.
                add_generation_prompt = True
                prompt_constructor = PromptConstructor()
                prompt = prompt_constructor.apply_chat_template(messages=messages, tokenizer=self.tokenizer, output_force="\nLabel: '" )
                row[self.output_column] = prompt
    
            return row

        else:
            prompt = f"""
{classification_system_prompt}

### Instruction:
{classification_task_prompt_single}

### Input:
{row['transcript']}

### Label: '"""
            if self.dataset_type != "test":
                prompt += row['label']+"'"
            row[self.output_column] = prompt        
            return row

    def convert_to_huggingface_dataset(self, df):
        """
        Convert a Pandas DataFrame to a Hugging Face Dataset.
        """
        return Dataset.from_pandas(df)

    def load_and_process_dataset(self):
        """
        Loads, preprocesses, and formats the dataset for model training/evaluation.
        """
        df = self.load_dataset()
        df = self.preprocess_data(df)
        dataset = self.convert_to_huggingface_dataset(df)
        dataset = dataset.map(self.format_chat_template, num_proc=4)

        self.dataset = dataset

    def get_dataset(self):
        """
        Returns the processed dataset.
        """
        self.dataset = self.dataset.rename_column("label", "gt_label")
        return self.dataset

# %% [cell 19: markdown]
# # ***Trainer***

# %% [cell 20: markdown]
# ## ***Callback***

# %% [cell 21: code]
class PushToHubCallback(TrainerCallback):
    def __init__(self, base_model, trainer_handler, train_dataset, valid_dataset, test_dataset, tokenizer, prompts,
                 output_dir="trained_models", model_par_name='', organization=None):
        self.trainer_handler = trainer_handler
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.healthy_token_id, self.healthy_token = self.token_id('Healthy')
        self.ad_token_id, self.ad_token = self.token_id('ADRD')
        self.prompts = prompts
        self.output_dir = output_dir
        self.organization = organization #used for pushing models to hub
        self.model_par_name = model_par_name #model name indicating parameters used for finetuning
        # self.save_prob = save_prob
        self.base_model = base_model

    def token_id(self, label='Healthy'):
        # Tokenize the sentence
        label_tokens = self.tokenizer.tokenize(label)
        # Convert tokens to input IDs
        label_input_ids = self.tokenizer.convert_tokens_to_ids(label_tokens)
        lebel_token_id = label_input_ids[0]
        label_token = self.tokenizer.decode([lebel_token_id])
        print(f'*******************{label}: {lebel_token_id}\ntoken: {label_token}')
        return lebel_token_id, label_token

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """
        Push the trained model to Hugging Face Hub and evaluate on validation dataset after each epoch.
        """
        epoch = str(int(state.epoch))
        model_name = f"ad-{self.model_par_name}_num_epoch_{epoch}"

        if model is not None:
            print(f"Pushing the model to the Hugging Face Hub at {model_name}...")
            # model.save_pretrained(f"{self.output_dir}/{model_name}")
        else:
            raise Exception("Error in saving model")

        print("Model saved in model directory!")

        # Evaluate validation dataset (ONLY THIS ONE IS EVALUATED)
        self.run_model_inference(model_name, model, self.valid_dataset, "validation", epoch, evaluate=True)
        torch.cuda.empty_cache()  # Free unused memory
        gc.collect()
        # Save predictions for train and test datasets (NO EVALUATION)
        self.run_model_inference(model_name, model, self.train_dataset, "train", epoch, evaluate=False)
        torch.cuda.empty_cache()  # Free unused memory
        gc.collect()
        self.run_model_inference(model_name, model, self.test_dataset, "test", epoch, evaluate=False)
        torch.cuda.empty_cache()  # Free unused memory
        gc.collect()

        model.train()
        # model = FastLanguageModel.for_training(model)

    def run_model_inference(self, model_name, model, dataset, dataset_type, epoch, evaluate=False):
        """
        Perform inference using the trained model and evaluate performance on the validation dataset.
        Saves model outputs for all datasets.
        """
        print(f"Running inference for model: {model_name} on {dataset_type} dataset")
        model.eval()
        
        # Load dataset into Pandas DataFrame
        data = dataset.to_pandas()
        # merged_model = PeftModel.from_pretrained(self.base_model, f"{self.output_dir}/{model_name}").to("cuda")#.merge_and_unload()
        for prompt_template in self.prompts:

            # ⏳ Start evaluation timer
            start_time = time.time()

            data, pred_texts = self.generate_predictions_and_compute_probabilities(model, dataset, dataset_type, epoch)
            torch.cuda.empty_cache()
            gc.collect()
            # Process predictions
            data = self.process_predictions(data, pred_texts, dataset_type)

            # ⏳ Stop evaluation timer
            end_time = time.time()
            evaluation_time_seconds = end_time - start_time
            evaluation_time_str = str(datetime.timedelta(seconds=int(evaluation_time_seconds)))

            # Save results to CSV with evaluation time (only for validation set)
            if evaluate:
                if int(epoch) > 2:
                    evaluation_results = self.evaluate_predictions(data, "gt_label", "pred_label_mapped")
                    self.save_evaluation_results(evaluation_results, model_name, dataset_type, evaluation_time_str)

            # Save model outputs for this dataset
            self.save_model_outputs(data, model_name, dataset_type, epoch)
            del pred_texts 

    def generate_predictions_and_compute_probabilities(self, model, dataset, dataset_type, epoch):
        """
        Generates predictions and computes probabilities for specific tokens (healthy, ad)
        in a single pass by reusing tokenization and model generation.
        Saves both predictions and probabilities in the dataset.

        Args:
        - model: LoRA-trained model
        - tokenizer: Tokenizer for the model
        - dataset: Pandas DataFrame containing input text
        - dataset_type: Indicates if dataset is train/validation/test
        - epoch: Current epoch number

        Returns:
        - DataFrame with predictions and probabilities for the selected tokens.
        """
        print(f"Generating predictions and computing probabilities for {dataset_type} dataset on {epoch}")

        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            pred_texts = []
            prob_healthy_values = []
            prob_ad_values = []
            dataset = dataset.to_pandas()
        
            for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
                # Tokenization (shared between prediction and probability computation)
                input_ids = self.tokenizer(
                    row["text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to("cuda")

                # Generate Predictions and Get Logits in One Step
                trial = 1
                pred_text = None
                max_tokens_limit = 0
                while (self.map_label(pred_text) is None or pd.isna(self.map_label(pred_text))) and trial <= 20:
                    if trial > 1:
                        del outputs
                        del generated_sequences
                    if trial%7 == 0:
                        max_tokens_limit +=1
                        # print(pred_text)
                    outputs = model.generate(
                        input_ids['input_ids'],
                        attention_mask=input_ids['attention_mask'],
                        max_new_tokens=1+max_tokens_limit,
                        return_dict_in_generate=True,
                        output_scores=True,
                        do_sample=False,  # Greedy decoding
                        num_beams=1
                    )

                    # Decode generated text
                    generated_sequences = outputs.sequences.cpu()
                    generated_text = tokenizer.decode(generated_sequences[0], skip_special_tokens=True)
                    pred_text = generated_text.split("Label:")[-1]
                    trial += 1

                if trial > 20 and self.map_label(pred_text) is None:
                    print(":(  ",pred_text)

                pred_texts.append(pred_text)

                # Compute Probabilities using the SAME OUTPUTS (No extra generation)
                logits_step = outputs.scores[-1].cpu()  # Logits for the last token

                selected_logits = torch.tensor([
                    logits_step[0, self.healthy_token_id],
                    logits_step[0, self.ad_token_id]
                ])

                selected_probs = torch.softmax(selected_logits, dim=0).cpu()

                prob_healthy_values.append(selected_probs[0].item())
                prob_ad_values.append(selected_probs[1].item())

                # ✅ Move tensors to CPU before deletion
                input_ids = input_ids.to("cpu")
                outputs = None
                logits_step = None
                selected_logits = None
                selected_probs = None
                del input_ids, outputs, logits_step, selected_logits, selected_probs
                torch.cuda.empty_cache()
                gc.collect()
                
        # Add predictions and probabilities to the dataset
        dataset[f"pred_text_epoch_{epoch}"] = pred_texts
        dataset[f"prob_healthy_epoch_{epoch}"] = prob_healthy_values
        dataset[f"prob_ad_epoch_{epoch}"] = prob_ad_values

        return dataset, pred_texts

    def process_predictions(self, data, pred_texts, dataset_type):
        """
        Saves full model outputs in pred_text and maps labels only for validation.
        """
        data["pred_text"] = pred_texts

        if dataset_type == "validation":
            # Only map labels for validation
            data["pred_label_mapped"] = data["pred_text"].apply(self.map_label)
            mapping = {'AD': 1, 'Healthy': 0, None:None}
            data["pred_label_mapped"] = data["pred_label_mapped"].map(mapping)
            print(data["pred_label_mapped"].value_counts())

        return data

    def evaluate_predictions(self, data, true_label_col, pred_col):
        """
        Compute evaluation metrics for validation predictions.
        """
        y_pred = data[pred_col]

        if y_pred.isna().any():
            print("NaN values...")
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            tn, fp, fn, tp =0.0, 0.0, 0.0, 0.0

        else:
            y_true = data[true_label_col].astype(int)
            y_pred = data[pred_col].astype(int)


            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        print(f"{pred_col} Evaluation:")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"True Positives: {tp}, False Positives: {fp}, True Negatives: {tn}, False Negatives: {fn}")
        print("\n")

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "tn": tn, "fp": fp, "fn": fn, "tp": tp}

    def save_evaluation_results(self, evaluation_results, model_name, dataset_type, evaluation_time_str):
        """
        Save evaluation metrics to a CSV file.
        """
        log_file = os.path.join(self.output_dir, f"evaluation_results.csv")

        results_df = pd.DataFrame([{
            **evaluation_results,
            "model_name": model_name,
            "dataset": dataset_type,
            "evaluation_time": evaluation_time_str,
            "date_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])

        if os.path.exists(log_file):
            results_df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            results_df.to_csv(log_file, index=False)

        print(f"Evaluation results saved to {log_file}")

    def save_model_outputs(self, data, model_name, dataset_type, epoch):
        """
        Save model responses for each dataset, appending to the respective file.
        """
        model_name = model_name.split('_num_epoch_')[0]
        output_file = os.path.join(self.output_dir, f"{model_name}_outputs_{dataset_type}.csv")

        # Add column with model predictions for this epoch
        epoch_col_name = f"pred_text_epoch_{epoch}"
        epoch_prob1_col_name = f"prob_healthy_epoch_{epoch}"
        epoch_prob2_col_name = f"prob_ad_epoch_{epoch}"
        data[epoch_col_name] = data["pred_text"]

        # Save or update CSV file
        if os.path.exists(output_file):
            existing_data = pd.read_csv(output_file)
            updated_data = existing_data.merge(data[["text", epoch_col_name, epoch_prob1_col_name, epoch_prob2_col_name]], on="text", how="left")
            updated_data.to_csv(output_file, index=False)
        else:
            data.to_csv(output_file, index=False)

        print(f"Model outputs saved for {dataset_type} dataset at {output_file}")


    def map_label(self,label): # tokenize the labels and take the first token
        """
        Maps the output text to AD or Healthy.
        """
        if label:

            if self.healthy_token.lower() in label.lower() or 'he' in label.lower():
                return 'Healthy'
            elif self.ad_token.lower() in label.lower():
                return 'AD'
            else:
                return None
        else:
            return None

# %% [cell 22: markdown]
# ## ***Trainer***

# %% [cell 23: code]
class TrainerHandler:
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        tokenizer,
        peft_config=None,
        output_dir="tuned_llama1B",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=5,
        eval_strategy="steps",
        eval_steps=0.5,
        logging_steps=1,
        warmup_ratio=0.03,
        lr_scheduler_type ='cosine',
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        group_by_length=True,
        packing=False,
        max_seq_length=1024,
        dataset_text_field="text",
        report_to="none" #"wandb",  # Change to "none" if you don't want logging
    ):
        """
        Initializes the trainer configuration and prepares the trainer object.

        Args:
            model: The model to fine-tune.
            train_dataset: Hugging Face dataset for training.
            eval_dataset: Hugging Face dataset for evaluation.
            tokenizer: Tokenizer used for training.
            peft_config: Optional PEFT configuration for LoRA or other parameter-efficient tuning methods.
            Other hyperparameters

        for the trainer are specified with default values.
        """
        self.model = model
        self.output_dir = output_dir
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.peft_config = peft_config

        self.kwargs = {
            'per_device_train_batch_size':per_device_train_batch_size,
            'gradient_accumulation_steps':gradient_accumulation_steps,
            'optim':optim,
            'num_train_epochs':num_train_epochs,
            'learning_rate':learning_rate,
            'max_seq_length':max_seq_length,
            'lora_rank':peft_config.r,
            'lora_alpha':peft_config.lora_alpha,
            'lora_dropout':peft_config.lora_dropout,
            # 'lora_modules':peft_config.target_modules,
        }
        # Define training arguments
        self.training_arguments = SFTConfig(
            report_to=report_to,
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            num_train_epochs=num_train_epochs,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            # warmup_steps=warmup_steps,
            lr_scheduler_type='cosine',
            warmup_ratio=0.03,
            logging_strategy=logging_strategy,
            learning_rate=learning_rate,
            fp16=fp16,
            bf16=bf16,
            group_by_length=group_by_length,
            packing=packing,
            
            dataset_text_field=dataset_text_field,
        )

        # Initialize trainer
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            peft_config=self.peft_config,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
        )
        

    def train(self):
        """
        Trains the model using the trainer.
        """
        start_time = time.time()  # Start timer

        train_result = self.trainer.train()  # Train model

        end_time = time.time()  # End timer

        # Calculate training duration
        training_time_seconds = end_time - start_time
        training_time_str = str(datetime.timedelta(seconds=int(training_time_seconds)))  # Convert to HH:MM:SS format

        loss = train_result.training_loss
        print("Total Train Time: ", training_time_str)

        # Log training details with time
        self._log_training_details(loss, training_time_str)


    def evaluate(self, test_dataset):
        """
        Evaluates the model on a test dataset.

        Args:
            test_dataset: Hugging Face dataset for testing.

        Returns:
            List of predictions.
        """
        predictions = []

        # Ensure model is in evaluation mode
        self.model.eval()

        for sample in tqdm(test_dataset, total=len(test_dataset)):
            prompt = self.tokenizer.apply_chat_template(
                sample["text"], tokenize=False, add_generation_prompt=True
            )

            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            ).to("cuda")

            outputs = self.model.generate(
                **inputs, max_new_tokens=150, num_return_sequences=1, do_sample=True, temperature=0.1
            )

            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(text)

        return predictions

    def save_model(self):
        model_name = self._generate_model_name()
        save_path = os.path.join(self.output_dir, model_name)
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

    def _generate_model_name(self):
        params = [
            f"bs{self.kwargs.get('per_device_train_batch_size', 1)}",
            f"ga{self.kwargs.get('gradient_accumulation_steps', 1)}",
            f"optim_{self.kwargs.get('optim', 'adamw')}",
            f"epochs{self.kwargs.get('num_train_epochs', 1)}",
            f"lr{self.kwargs.get('learning_rate', 2e-5)}",
            f"fp16{self.kwargs.get('fp16', False)}",
            f"lora{self.kwargs.get('lora_rank', 8)}",
            f"quant{self.kwargs.get('load_quantized', 4)}",
        ]
        return "_".join(params)

    def push_to_hub(self, repo_name='', organization=None):
        """
        Pushes the trained model to the Hugging Face Hub.

        Args:
            repo_name (str): Name of the repository to push to.
            organization (str, optional): Organization name if pushing to a team account.
        """
        if len(repo_name)<2:
            repo_name = self._generate_model_name()
        if organization:
            repo_name = f"{organization}/{repo_name}"

        self.model.push_to_hub(repo_name)
        self.tokenizer.push_to_hub(repo_name)
        print(f"Model pushed to Hugging Face Hub at: {repo_name}")

    def _log_training_details(self, loss, training_time):
        """
        Saves all training configurations, hyperparameters, model name,
        date, time, and training loss to a JSON log file.
        """
        log_data = {
            "model_name": self._generate_model_name(),
            "date_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hyperparameters": self.kwargs,
            "training_loss": loss,
            "training_time": training_time  # Add training duration
        }

        log_file = os.path.join(self.output_dir, "training_logs.json")

        # Append log entry to JSON file
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(log_data)

        with open(log_file, "w") as f:
            json.dump(logs, f, indent=4)

        print(f"Training details logged in {log_file}")

# %% [cell 24: markdown]
# # ***Component 3: (1)	Token-Level Supervised Fine-Tuning***

# %% [cell 25: markdown]
# ## ***Loop Through Hyperparameters***

# %% [cell 26: markdown]
# This cell performs a **hyperparameter search and training loop** for a language model with **LoRA fine-tuning** and optional **quantization**.  
# 
# For each variation of training and model parameters:  
# 1. **Model & Tokenizer Initialization** – Loads the base model with specified LoRA and quantization settings.  
# 2. **Dataset Loading** – Prepares training, validation, and test datasets using `DatasetHandler`.  
# 3. **Training** – Runs fine-tuning with `TrainerHandler` and optionally pushes results to a model hub.  
# 4. **Resource Cleanup** – Frees CUDA memory and clears cache before moving to the next run.  
# 
# The design ensures efficient GPU usage while systematically evaluating multiple hyperparameter configurations, making it easy to compare model performance across different settings.  

# %% [cell 27: code]
# Define hyperparameter search space
# Each entry contains: [training_params, model_params]
hyperparameter_variations = [
    # Uncomment variations as needed
    # [{"per_device_train_batch_size":1, "gradient_accumulation_steps":4,
    #   "num_train_epochs":13, "learning_rate":2e-4,},
    #  {"load_quantized": 4, "lora_dropout": 0.1, "lora_rank": 32}],
    
    [{"per_device_train_batch_size":1, "gradient_accumulation_steps":4,
      "num_train_epochs":13, "learning_rate":2e-4,},
     {"load_quantized": 4, "lora_dropout": 0.1, "lora_rank": 64}],
]

# Default hyperparameters (used as baseline for naming and comparison)
default_params = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 2,
    "learning_rate": 2e-4,
    "optim": "paged_adamw_32bit",
    "fp16": True,
    "load_quantized": None,
    "lora_rank": 8,
    "lora_dropout": 1,
}

# Detect device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Loop over hyperparameter variations
for params in hyperparameter_variations:
    print(f"Training with parameters: {params}")

    training_params, model_params = params[0], params[1]

    # Identify changed hyperparameters for logging/naming
    changed_params = {
        k: v for k, v in {**training_params, **model_params}.items()
        if v != default_params.get(k)
    }
    model_name = "llama3.3_70B_ad_" + "_".join([f"{k}{v}" for k, v in changed_params.items()])
    print(f"🔹 Model Name: {model_name}")

    # ------------------------------
    # 1. Load Model & Tokenizer
    # ------------------------------
    model_handler = ModelHandler(
        base_model='/workspace/models/llama3.3_70B_instruct',
        device_map=device,
        tokenizer_trust_remote_code=True,
        use_lora=True,
        **model_params  # LoRA & Quantization parameters
    )
    model, tokenizer = model_handler.get_model_and_tokenizer()
    peft_config = model_handler.get_peft_config()
    print(peft_config)
    print('Tokenizer chat template available:', bool(tokenizer.chat_template))

    # ------------------------------
    # 2. Load Datasets
    # ------------------------------
    # Training dataset
    train_handler = DatasetHandler(
        path=train_dataset_path,
        dataset_type="train",
        file_type="csv",
        transcript_column="text",
        label_column="label",
        map_labels=True,
        mapping_dictionary=label_mapping_dict,
        prompt_to_use=classification_task_prompts,
        tokenizer=tokenizer,
    )
    train_dataset = train_handler.get_dataset()

    # Validation dataset
    valid_handler = DatasetHandler(
        path=valid_dataset_path,
        dataset_type="valid",
        file_type="csv",
        transcript_column="text",
        label_column="label",
        map_labels=True,
        mapping_dictionary=label_mapping_dict,
        prompt_to_use=classification_task_prompts,
        tokenizer=tokenizer,
    )
    valid_dataset = valid_handler.get_dataset()

    # Train/valid datasets for evaluation (labels not mapped)
    train_eval_dataset = DatasetHandler(
        path=train_dataset_path,
        dataset_type="test",
        file_type="csv",
        transcript_column="text",
        label_column="label",
        map_labels=False,
        mapping_dictionary=None,
        prompt_to_use=classification_task_prompts,
        tokenizer=tokenizer,
    ).get_dataset()

    valid_eval_dataset = DatasetHandler(
        path=valid_dataset_path,
        dataset_type="test",
        file_type="csv",
        transcript_column="text",
        label_column="label",
        map_labels=False,
        mapping_dictionary=None,
        prompt_to_use=classification_task_prompts,
        tokenizer=tokenizer,
    ).get_dataset()

    # Test dataset (Excel format)
    test_dataset = DatasetHandler(
        path=test_dataset_path,
        dataset_type="test",
        file_type="excel",
        transcript_column="text",
        label_column="label",
        map_labels=False,
        prompt_to_use=classification_task_prompts,
        tokenizer=tokenizer,
    ).get_dataset()

    # ------------------------------
    # 3. Train Model
    # ------------------------------
    trainer_handler = TrainerHandler(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        output_dir=f"/workspace/trained_models/{model_name}",  # Save with unique name
        **training_params
    )

    # Push to hub after training
    trainer_handler.trainer.add_callback(
        PushToHubCallback(
            base_model=model,
            trainer_handler=trainer_handler,
            train_dataset=train_eval_dataset,
            valid_dataset=valid_eval_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            prompts=classification_task_prompts,
            output_dir="/workspace/trained_models",
            model_par_name=model_name,
            organization="speechCare",
        )
    )

    results = trainer_handler.train()

    # ------------------------------
    # 4. Cleanup CUDA memory
    # ------------------------------
    print(f"🔹 Cleaning up CUDA memory for {model_name}")

    # Free objects from GPU memory
    del model, tokenizer, peft_config, trainer_handler
    del train_dataset, valid_dataset, test_dataset
    del model_handler, train_handler, valid_handler

    # Garbage collection + empty CUDA cache
    gc.collect()
    torch.cuda.empty_cache()

    # Finish wandb run if enabled
    # wandb.finish()
