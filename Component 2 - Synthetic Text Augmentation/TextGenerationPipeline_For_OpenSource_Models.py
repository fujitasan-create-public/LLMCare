# Auto-converted from notebook

# %% [cell 1: markdown]
# # ***Install Libraries***

# %% [cell 2: markdown]
# ## Install Required Libraries
# This cell installs the main Hugging Face and training-related libraries:
# - `transformers`: for model loading and text generation  
# - `datasets`: for handling datasets  
# - `accelerate`: for efficient distributed training  
# - `peft`: parameter-efficient fine-tuning methods  
# - `trl`: reinforcement learning with human feedback  
# - `bitsandbytes`: low-precision optimization  
# - `evaluate`: evaluation utilities  
# 
# Run this cell once at the start of your environment setup.

# %% [cell 3: code]
%pip install -U transformers==4.48.0
%pip install -U datasets
%pip install -U accelerate
%pip install 'accelerate>=0.26.0'
%pip install -U peft
%pip install -U trl
%pip install -U bitsandbytes
%pip install wandb


!pip install openpyxl
!pip install typing_extensions==4.11.0
!pip install tiktoken
!pip install protobuf
!pip install sentencepiece

# %% [cell 4: markdown]
# ## Install Additional Dependencies
# This cell installs supporting libraries:  
# - `huggingface_hub`: to interact with Hugging Face Hub  
# - `scikit-learn`: for evaluation and preprocessing  
# - `flash-attn`: optimized attention kernels for faster training  
# - `vllm`: high-performance inference for large language models  

# %% [cell 5: code]
!pip install -U huggingface_hub
!pip install scikit-learn
!pip install flash-attn --no-build-isolation
!pip install vllm

# %% [cell 6: markdown]
# # ***Import Libraries***

# %% [cell 7: code]
import transformers
import torch

import os
import gc
import json
import random
import datetime
import pandas as pd
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from datasets import Dataset, load_dataset

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
    prepare_model_for_kbit_training,
    get_peft_model,
)

# %% [cell 8: markdown]
# ## Configure Model Precision and Attention
# This cell sets key configuration values:  
# - `torch_dtype = torch.float16` → use half precision for efficiency  
# - `attn_implementation = "eager"` → attention implementation mode  

# %% [cell 9: code]
torch_dtype = torch.float16
attn_implementation = "eager"

# %% [cell 10: markdown]
# ## Configure Weights & Biases
# This cell initializes [Weights & Biases](https://wandb.ai/) logging.  
# Replace `'YOUR_wandb_KEY'` with your API key to track experiments.  

# %% [cell 11: code]
import wandb
wandb.login(key='YOUR_wandb_KEY')

# %% [cell 12: markdown]
# ## Configure Logging
# This cell sets the logging verbosity for Hugging Face Transformers.  
# Warnings will be displayed to help debug model usage.  

# %% [cell 13: code]
from transformers import logging

logging.set_verbosity_warning()

# %% [cell 14: markdown]
# ## Login to Hugging Face Hub
# This cell logs into Hugging Face Hub using a token.  
# Replace `'huggingface_hub_TOKEN'` with your personal access token.  

# %% [cell 15: code]
from huggingface_hub import login
login(token="huggingface_hub_TOKEN")

# %% [cell 16: markdown]
# # ***Constant Variables and Paths***

# %% [cell 17: markdown]
# This cell specifies directories for saving datasets, predictions and prompts:  
# - Training, validation, and test dataset paths  
# - Root directory for workspace  

# %% [cell 18: code]
root_dir = '/workspace/'
data_dir = root_dir
pred_dir = root_dir + 'predictions/'
train_dataset_path = data_dir + 'train.csv'
valid_dataset_path = data_dir + 'validation.csv'
test_dataset_path = data_dir + 'Test_DePiC.xlsx'


label_mapping_dict = {0: "Healthy", 1: "ADRD"}

# System instruction for chat template
generation_system_prompt = """You are an expert cognitive impairment analyst.
Your role is to generate spoken language transcripts based on linguistic patterns.
"""
generation_task_prompts =[ (
"As a language and cognition specialist, generate a realistic spoken monologue of someone describing the “Cookie Theft” image."
"\nHealthy: Include advanced sentence structures, precise vocabulary, and an organized depiction of the scene."
"\nADRD: Include repeated segments, stumbling or halting speech, misplaced words, and sentence fragments."
"\nLabel: {label}"
"\ntext:"
),
("You are a neurocognitive researcher studying everyday speech. Craft a spoken-style transcript of a person talking about the “Cookie Theft” image."
"\nHealthy: Show natural fluency, clear reference to the main elements in the picture, and cohesive transitions."
"\nADRD: Show echoes of previous statements, grammatical mishaps, filler words, and abrupt topic shifts."
"\nLabel: {label}"
"\ntext:"
),
("You are an expert in cognitive assessments for older adults. Provide a natural, conversational transcript of a person describing the “Cookie Theft” picture."
"\nHealthy: Use elaborate syntax, coherent progress from one detail to another, and minimal disfluencies."
"\nADRD: Use frequent filler words (“you know,” “like”), disjointed or incomplete clauses, and noticeable grammatical errors."
"\nLabel: {label}"
"\ntext:"
),
(
"As a specialist in cognitive health and communication, produce a brief, spoken-style transcript of someone describing the “Cookie Theft” image."
"\nHealthy: Expect detailed observation, fluid speech, and well-formed sentences."
"\nADRD: Expect word-finding pauses, repetition of concepts, grammatical inconsistencies, and less organized content."
"\nLabel: {label}"
"\ntext:"
),
(
"You are an advanced language model trained in speech analysis for cognitive health. Create a spontaneous-soundingexplanation of the “Cookie Theft” picture."
"\nHealthy: Demonstrate complex grammatical structures, coherent narrative flow, and smooth connectivity."
"\nADRD: Demonstrate repeated attempts at words, filler phrases, sentence fragments, and reduced coherence"
"\nLabel: {label}"
"\ntext:"
),
(
"As a researcher in cognitive-linguistic assessment, generate a spoken language transcript of an individual describing the “Cookie Theft” image. Keep it natural and unrehearsed."
"\nHealthy: Incorporate sophisticated syntax, purposeful word choice, and a clear storyline."
"\nADRD: Include repetitions, stumbling over words, run-on or abruptly cut-off sentences, and difficulty finding the right words."
"\nLabel: {label}"
"\ntext:"
),
(
"You are an expert in geriatric neuropsychology. Produce a short, speech-like narration of a person describing the “Cookie Theft” picture."
"\nHealthy: Look for varied vocabulary, coherent transitions, and overall fluency."
"\nADRD: Capture frequent pauses, filler utterances (“um,” “uh”), grammatical mistakes, and incomplete thoughts."
"\nLabel: {label}"
"\ntext:"
),
(
"Act as a clinician studying language use in older adults. Generate a spoken transcript of someone describing the “Cookie Theft” scenario as if they’re talking naturally (not reading prepared text)."
"\nHealthy: Emphasize complex syntax, detailed description, and logical flow."
"\nADRD: Emphasize repeated words, hesitations, grammar errors, and disjointed phrases."
"\nLabel: {label}"
"\ntext:"
),
(
"You are a specialized speech-language pathologist focusing on cognitive health. Please create a short, spontaneous-sounding transcript of an individual describing the “Cookie Theft” picture."
"\nFor Healthy: Observe intricate sentence structure, clear semantics, fluent delivery, and coherent storytelling."
"\nFor ADRD: Pay attention to repeating phrases, filler words, noticeable grammatical slips, fragmented sentences, and disfluencies."
"\nLabel: {label}"
"\ntext:"
),
(
"You are a recognized expert in geriatric language assessment. Create a short, unpolished spoken transcript of a person explaining what they see in the “Cookie Theft” image."
"\nHealthy: Characterize fluid sentences, organized thoughts, and diverse vocabulary."
"\nADRD: Characterize repeated or circular phrasing, noticeable disfluencies, incomplete ideas, and filler expressions"
"\nLabel: {label}"
"\ntext:"
)
]

Inference_prompts = [
    ("You are an expert in cognitive health and language analysis. You will generate a spoken language transcript of a person describing the 'Cookie Theft' picture. This should reflect spontaneous speech rather than formal written text. Generate a text based on the given label."
    "\nLabel: {label}"
     "\ntext:"
)
]

generation_task_prompts_no_chat_template =[ (
'''As a language and cognition specialist, generate a realistic spoken monologue of someone describing the “Cookie Theft” image.
Healthy: Include advanced sentence structures, precise vocabulary, and an organized depiction of the scene.
ADRD: Include repeated segments, stumbling or halting speech, misplaced words, and sentence fragments.'''
),
('''You are a neurocognitive researcher studying everyday speech. Craft a spoken-style transcript of a person talking about the “Cookie Theft” image.
Healthy: Show natural fluency, clear reference to the main elements in the picture, and cohesive transitions.
ADRD: Show echoes of previous statements, grammatical mishaps, filler words, and abrupt topic shifts.'''
),
('''You are an expert in cognitive assessments for older adults. Provide a natural, conversational transcript of a person describing the “Cookie Theft” picture.
Healthy: Use elaborate syntax, coherent progress from one detail to another, and minimal disfluencies.
ADRD: Use frequent filler words (“you know,” “like”), disjointed or incomplete clauses, and noticeable grammatical errors.'''
),
(
'''As a specialist in cognitive health and communication, produce a brief, spoken-style transcript of someone describing the “Cookie Theft” image.
Healthy: Expect detailed observation, fluid speech, and well-formed sentences.
ADRD: Expect word-finding pauses, repetition of concepts, grammatical inconsistencies, and less organized content.'''
),
(
'''You are an advanced language model trained in speech analysis for cognitive health. Create a spontaneous-soundingexplanation of the “Cookie Theft” picture.
Healthy: Demonstrate complex grammatical structures, coherent narrative flow, and smooth connectivity.
ADRD: Demonstrate repeated attempts at words, filler phrases, sentence fragments, and reduced coherence'''
),
(
'''As a researcher in cognitive-linguistic assessment, generate a spoken language transcript of an individual describing the “Cookie Theft” image. Keep it natural and unrehearsed.
\nHealthy: Incorporate sophisticated syntax, purposeful word choice, and a clear storyline.
\nADRD: Include repetitions, stumbling over words, run-on or abruptly cut-off sentences, and difficulty finding the right words.'''
),
(
'''You are an expert in geriatric neuropsychology. Produce a short, speech-like narration of a person describing the “Cookie Theft” picture.
Healthy: Look for varied vocabulary, coherent transitions, and overall fluency.
ADRD: Capture frequent pauses, filler utterances (“um,” “uh”), grammatical mistakes, and incomplete thoughts.'''
),
(
'''Act as a clinician studying language use in older adults. Generate a spoken transcript of someone describing the “Cookie Theft” scenario as if they’re talking naturally (not reading prepared text).
Healthy: Emphasize complex syntax, detailed description, and logical flow.
ADRD: Emphasize repeated words, hesitations, grammar errors, and disjointed phrases.'''
),
(
'''You are a specialized speech-language pathologist focusing on cognitive health. Please create a short, spontaneous-sounding transcript of an individual describing the “Cookie Theft” picture.
For Healthy: Observe intricate sentence structure, clear semantics, fluent delivery, and coherent storytelling.
For ADRD: Pay attention to repeating phrases, filler words, noticeable grammatical slips, fragmented sentences, and disfluencies.'''
),
(
'''You are a recognized expert in geriatric language assessment. Create a short, unpolished spoken transcript of a person explaining what they see in the “Cookie Theft” image.
Healthy: Characterize fluid sentences, organized thoughts, and diverse vocabulary.
ADRD: Characterize repeated or circular phrasing, noticeable disfluencies, incomplete ideas, and filler expressions'''
)
]

# %% [cell 19: markdown]
# # ***Model***

# %% [cell 20: markdown]
# This cell defines a `ModelHandler` class that:  
# - Loads a causal language model and tokenizer  
# - Applies LoRA (Low-Rank Adaptation) using PEFT  
# - Supports mixed precision with bitsandbytes  
# 
# This encapsulates all model-related setup.  

# %% [cell 21: code]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb


class ModelHandler:
    def __init__(
        self,
        base_model: str,
        load_quantized: int = None, #choose values of [None, 4, 8]
        device_map: str = "auto",
        tokenizer_trust_remote_code: bool = True,
        linear_modules=None, #either pass required models in a list or None for all the modules to be considered
        use_lora: bool = True,
        lora_rank: int = 32,
        lora_dropout: float = 0.05,
    ):
        self.base_model = base_model
        self.load_quantized = load_quantized
        self.device_map = device_map
        self.tokenizer_trust_remote_code = tokenizer_trust_remote_code

        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_dropout = lora_dropout

        self.model = None
        self.tokenizer = None
        self.lora_config = None

        # Determine attention implementation and dtype
        self.torch_dtype, self.attn_implementation = self.set_attention_config()

        # self.load_model_and_tokenizer(linear_modules)

    def find_all_linear_names(self):
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in self.model.named_modules():
            # print(name)
            if isinstance(module, cls):
                # print("cls: ", name)
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names:  # needed for 16 bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def get_linear_modules(self):
        modules = self.find_all_linear_names()
        if len(modules) > 1:
            #works when bits&bytes is enabled
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
        Sets attention implementation and dtype based on CUDA device capability.
        Installs FlashAttention if necessary.
        """
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            try:
                # subprocess.run(
                #     ["pip", "install", "-qqq", "flash-attn"], check=True
                # )  # Install FlashAttention
                !pip install -qqq flash-attn
            except:# subprocess.CalledProcessError:
                print("Failed to install flash-attn, falling back to eager attention.")
                return torch.float16, "eager"

            return torch.bfloat16, "flash_attention_2"
        else:
            return torch.float16, "eager"

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
              self.base_model, trust_remote_code=self.tokenizer_trust_remote_code
          )

        return self.tokenizer

    def load_model_and_tokenizer(self, linear_modules):
        """
        Loads the model and tokenizer based on the provided configuration.
        """

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=self.tokenizer_trust_remote_code
        )

        if self.tokenizer.pad_token:
            pass
        else:
            self.tokenizer_pad_token = True
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print('****Tokenizer Loaded****')

        # Handle quantization configuration
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
                quantization_config = BitsAndBytesConfig(load_in_8bit= True)

        # Load model with quantization settings if applicable
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,  # Use determined dtype
            # attn_implementation=self.attn_implementation,  # Use determined attention implementation
            quantization_config=quantization_config#(quantization_config or {}),
        )

        # Apply LoRA if enabled
        self.linear_modules = (
            self.get_linear_modules() if linear_modules is None else linear_modules
        )

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

    def get_model_and_tokenizer(self):
        """
        Returns the loaded model and tokenizer.
        """
        return self.model, self.tokenizer

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model

    def get_peft_config(self):
        return self.lora_config

# %% [cell 22: markdown]
# # ***Dataset***

# %% [cell 23: markdown]
# This cell defines a `DatasetHandler` class that:  
# - Loads datasets from CSV files  
# - Converts them into Hugging Face `Dataset` objects  
# - Handles train/validation/test splits  
# 
# This prepares data for fine-tuning and evaluation.  

# %% [cell 24: code]
import pandas as pd
import random
from datasets import Dataset


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
        """

        def apply_prompt(row):
            if self.prompt_to_use:
                prompt = random.choice(self.prompt_to_use)
                row["instruction"] = prompt.format(label=row["label"])
            if self.map_labels and self.mapping_dictionary:
                row[self.label_column] = self.mapping_dictionary.get(row[self.label_column], row[self.label_column])
            return row

        return df.apply(apply_prompt, axis=1)

    def format_chat_template(self, row):
        """
        Apply chat template formatting based on dataset type.
        """
        if self.tokenizer.chat_template:
            messages = [{"role": "system", "content": self.system_prompt_to_use},
                        {"role": "user", "content": row["instruction"]}]
    
            if self.dataset_type != "test":
                add_generation_prompt = False #results no difference in Llama Chat format (assistant's answer comes immediately after the user's)
                messages.append({"role": "assistant", "content": str(row['transcript'])})
            else:
                add_generation_prompt = True
    
            row[self.output_column] = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt) if self.tokenizer else messages
            return row
        else:
            prompt_single = random.choice(generation_task_prompts_no_chat_template)
            prompt = f"""
{generation_system_prompt}

### Instruction:
{prompt_single}

### Label:
{row['label']}

### text: '"""
            if self.dataset_type != "test":
                prompt += row['transcript']+"'"
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

# %% [cell 25: markdown]
# # ***Trainer***

# %% [cell 26: markdown]
# This cell sets up the training pipeline:  
# - Loads the model and dataset via handler classes  
# - Configures training arguments  
# - Defines evaluation strategy and metrics  
# - Starts fine-tuning using Hugging Face `Trainer`  

# %% [cell 27: code]
import os
import time
import torch
import pandas as pd
from peft import PeftConfig, PeftModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, pipeline, TrainerCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


import datetime
from tqdm import tqdm
from vllm import LLM, SamplingParams


class PushToHubCallback(TrainerCallback):
    def __init__(self, base_model, trainer_handler, train_dataset, valid_dataset, test_dataset, tokenizer, prompts,
                 output_dir="trained_models", model_par_name='', organization=None):
        self.trainer_handler = trainer_handler
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.output_dir = output_dir
        self.organization = organization #used for pushing models to hub
        self.model_par_name = model_par_name #model name indicating parameters used for finetuning
        self.base_model = base_model

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """
        Push the trained model to Hugging Face Hub and evaluate on validation dataset after each epoch.
        """
        epoch = str(int(state.epoch))
        if int(epoch) > 3:
            model_name = f"ad-{self.model_par_name}_num_epoch_{epoch}_loraWeights"
            if model is not None:
                print(f"Pushing the model to the Hugging Face Hub at {model_name}...")
                model.push_to_hub(model_name , token="YOUR_TOKEN")
                self.tokenizer.push_to_hub(model_name , token="YOUR_TOKEN")
                
            else:
                raise Exception("Error in saving model")
        
            print("Model saved in model directory!")

# %% [cell 28: code]
class TrainerHandler:
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        tokenizer,
        peft_config=None,
        output_dir="tuned_medalpaca7B",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=5,
        eval_strategy="steps",
        eval_steps=0.4,
        logging_steps=1,
        warmup_ratio=0.03,
        lr_scheduler_type ='cosine',
        logging_strategy="steps",
        learning_rate=2e-5,
        fp16=False,
        bf16=False,
        group_by_length=True,
        packing=False,
        max_seq_length=1024,
        dataset_text_field="text",
        report_to=None,  # Change to "none" if you don't want logging
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
            max_seq_length=max_seq_length,
            dataset_text_field=dataset_text_field,
            # callbacks=[PushToHubCallback()] ,
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
        # self.trainer.add_callback(PushToHubCallback())

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
            f"quant{self.kwargs.get('load_quantized', 8)}",
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

# %% [cell 29: markdown]
# # ***Main***

# %% [cell 30: code]
# Define hyperparameter search space
hyperparameter_variations = [
    [{"per_device_train_batch_size":2, "gradient_accumulation_steps":2, "num_train_epochs":13, "learning_rate":2e-4,},
     {"load_quantized": None, "lora_dropout": 0.1, "lora_rank": 128}],
]

# Default hyperparameters (for comparison)
default_params = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 0,
    "learning_rate": 2e-4,
    "optim": "paged_adamw_32bit",
    "fp16": True,
    "load_quantized": None,
    "lora_rank": 128,
    "lora_dropout": 0,
}

# %% [cell 31: code]

for params in hyperparameter_variations:
    print(f"Training with parameters: {params}")

    training_params, model_params = params[0], params[1]
    # 🔹 Identify changed hyperparameters and construct a model name
    changed_params = {k: v for k, v in {**training_params, **model_params}.items() if v != default_params.get(k)}
    model_name = "medalpaca" + "_".join([f"{k}{v}" for k, v in changed_params.items()])

    print(f"🔹 Model Name: {model_name}")

    # Load Model and Tokenizer
    model_handler = ModelHandler(
        base_model='medalpaca/medalpaca-7b',
        device_map="auto",
        tokenizer_trust_remote_code=True,
        use_lora=True,
        **model_params  # LoRA & Quantization parameters
    )

    model, tokenizer = model_handler.get_model_and_tokenizer()
    peft_config = model_handler.get_peft_config()

    # Load Datasets
    train_handler = DatasetHandler(
        path=train_dataset_path,
        dataset_type="train",
        file_type="csv",
        transcript_column="text",
        label_column="label",
        map_labels=True,
        mapping_dictionary=label_mapping_dict,
        prompt_to_use=generation_task_prompts
        ,
        tokenizer=tokenizer,
    )
    train_dataset = train_handler.get_dataset()

    valid_handler = DatasetHandler(
        path=valid_dataset_path,
        dataset_type="valid",
        file_type="csv",
        transcript_column="text",
        label_column="label",
        map_labels=True,
        mapping_dictionary=label_mapping_dict,
        prompt_to_use=generation_task_prompts,
        tokenizer=tokenizer,
    )
    valid_dataset = valid_handler.get_dataset()

    train_eval_handler = DatasetHandler(
        path=train_dataset_path,
        dataset_type="test",
        file_type="csv",
        transcript_column="text",
        label_column="label",
        map_labels=False,
        mapping_dictionary=None,
        prompt_to_use=generation_task_prompts,
        tokenizer=tokenizer,
    )
    train_eval_dataset = train_eval_handler.get_dataset()

    valid_eval_handler = DatasetHandler(
        path=valid_dataset_path,
        dataset_type="test",
        file_type="csv",
        transcript_column="text",
        label_column="label",
        map_labels=False,
        mapping_dictionary=None,
        prompt_to_use=generation_task_prompts,
        tokenizer=tokenizer,
    )
    valid_eval_dataset = valid_eval_handler.get_dataset()

    test_handler = DatasetHandler(
        path=test_dataset_path,
        dataset_type="test",
        file_type="excel",
        transcript_column="text",
        label_column="label",
        map_labels=False,
        prompt_to_use=generation_task_prompts,
        tokenizer=tokenizer,
    )
    test_dataset = test_handler.get_dataset()

    # Train Model
    trainer_handler = TrainerHandler(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        output_dir=f"/workspace/trained_models/{model_name}",  # Save with dynamic name
        **training_params  # Training parameters
    )

    trainer_handler.trainer.add_callback(
        PushToHubCallback(
            base_model=model,
            trainer_handler=trainer_handler,
            train_dataset=train_eval_dataset,
            valid_dataset=valid_eval_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            prompts=generation_task_prompts,
            output_dir="/content/trained_models",
            model_par_name=model_name,
            # save_prob=True,
            organization="speechCare",
        )
    )

    results = trainer_handler.train()
    # trainer_handler.save_model()
    #### 🛑 CUDA MEMORY CLEANUP 🛑 ####
    print(f"🔹 Cleaning up CUDA memory for {model_name}")

    # Delete model, datasets, and trainer to free GPU memory
    del model
    del tokenizer
    del peft_config
    del trainer_handler
    del train_dataset
    del valid_dataset
    del test_dataset
    del model_handler
    del train_handler
    del valid_handler
    del test_handler

    # Force garbage collection
    gc.collect()

    # Empty CUDA cache
    torch.cuda.empty_cache()

    # Finish wandb run properly
    wandb.finish()

# %% [cell 32: markdown]
# # Inference

# %% [cell 33: code]
params = [{"per_device_train_batch_size":2, "gradient_accumulation_steps":2, "num_train_epochs":12, "learning_rate":2e-4,},
     {"load_quantized": None, "lora_dropout": 0.1, "lora_rank": 128}]

training_params, model_params = params[0], params[1]

# Load Model and Tokenizer
model_handler = ModelHandler(
    base_model='medalpaca/medalpaca-7b',
    device_map="auto",
    tokenizer_trust_remote_code=True,
    use_lora=True,
    **model_params  # LoRA & Quantization parameters
)

tokenizer = model_handler.load_tokenizer()

train_eval_handler = DatasetHandler(
    path=train_dataset_path,
    dataset_type="test",
    file_type="csv",
    transcript_column="text",
    label_column="label",
    map_labels=False,
    mapping_dictionary=None,
    prompt_to_use=Inference_prompts,
    tokenizer=tokenizer,
)
train_eval_dataset = train_eval_handler.get_dataset()

dataset = train_eval_dataset

# %% [cell 34: code]
model_name = "medalpaca_lora_dropout0.1_rank128"

# %% [cell 35: markdown]
# ### Save Fine-Tuned Model
# This cell saves the final fine-tuned model and tokenizer to disk.  
# It can later be pushed to Hugging Face Hub or reloaded for inference.  

# %% [cell 36: code]
model_to_merge = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("medalpaca/medalpaca-7b").to("cuda"), "HUGGINGFACE_REPO")

merged_model = model_to_merge.merge_and_unload()

merged_model.save_pretrained(f"/workspace/trained_model/{model_name}")
tokenizer.save_pretrained(f"/workspace/trained_model/{model_name}")

# %% [cell 37: markdown]
# Initialize vllm for generation

# %% [cell 38: code]
llm = LLM(model=f"/workspace/trained_model/{model_name}", task="generate")

# %% [cell 39: code]
import pandas as pd

def generate_text(sampling_params, temp, p, k):
    pred_texts = []
    for i, row in tqdm(enumerate(dataset), total=len(dataset)):
        # If you're using a prompt template, you might do:
        # prompt = prompt_template.format(text=row["text"])
        prompt = row["text"]
        label = row['gt_label']
    
        # generate() returns an object with an 'outputs' attribute that is a list of responses.
        outputs = llm.generate(prompt, sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            generated_text = generated_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1]
            
        generated_text = generated_text.split("text:")[-1]    
        pred_texts.append((generated_text, label))

    # Convert the list to a DataFrame
    df = pd.DataFrame(pred_texts, columns=["Text", "Label"])

    # Save the DataFrame to an Excel file
    df.to_excel(f"/workspace/generated_text/generated_text_lama3.1_8b_epoch15_tempture_{str(temp)}_top_p_{str(p)}_top_k_{str(k)}.xlsx", index=False)
    
    print(f"Excel file '/workspace/generated_text_lama3.1_8b_epoch15_tempture_{str(temp)}_top_p_{str(p)}_top_k_{str(k)}.xlsx' created successfully.")

# %% [cell 40: code]
tempture = [1]
top_p = [0.95]
top_k = [50]

for temp in tempture:
    for p in top_p:
        for k in top_k:
            # Set sampling parameters; note that vLLM uses max_tokens (similar to max_new_tokens)
            sampling_params = SamplingParams(max_tokens=512, temperature=temp, top_p=p, top_k=k)
            generate_text(sampling_params, temp, p, k)
