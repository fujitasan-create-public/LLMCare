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
%pip install -q -U google-generativeai
%pip install openai
# %pip install wandb


%pip install openpyxl
%pip install typing_extensions==4.7.1 --upgrade
%pip install tiktoken
%pip install protobuf
%pip install sentencepiece

%pip install scikit-learn
%pip install --upgrade Pillow
%pip install typing_extensions --upgrade
%pip install huggingface_hub

# %% [cell 4: markdown]
# ## ***Import***

# %% [cell 5: code]
import os
import time
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import openai
from openai import OpenAI

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import google.generativeai as genai

from huggingface_hub import login

# %% [cell 6: markdown]
# # ***Constants***

# %% [cell 7: code]
google_api_key = ''
openai_api_key = ''
openrouter_api_key = ''
openrouter_base_url = 'https://openrouter.ai/api/v1'

# root_dir = 'D:/speechADRD/dataSynthesis'
root_dir = '/content/drive/MyDrive/speechCare/LLMCare+CDSS'
data_dir = os.path.join(root_dir, 'data/delaware')
pred_dir = os.path.join(root_dir, 'predictions/delaware')

# Define prompts
prompts = {
    "noExp": (
                "You are an expert in cognitive health and language analysis. You will analyze a spoken language transcript from a person describing the 'cookie theft' picture. This is not written text but a transcription of spontaneous speech."
                "\nAnalyze the provided transcript and classify it into one of two categories: 'Healthy' for a healthy cognitive state or 'ADRD' for Alzheimer's disease and related dementias."
                "\nProvide only the label ('Healthy' or 'ADRD') as the output. Do not include explanations or additional text."
                "\nText: {text}:"
                "\nLabel:"
            )
}

# %% [cell 8: markdown]
# # ***Utils***

# %% [cell 9: code]
# Function to process predictions
def process_predictions(data, labels, prompt_name):
    mapping = {'ADRD': 1, 'Healthy': 0}
    data[f'pred_{prompt_name}'] = ["Healthy" if "Healthy" in label else "ADRD" if "ADRD" in label else None for label in labels]
    data[f'pred_{prompt_name}_mapped'] = data[f'pred_{prompt_name}'].map(mapping)
    return data

# Function to evaluate predictions
def evaluate_predictions(data, true_label_col, pred_col):
    y_true = data[true_label_col].dropna()
    y_pred = data[pred_col].dropna()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"{pred_col} Evaluation:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"True Positives: {tp}, False Positives: {fp}, True Negatives: {tn}, False Negatives: {fn}")
    print("\n")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "tn": tn, "fp": fp, "fn": fn, "tp": tp}

# Function to map label
def map_label(label):
    if "Healthy" in label:
        return 'Healthy'
    elif "ADR" in label:
        return 'ADRD'
    else:
        return None



# %% [cell 10: markdown]
# # ***Openweigh Models without OpenRouter***

# %% [cell 11: markdown]
# ## ***Download Model***

# %% [cell 12: code]
# Add your Huggingface token here
login(token="")

!huggingface-cli download unsloth/Meta-Llama-3.1-8B-Instruct --local-dir ./llama8B

# %% [cell 13: markdown]
# ## ***Functions***

# %% [cell 14: code]
# Initialize a Hugging Face text-generation pipeline
def initialize_pipeline(model_id, max_new_tokens=20, do_sample=True, temperature=0.1):
    if "medalpaca" in model_id:
        print("medalpaca")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        # Ensure pad token is set properly
        pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id
    else:
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        # Handle cases where tokenizer has no pad token
        if 'Ministral' in model_id:
            pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id
    return pipeline


# Generate predictions for a dataset
def generate_predictions(pipeline, dataset, prompt_name, model_id):
    outputs = []
    if "medalpaca" in model_id:
        # Special handling for MedAlpaca: retry until label is parsed or max attempts reached
        labels = []
        for i, row in tqdm(dataset.iterrows()):
            trial = 1
            label = row.get(f'pred_{prompt_name}_mapped', None)
            while (label is None or pd.isna(label)) and trial <= 10:
                results = pipeline(row['input'], pad_token_id=pipeline.tokenizer.eos_token_id)[0]["generated_text"]
                label = map_label(results.split('Label: ')[-1])
                trial += 1
            labels.append(label)
        return labels
    else:
        # Batched inference for Hugging Face datasets
        for out in tqdm(pipeline(KeyDataset(dataset, "input"), batch_size=8)):
            outputs.append(out)
        outputs1 = [out[0]['generated_text'] for out in outputs]
        labels = [out.split('Label: ')[-1].strip() for out in outputs1]
        return labels
    

# Full inference pipeline with retries and evaluation
def run_model_inference2(model_id, prompts, data_path, output_path, max_attempts=10, img=None):
    # Load dataset (Excel assumed, fallback to CSV is omitted here)
    data = pd.read_excel(data_path)
    init_temp = 0.1
    
    # Initialize generation pipeline
    pipeline = initialize_pipeline(model_id, 20, True, init_temp)

    for prompt_name, prompt_template in prompts.items():
        # Build inputs from prompt template
        data["input"] = data["text"].apply(lambda x: prompt_template.format(text=x))
        
        # Convert to HF dataset unless using MedAlpaca
        dataset = datasets.Dataset.from_pandas(data) if "medalpaca" not in model_id else data
        labels = generate_predictions(pipeline, dataset, prompt_name, model_id)

        # Insert initial predictions into dataframe
        data = process_predictions(data, labels, prompt_name)

        # Retry loop for NaN predictions with increasing temperature
        i = 0
        while data[f'pred_{prompt_name}_mapped'].isna().sum() > 0:
            new_temp = init_temp + (i * 0.06)
            if new_temp > 0.4:
                new_temp = 0.4
                # Switch to vision model if needed
                labels = generate_vision_predictions(
                    model, processor,
                    data[data[f'pred_{prompt_name}_mapped'].isna()],
                    prompt_name,
                    temperature=new_temp
                )
            else:
                # Re-initialize pipeline with higher temperature
                pipeline = initialize_pipeline(model_id, do_sample=True, temperature=new_temp)
                nan_data = data[data[f'pred_{prompt_name}_mapped'].isna()].reset_index(drop=True)
                nan_data = datasets.Dataset.from_pandas(nan_data) if "medalpaca" not in model_id else nan_data
                labels = generate_predictions(pipeline, nan_data, prompt_name, model_id)

            # Map labels to numeric values
            mapping = {'ADRD': 1, 'Healthy': 0, None: None}
            labels = [map_label(label) for label in labels]
            mapped_labels = [mapping[label] for label in labels]

            # Fill missing predictions
            data.loc[data[f'pred_{prompt_name}_mapped'].isna(), f'pred_{prompt_name}'] = labels
            data.loc[data[f'pred_{prompt_name}_mapped'].isna(), f'pred_{prompt_name}_mapped'] = mapped_labels

            i += 1

        # Ensure predictions are integer-typed
        data[f'pred_{prompt_name}_mapped'] = data[f'pred_{prompt_name}_mapped'].apply(lambda x: int(x))
        # Evaluate predictions against ground truth
        evaluation_results = evaluate_predictions(data, "label", f"pred_{prompt_name}_mapped")

    # Save predictions with dataset
    data.to_csv(output_path, index=False)
    return data

# %% [cell 15: markdown]
# ## ***Inference***

# %% [cell 16: markdown]
# ### ***Llama 3.2 3B***

# %% [cell 17: code]
# Initialize model pipeline
model_id = 'unsloth/Llama-3.2-3B-Instruct'
data_path = os.path.join(data_dir, 'Test_DePiC.xlsx')
output_path =os.path.join(pred_dir, 'llama3B_zeroShot_predictions.csv')
data = run_model_inference2(model_id, prompts, data_path, output_path)

# %% [cell 18: markdown]
# ### ***MedAlpaca***

# %% [cell 19: code]
# Initialize model pipeline
model_id = "medalpaca/medalpaca-7b"
data_path = os.path.join(data_dir, 'Test_DePiC.xlsx')
output_path =os.path.join(pred_dir, 'medAlpsca7_zeroShot_predictions.csv')
data = run_model_inference2(model_id, prompts, data_path, output_path)

# %% [cell 20: markdown]
# ### ***Ministral***

# %% [cell 21: code]
# Initialize model pipeline
model_id = "mistralai/Ministral-8B-Instruct-2410"
data_path = os.path.join(data_dir, 'Test_DePiC.xlsx')
output_path =os.path.join(pred_dir, 'ministral8B_zeroShot_predictions.csv')
data = run_model_inference2(model_id, prompts, data_path, output_path)

# %% [cell 22: markdown]
# ### ***Llama3.3 70B***

# %% [cell 23: code]
# Initialize model pipeline
model_id = "unsloth/Llama-3.3-70B-Instruct"
data_path = os.path.join(data_dir, 'Test_DePiC.xlsx')
output_path =os.path.join(pred_dir, 'llama70B_zeroShot_predictions.csv')
data = run_model_inference2(model_id, prompts, data_path, output_path)

# %% [cell 24: markdown]
# ### ***Llama3.1 8B***

# %% [cell 25: code]
# Initialize model pipeline
model_id = "/workspace/llama8B"
data_path = os.path.join(data_dir, 'Test_DePiC.xlsx')
output_path =os.path.join(pred_dir, 'llama8B_zeroShot_predictions.csv')
data = run_model_inference2(model_id, prompts, data_path, output_path)

# %% [cell 26: markdown]
# # ***GPT***

# %% [cell 27: code]
def initialize_openai_client(api_key=openai_api_key, base_url=None):
    """
    Initialize OpenAI client with optional custom base URL.
    """
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    else:
        return OpenAI(api_key=api_key)


def chat_with_llm(client, system_message, user_instruction, model_name, temperature=0.0, seed=0):
    """
    Send a chat completion request to the LLM and return the response text.
    """
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_instruction},
        ],
        temperature=temperature,
        seed=seed,
        max_completion_tokens=50,
    )
    return chat_response.choices[0].message.content


def run_gpt_inference(model_name, data_path, output_path, prompts, name='gpt4o', temperature=0.0, seed=0):
    """
    Run zero/few-shot inference using an OpenAI GPT model.
    - Reads input dataset (Excel or CSV).
    - Generates predictions for each prompt template.
    - Handles retries for missing predictions.
    - Saves results to CSV.
    """
    client = initialize_openai_client()

    # Load dataset (try Excel first, fallback to CSV)
    try:
        data = pd.read_excel(data_path)
    except:
        data = pd.read_csv(data_path)
        # Map ground-truth labels to numeric values (for consistency)
        data['label'] = data['label'].map({'ADRD': 1, 'Control': 0})

    # Mapping dictionary for model outputs → numeric labels
    mapping = {'ADRD': 1, 'Healthy': 0, None: None}

    # Loop through all prompt variations
    for prompt_name, prompt_template in prompts.items():
        # Build input column for current prompt
        data[f"input_{prompt_name}"] = data['text'].apply(lambda x: prompt_template.format(text=x))

        # Initialize prediction column if it doesn’t exist
        pred_col = f'{name}_temp{seed}_pred_label_{prompt_name}'
        if pred_col in data.columns:
            print('skip naming')
        else:
            data[pred_col] = ''

        output_tag = []
        # Run inference row by row
        for i, row in tqdm(data.iterrows(), total=len(data)):
            if len(row[pred_col]) > 2:  # Skip rows already processed
                print('skipping done rows')
                continue

            out = chat_with_llm(client, '', row[f'input_{prompt_name}'], model_name, seed=seed)
            output_tag.append(out)

            # Save raw and mapped predictions
            data.loc[i, pred_col] = map_label(out.split('Label: ')[-1])
            data.loc[i, f'{pred_col}_mapped'] = mapping[data.loc[i, pred_col]]

        # Retry loop for missing predictions (NaN rows)
        i = 0
        while data[pred_col].isna().sum() > 0:
            nan_data = data[data[pred_col].isna()].reset_index(drop=True)

            labels = [
                map_label(chat_with_llm(client, '', text, model_name, seed=seed).split('Label: ')[-1])
                for text in nan_data[f'input_{prompt_name}']
            ]
            mapped_labels = [mapping[label] for label in labels]

            data.loc[data[pred_col].isna(), pred_col] = labels
            data.loc[data[pred_col].isna(), f'{pred_col}_mapped'] = mapped_labels
            i += 1

        # Ensure mapped predictions column exists and enforce integer type
        if f'{pred_col}_mapped' not in data.columns:
            data[f'{pred_col}_mapped'] = data[pred_col].map(mapping)

        data[f'{pred_col}_mapped'] = data[f'{pred_col}_mapped'].apply(
            lambda x: int(x) if x is not None else x
        )

        # Evaluate predictions for this prompt variation
        evaluation_results = evaluate_predictions(data, "label", f"{pred_col}_mapped")

    # Save all predictions to output file
    data.to_csv(output_path, index=False)

# %% [cell 28: code]
model_name = 'gpt-4o-2024-08-06'
data_path = os.path.join(data_dir,"test_delaware.csv")
output_path = os.path.join(pred_dir, "gpt4o_seed_zeroShot_predictions.csv")
run_gpt_inference(model_name, data_path, output_path, prompts, seed=0)

# %% [cell 29: markdown]
# # ***Open weight models with OpenRouter***

# %% [cell 30: markdown]
# ## ***LLama 3.1 8B***

# %% [cell 31: code]
# Define model and file paths
model_name = "meta-llama/llama-3.1-8b-instruct"
data_path = os.path.join(data_dir, "test_delaware.csv")
output_path = os.path.join(pred_dir, "llama8B_zeroShot_seed_predictions.csv")

# Initialize OpenAI API client
client = initialize_openai_client(openrouter_api_key, openrouter_base_url)

# Load dataset
data = pd.read_csv(data_path)

# Define label mapping (for predictions and evaluation)
mapping = {'MCI': 1, 'Healthy': 0, None: None}

# Map ground-truth labels to numeric format
data['label'] = data['label'].map({'MCI': 1, 'Control': 0})

# %% [cell 32: code]
for prompt_name, prompt_template in prompts.items():
    # Create model input column for current prompt
    data[f"input_{prompt_name}"] = data['transcription'].apply(
        lambda x: prompt_template.format(text=x)
    )

    # Initialize prediction column if not already present
    if f'llama8B_pred_label_{prompt_name}' in data.columns:
        print('skip naming')
    else:
        data[f'llama8B_pred_label_{prompt_name}'] = ''

    output_tag = []
    # Run inference row by row
    for i, row in tqdm(data.iterrows(), total=len(data)):
        if len(row[f'llama8B_pred_label_{prompt_name}']) > 2:
            print('skipping done rows')
            continue
        out = chat_with_llm(client, '', row[f'input_{prompt_name}'], model_name)
        output_tag.append(out)

        # Save raw and mapped predictions
        data.loc[i, f'llama8B_pred_label_{prompt_name}'] = map_label(out.split('Label: ')[-1])
        data.loc[i, f'llama8B_pred_label_{prompt_name}_mapped'] = mapping[
            data.loc[i, f'llama8B_pred_label_{prompt_name}']
        ]

    # Handle missing predictions (retry loop until none are NaN)
    i = 0
    while data[f'llama8B_pred_label_{prompt_name}'].isna().sum() > 0:
        nan_data = data[data[f'llama8B_pred_label_{prompt_name}'].isna()].reset_index(drop=True)
        labels = [
            map_label(chat_with_llm(client, '', text, model_name).split('Label: ')[-1])
            for text in nan_data[f'input_{prompt_name}']
        ]
        mapped_labels = [mapping[label] for label in labels]

        data.loc[data[f'llama8B_pred_label_{prompt_name}'].isna(),
                 f'llama8B_pred_label_{prompt_name}'] = labels
        data.loc[data[f'llama8B_pred_label_{prompt_name}'].isna(),
                 f'llama8B_pred_label_{prompt_name}_mapped'] = mapped_labels
        i += 1

    # Ensure mapped column exists and is properly typed
    if f'llama8B_pred_label_{prompt_name}_mapped' not in data.columns:
        data[f'llama8B_pred_label_{prompt_name}_mapped'] = data[f'llama8B_pred_label_{prompt_name}'].map(mapping)
    data[f'llama8B_pred_label_{prompt_name}_mapped'] = data[f'llama8B_pred_label_{prompt_name}_mapped'].apply(
        lambda x: int(x) if x is not None else x
    )

    # Evaluate predictions against ground truth
    evaluation_results = evaluate_predictions(data, "label", f"llama8B_pred_label_{prompt_name}_mapped")

# %% [cell 33: markdown]
# ## ***Llama 3.3 70B***

# %% [cell 34: code]
# Define model and file paths
model_name = "meta-llama/llama-3.3-70b-instruct"
data_path = os.path.join(data_dir, "test_delaware.csv")
output_path = os.path.join(pred_dir, "llama70B_seed_zeroShot_predictions.csv")

# Initialize OpenAI API client
client = initialize_openai_client(openrouter_api_key, openrouter_base_url)

# Load evaluation dataset
data = pd.read_csv(data_path)

# Define mapping for predictions (model outputs → numeric labels)
mapping = {'MCI': 1, 'Healthy': 0, None: None}

# Convert ground-truth labels to numeric format
data['label'] = data['label'].map({'MCI': 1, 'Control': 0})

# %% [cell 35: code]
for prompt_name, prompt_template in prompts.items():
    # Create model input column using the current prompt template
    data[f"input_{prompt_name}"] = data['transcription'].apply(
        lambda x: prompt_template.format(text=x)
    )

    # Initialize prediction column if it doesn't already exist
    if f'llama70_seed0_pred_label_{prompt_name}' in data.columns:
        print('skip naming')
    else:
        data[f'llama70_seed0_pred_label_{prompt_name}'] = ''

    output_tag = []
    # Run inference row by row
    for i, row in tqdm(data.iterrows(), total=len(data)):
        if len(row[f'llama70_seed0_pred_label_{prompt_name}']) > 2:
            print('skipping done rows')
            continue

        out = chat_with_llm(client, '', row[f'input_{prompt_name}'], model_name)
        output_tag.append(out)

        # Save raw and mapped predictions
        data.loc[i, f'llama70_seed0_pred_label_{prompt_name}'] = map_label(out.split('Label: ')[-1])
        data.loc[i, f'llama70_seed0_pred_label_{prompt_name}_mapped'] = mapping[
            data.loc[i, f'llama70_seed0_pred_label_{prompt_name}']
        ]

    # Retry loop: handle rows that are still NaN after first pass
    i = 0
    while data[f'llama70_seed0_pred_label_{prompt_name}'].isna().sum() > 0:
        nan_data = data[data[f'llama70_seed0_pred_label_{prompt_name}'].isna()].reset_index(drop=True)

        labels = [
            map_label(chat_with_llm(client, '', text, model_name).split('Label: ')[-1])
            for text in nan_data[f'input_{prompt_name}']
        ]
        mapped_labels = [mapping[label] for label in labels]

        data.loc[data[f'llama70_seed0_pred_label_{prompt_name}'].isna(),
                 f'llama70_seed0_pred_label_{prompt_name}'] = labels
        data.loc[data[f'llama70_seed0_pred_label_{prompt_name}'].isna(),
                 f'llama70_seed0_pred_label_{prompt_name}_mapped'] = mapped_labels
        i += 1

    # Ensure mapped predictions column exists and has correct type
    if f'llama70_seed0_pred_label_{prompt_name}_mapped' not in data.columns:
        data[f'llama70_seed0_pred_label_{prompt_name}_mapped'] = data[f'llama70_seed0_pred_label_{prompt_name}'].map(mapping)

    data[f'llama70_seed0_pred_label_{prompt_name}_mapped'] = data[f'llama70_seed0_pred_label_{prompt_name}_mapped'].apply(
        lambda x: int(x) if x is not None else x
    )

    # Evaluate predictions against ground-truth labels
    evaluation_results = evaluate_predictions(
        data, "label", f"llama70_seed0_pred_label_{prompt_name}_mapped"
    )

# %% [cell 36: markdown]
# ## ***Ministral***

# %% [cell 37: code]
# Define model and file paths
model_name = "mistralai/ministral-8b"
data_path = os.path.join(data_dir, "test_delaware.csv")
output_path = os.path.join(pred_dir, "ministral_zeroShot_predictions.csv")

# Initialize OpenAI API client
client = initialize_openai_client(openrouter_api_key, openrouter_base_url)

# Load evaluation dataset
data = pd.read_csv(data_path)

# Define mapping for model outputs → numeric labels
mapping = {'MCI': 1, 'Healthy': 0, None: None}

# Convert ground-truth labels to numeric format
data['label'] = data['label'].map({'MCI': 1, 'Control': 0})

# %% [cell 38: code]
for prompt_name, prompt_template in prompts.items():
    # Build model input column for this prompt
    data[f"input_{prompt_name}"] = data['transcription'].apply(
        lambda x: prompt_template.format(text=x)
    )

    # Initialize prediction column if it doesn't already exist
    if f'ministral_pred_label_{prompt_name}' in data.columns:
        print('skip naming')
    else:
        data[f'ministral_pred_label_{prompt_name}'] = ''

    output_tag = []
    # Run inference row by row
    for i, row in tqdm(data.iterrows(), total=len(data)):
        if len(row[f'ministral_pred_label_{prompt_name}']) > 2:
            print('skipping done rows')
            continue

        out = chat_with_llm(client, '', row[f'input_{prompt_name}'], model_name)
        output_tag.append(out)

        # Save raw prediction and mapped numeric label
        data.loc[i, f'ministral_pred_label_{prompt_name}'] = map_label(out.split('Label: ')[-1])
        data.loc[i, f'ministral_pred_label_{prompt_name}_mapped'] = mapping[
            data.loc[i, f'ministral_pred_label_{prompt_name}']
        ]

    # Retry loop: handle rows with missing predictions (NaN) until resolved
    i = 0
    while data[f'ministral_pred_label_{prompt_name}'].isna().sum() > 0:
        nan_data = data[data[f'ministral_pred_label_{prompt_name}'].isna()].reset_index(drop=True)

        labels = [
            map_label(chat_with_llm(client, '', text, model_name).split('Label: ')[-1])
            for text in nan_data[f'input_{prompt_name}']
        ]
        mapped_labels = [mapping[label] for label in labels]

        data.loc[data[f'ministral_pred_label_{prompt_name}'].isna(),
                 f'ministral_pred_label_{prompt_name}'] = labels
        data.loc[data[f'ministral_pred_label_{prompt_name}'].isna(),
                 f'ministral_pred_label_{prompt_name}_mapped'] = mapped_labels
        i += 1

    # Ensure mapped column exists and enforce integer type
    if f'ministral_pred_label_{prompt_name}_mapped' not in data.columns:
        data[f'ministral_pred_label_{prompt_name}_mapped'] = data[f'ministral_pred_label_{prompt_name}'].map(mapping)

    data[f'ministral_pred_label_{prompt_name}_mapped'] = data[f'ministral_pred_label_{prompt_name}_mapped'].apply(
        lambda x: int(x) if x is not None else x
    )

    # Evaluate predictions vs ground-truth labels
    evaluation_results = evaluate_predictions(
        data, "label", f"ministral_pred_label_{prompt_name}_mapped"
    )
