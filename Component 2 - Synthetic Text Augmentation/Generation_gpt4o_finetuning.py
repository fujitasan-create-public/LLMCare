# Auto-converted from notebook

# %% [cell 1: markdown]
# # Libraries + Requirements
# This cell imports all required libraries and initializes the OpenAI client.
# 
# Run this cell first to prepare the environment.

# %% [cell 2: code]
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import json
import random 
import pandas as pd 
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import openai
from openai import OpenAI

model_name = 'gpt-4o-2024-11-20'
api_key = 'YOUR_API_KEY' #rs_user_description

client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key,
)

label_map = {0:'Healthy', 1:'AD'}

# %% [cell 3: markdown]
# # Data Construction

# %% [cell 4: markdown]
# ## Constant 
# 
# This cell sets the locations of training/validation/test datasets and builds prompt templates used for synthetic text generation.

# %% [cell 5: code]
# System instruction for chat template

label_mapping_dict = {0: "Healthy", 1: "ADRD"}

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

Inference_prompts = [(
    "You are an expert in cognitive health and language analysis. You will generate a spoken language transcript of a person describing the 'Cookie Theft' picture. This should reflect spontaneous speech rather than formal written text. Generate a text based on the given label."
    "\nLabel: {label}"
    "\ntext:"
)]


train_data_path = './Data/train.csv'
valid_data_path = './Data/validation.csv'
test_data_path = './Data/Test_DePiC.xlsx'

train_out_file = './Data/train_prompt.jsonl'
valid_out_file = './Data/valid_prompt.jsonl'

# %% [cell 6: markdown]
# ## Utility: Convert dataset into JSONL format
# This function takes a DataFrame, wraps each row into a structured  chat-style format (system, user, assistant), and saves it to disk.

# %% [cell 7: code]
def save_to_jsonl(data, output_file_path):
    jsonl_data = []
    for index, row in data.iterrows():
        selected_prompt = random.choice(generation_task_prompts)
        prompt = selected_prompt.format(label=label_mapping_dict[row["label"]])
        jsonl_data.append({
            "messages": [
                {"role": "system", "content": generation_system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": str(row['text'])}
            ]
        })

    # Save to JSONL format
    with open(output_file_path, 'w') as f:
        for item in jsonl_data:
            f.write(json.dumps(item) + '\n')
    return jsonl_data

# %% [cell 8: markdown]
# Loads CSV datasets, applies the JSONL conversion function, and generates training/validation prompt files.

# %% [cell 9: code]
train_df = pd.read_csv(train_data_path)
valid_df = pd.read_csv(valid_data_path)

train_data = save_to_jsonl(train_df, train_out_file)
valid_data = save_to_jsonl(valid_df, valid_out_file)

# %% [cell 10: markdown]
# ## Validate dataset format
# This function ensures the JSONL files have the correct structure:
# 
# - Messages contain role and content
# - Assistant responses are present
# - No unrecognized keys
# 
# Run this after dataset preparation to confirm integrity.

# %% [cell 11: code]
def check_data(dataset):
    # Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue
            
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
            
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            
            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1
            
            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1
                
            content = message.get("content", None)
            function_call = message.get("function_call", None)
            
            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1
        
        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")
        
        
check_data(train_data)
check_data(valid_data)

# %% [cell 12: markdown]
# ## Upload Datasets

# %% [cell 13: code]
train_file = client.files.create(
  file=open(train_out_file, "rb"),
  purpose="fine-tune"
)

valid_file = client.files.create(
  file=open(valid_out_file, "rb"),
  purpose="fine-tune"
)

print(f"Training file Info: {train_file}")
print(f"Validation file Info: {valid_file}")

# %% [cell 14: code]
train_file = client.files.retrieve('FILE_ID')
valid_file = client.files.retrieve('FILE_ID')

# %% [cell 15: markdown]
# # Tune Model

# %% [cell 16: markdown]
# ## Creat Job

# %% [cell 17: markdown]
# #### Train2 - LR multiplier = 2.5

# %% [cell 18: code]
model = client.fine_tuning.jobs.create(
  training_file=train_file.id, 
  validation_file=valid_file.id,
  model="gpt-4o-2024-08-06", 
  hyperparameters={
    "n_epochs": 3,
	"batch_size": 16,
	"learning_rate_multiplier": 2.5
  }
)
job_id = model.id
status = model.status

print(f'Fine-tuning model with jobID: {job_id}.')
print(f"Training Response: {model}")
print(f"Training Status: {status}")

# %% [cell 19: markdown]
# ## Check Status

# %% [cell 20: code]
# Retrieve the state of a fine-tune
client.fine_tuning.jobs.retrieve(job_id).status

# %% [cell 21: markdown]
# #### Train2 - LR Multiplier = 2.5

# %% [cell 22: code]
generation_tuned_model_id = client.fine_tuning.jobs.retrieve(job_id).fine_tuned_model
print(f'model id: {generation_tuned_model_id}')
print(client.fine_tuning.jobs.retrieve(job_id))

# %% [cell 23: markdown]
# ## Train and Validation Metrics

# %% [cell 24: code]
fine_tune_results = client.fine_tuning.jobs.retrieve(job_id).result_files
result_file = client.files.retrieve(fine_tune_results[0])
content = client.files.content(result_file.id)
import base64
base64.b64decode(content.text.encode('utf-8'))
with open('./Data/gptFinetuning_results_first_prompt.csv', 'wb') as f:
    f.write(base64.b64decode(content.text.encode('utf-8')))

# %% [cell 25: markdown]
# ## Checkpoints

# %% [cell 26: markdown]
# #### Train2 - LR Multiplier=2.5

# %% [cell 27: code]
chck_point = 'YOUR_CHECKPOINT'

# %% [cell 28: code]
model_checkpoin_list = client.fine_tuning.jobs.checkpoints.list(chck_point)
# print(model_checkpoin_list)
checkpoint_names = []
for chckpnt in model_checkpoin_list:
    print(f'model id: {chckpnt.id}')
    print(f'model name: {chckpnt.fine_tuned_model_checkpoint}')
    checkpoint_names.append(chckpnt.fine_tuned_model_checkpoint)
    print(f'step number: {chckpnt.step_number}')
    print(f'valid loss: {chckpnt.metrics.full_valid_loss}')
    print(f'valid accuracy: {chckpnt.metrics.full_valid_mean_token_accuracy}')
    print('\n\n')

# %% [cell 29: markdown]
# # Use the Model

# %% [cell 30: code]
def chat_with_llm(system_message, user_instruction):
    chat_response = client.chat.completions.create(
    model=model_name, 
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_instruction},
    ],
    max_completion_tokens=350,
    temperature=1,
    # top_p=0.95,
    )
    return chat_response.choices[0].message.content

def inference_tuned_model(model_id, data, save_path):
    pred_texts = []
    for j in range(0, aug_count):
        for i, row in tqdm(data.iterrows()):
            prompt = Inference_prompts[0].format(label=label_mapping_dict[row['label']])
            pred_text = chat_with_llm(generation_system_prompt, prompt)
            pred_texts.append((int(row['label']), pred_text))

    df = pd.DataFrame(pred_texts, columns=["label", "text"])
    df.to_excel(save_path, index=False)
        
    return data

# %% [cell 31: code]
aug_count = 1
for i, chkpnt in enumerate(checkpoint_names):
    train_data = pd.read_csv(train_data_path)
    modif_train_data = inference_tuned_model(chkpnt, train_data, f"./Data/gpt4_temp1_two_aug_step{i}.xlsx")
