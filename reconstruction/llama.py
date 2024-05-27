import pandas as pd
import json
import os
import datetime
from pprint import pprint
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset, Dataset
from huggingface_hub import notebook_login

from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

model = "meta-llama/Llama-2-7b-hf"
data_fp = "data.csv"
MODEL_NAME = model
access_token = "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"

bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_use_double_quant=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_compute_dtype=torch.bfloat16
)

print("\n##### Loading model #####\n")

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)


import re
def get_num_layers(model):
    numbers = set()
    for name, _ in model.named_parameters():
        for number in re.findall(r'\d+', name):
            numbers.add(int(number))
    return max(numbers)

def get_last_layer_linears(model):
    names = []
    
    num_layers = get_num_layers(model)
    for name, module in model.named_modules():
        if str(num_layers) in name and not "encoder" in name:
            if isinstance(module, torch.nn.Linear):
                names.append(name)
    return names
    
config = LoraConfig(
    r=2,
    lora_alpha=32,
    target_modules=get_last_layer_linears(model),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

generation_config = model.generation_config
generation_config.max_new_tokens = 10
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

def generate_and_tokenize_prompt(data_point):
    tokenized_full_prompt = tokenizer(data_point["prompt"], padding=True, 
                                      truncation=True)
    return tokenized_full_prompt

print("\n##### Loading dataset #####\n")
# load dataset
data = pd.read_csv(data_fp)# .drop("Unnamed: 0", axis=1)
# for testing
# data = data[:2000]
# split 70/30 train/test
dataset = Dataset.from_pandas(data).train_test_split(test_size=0.3, seed=42)
print("\n##### Tokenizing dataset #####\n")
dataset["train"] = dataset["train"].map(generate_and_tokenize_prompt, batched=True, batch_size=1)
dataset["test"] = dataset["test"].map(generate_and_tokenize_prompt, batched=True, batch_size=1)
print("\n##### Dataset #####\n", dataset,'\n')

training_args = transformers.TrainingArguments(
    overwrite_output_dir=True, # Overwrite the content of the output directory
    per_device_train_batch_size=2,  # Batch size for training
    per_device_eval_batch_size=2,  # Batch size for evaluation
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=1e-4,
    weight_decay=0.01,  # Weight decay
    fp16=True,
    output_dir="finetune",
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.01,
    report_to="none",
    logging_dir='./logs',
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,  # Limit the total number of checkpoints
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False
print("\n##### Base model #####\n")
print(model)
print("\n##### Training model #####\n")
start_time = datetime.datetime.now()  # Record the start time
trainer.train()  # Start training
end_time = datetime.datetime.now()  # Record the end time
print("\nStart: ", str(start_time), "End: ", str(end_time), "\n")
training_time = end_time - start_time  # Calculate total training time
print(f"\n##### Training completed in {training_time} seconds. #####\n")
print("\n##### Saving model #####\n")
model.save_pretrained("trained-model")