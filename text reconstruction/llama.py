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
import re
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, logging
from transformers.trainer_callback import TrainerCallback
import datasets 

print("\n##### Starting code #####\n")
datasets.disable_progress_bar()
        
model = "meta-llama/Llama-2-7b-hf"
data_fp = "data3.csv"
save_fp = "trained-model3"
MODEL_NAME = model
access_token = "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"
extra_data = "added_data3.csv"


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
    r=64,
    lora_alpha=16,
    target_modules=get_last_layer_linears(model),
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

generation_config = model.generation_config
generation_config.max_new_tokens = 20
generation_config.temperature = 0.75
generation_config.top_p = 0.75
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

def generate_and_tokenize_prompt(data_point):
    tokenized_full_prompt = tokenizer(data_point["prompt"], padding=True, 
                                      truncation=True)
    return tokenized_full_prompt

print("\n##### Loading dataset #####\n")
# load dataset
data = pd.read_csv(data_fp).drop("Unnamed: 0", axis=1)
# for testing
# data = data[:2000]
# split 70/30 train/test
dataset = Dataset.from_pandas(data).train_test_split(test_size=0.3, seed=42)
print("\n##### Tokenizing dataset #####\n")
dataset["train"] = dataset["train"].map(generate_and_tokenize_prompt, batched=True, batch_size=1)
dataset["test"] = dataset["test"].map(generate_and_tokenize_prompt, batched=True, batch_size=1)
print("\n##### Dataset #####\n", dataset,'\n')
print(dataset["train"][0])

# Extra train dataset (domain specific info)
print("\n##### Adding to dataset #####\n")
data2 = pd.read_csv(extra_data).drop("Unnamed: 0", axis=1)
dataset['train'] = Dataset.from_pandas(pd.concat([dataset['train'].to_pandas(), data2]))
print("\n##### Dataset #####\n", dataset,'\n')
print(dataset["train"][0])

print("\n##### Setting up train args #####\n")
training_args = transformers.TrainingArguments(
    overwrite_output_dir=True, # Overwrite the content of the output directory
    per_device_train_batch_size=4,  # Batch size for training
    per_device_eval_batch_size=4,  # Batch size for evaluation
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    learning_rate=2e-4,
    weight_decay=0.001,  # Weight decay
    fp16=False,
    bf16=True,
    output_dir="finetune",
    optim="paged_adamw_32bit",
    lr_scheduler_type="constant",
    warmup_ratio=0.03,
    logging_dir='./logs',
    logging_strategy="steps",
    logging_steps=200,
    save_strategy="steps",
    save_steps=200,
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    report_to="tensorboard",
    disable_tqdm=True,
)

logging.disable_progress_bar()

trainer = transformers.Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
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
print(f"\n##### Training completed in {training_time}. #####\n")
print("\n##### Saving model #####\n")
model.save_pretrained(save_fp)
