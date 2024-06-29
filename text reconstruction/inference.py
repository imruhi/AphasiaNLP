import pandas as pd
import torch
import numpy as np
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re
import pickle
import random

PEFT_MODEL = "trained-model1" # trained-model3 for 3 sentence model
data_fp = "eval1.csv" # eval1 for one sentence model, eval for 3 sentence model
results = "preds"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

config = PeftConfig.from_pretrained(PEFT_MODEL)

print("\n##### Loading model #####\n")
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map = 'auto',
    trust_remote_code=True,
)

tokenizer=AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, PEFT_MODEL)

generation_config = model.generation_config
generation_config.max_new_tokens = 20
generation_config.temperature = 0.75
generation_config.top_p = 0.75
generation_config.num_return_sequences = 5
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id
generation_config.num_beams = 5

print("\n##### Loading dataset #####\n")

def get_answers(output):
    a = re.findall(r'###Answer:\n(.*)', str(output))
    # print(a[0])
    return a[0]
        
data = pd.read_csv(data_fp)# .drop("Unnamed: 0", axis=1)
data = list(data["prompt"])
# test purpose
# random.seed(10)
# data = random.sample(data,500)

print("\n##### Doing inference #####\n")
all_answers = []
device = "cuda"
i = 0

for prompt in data:
    answers = []
    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
      outputs = model.generate(
          input_ids = encoding.input_ids,
          attention_mask = encoding.attention_mask,
          generation_config = generation_config, 
      )
      i += 1
      # print("-----------------------------------------------------------------")
      for output_sentence in tokenizer.batch_decode(outputs):
        # print(output_sentence)
        # print()
        answer = get_answers(output_sentence)
        answers.append(answer)
    all_answers.append(answers)
    if i%10 == 0:
      print(f"Finished {i} examples")
    if i%200 == 0:
      print(f"Saving {i} examples")
      arr = np.array(all_answers)
      np.save(results+str(i)+".npy", arr)
      all_answers = []

# with open('preds.pkl', 'wb') as f:
#  pickle.dump(all_answers, f)
print("\n##### Saving remaining output #####\n")  
arr = np.array(all_answers)
np.save(results, arr)
