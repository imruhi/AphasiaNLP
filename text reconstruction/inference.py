import pandas as pd
import torch
import numpy as np
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re
import pickle
import random

save_fp = "trained-model3-2"
PEFT_MODEL = save_fp + "/best-model" # "trained-model3" # trained-model3 for 3 sentence model
data_fp = "data/eval3.csv" # eval1 for one sentence model, eval for 3 sentence model
results = save_fp + "/preds"
max_new_tokens = 45

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("\n##### Loading model #####\n")

config = PeftConfig.from_pretrained(PEFT_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map = 'auto',
    trust_remote_code=True,
)

tokenizer=AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(model, PEFT_MODEL)


generation_config = model.generation_config
generation_config.max_new_tokens = max_new_tokens
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 5
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

print("\n##### Generation config #####\n")
print(generation_config)

print("\n##### Loading dataset #####\n")

def get_answers(output):
    a = re.findall(r'### Answer:\n(.*)', str(output))
    # a = re.findall(r'### Correct sentences:\n(.*)', str(output))
    # print(output)
    # print(a)
    # print(a[0])
    return a[0]
        
data = pd.read_csv(data_fp)# .drop("Unnamed: 0", axis=1)
data = list(data["prompt"])
# test purpose
# data = data[:5]

print("\n##### Doing inference #####\n")
all_answers = []
device = "cuda"
i = 0

model.config.use_cache = True

for prompt in data:
    answers = []
    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
      outputs = model.generate(
          input_ids = encoding.input_ids,
          attention_mask = encoding.attention_mask,
          generation_config = generation_config,
          num_beams=5, 
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
np.save(results+str(i)+".npy", arr)
