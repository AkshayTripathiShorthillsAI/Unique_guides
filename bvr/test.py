
# Load adapters from the Hub

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dotenv import load_dotenv

load_dotenv()
hf_auth = os.getenv("HF_AUTH")

peft_model_id = "Akshay47/Llama-2-7b-chat-hf-english-quotes"
config = PeftConfig.from_pretrained(peft_model_id ,use_auth_token=hf_auth)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto',
            token=hf_auth)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path , trust_remote_code=True ,
            token=hf_auth)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)
# Inference

batch = tokenizer("“Training models with PEFT and LoRa is cool” ->: ", return_tensors='pt')

with torch.cuda.amp.autocast():
  output_tokens = model.generate(**batch, max_new_tokens=50)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))