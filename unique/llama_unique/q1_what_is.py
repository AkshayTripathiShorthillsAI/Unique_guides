
import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
load_dotenv()
import time
import retreival
import transformers
from torch import cuda, bfloat16
import time
 
begin = time.time() 

model_id = 'meta-llama/Llama-2-7b-chat-hf'
hf_auth = os.getenv("HF_AUTH") 
 
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

# 4-bit quantization
# bnb_config = transformers.BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type='nf4',
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=bfloat16
# )

# 8-bit quantization
bnb_config = transformers.BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)


model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    token=hf_auth
)
# model.eval()
 
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    token=hf_auth
)

pipeline = transformers.pipeline(
   "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

model_load_time = time.time()

prompt = input("\n\nEnter the prompt: ")
context = retreival.query_pdf(prompt)

# Prompt for What is Mpph Syndrome ?

# full_prompt = f""" You a Medical Geneticist .Based on the provided context, answer questions about specific events, characters, and details mentioned in the text. Provide the sources also from where the conclusion is drawn.
# context: {context}
# prompt: {prompt}
# """

full_prompt = f""" 
You are a Medical Geneticist. Based on the provided context, answer question and follow the template fill the brackets regarding question and generate response limit upto 100 words. 

template: [Name of condition] is caused by specific changes (known as pathogenic variants) to the DNA sequence of the gene [name of gene] ([name of gene]). The [name of gene] gene is located in the [short/long ‘p/q’] arm of chromosome [insert number] in a region called [region] as shown in the image below.
context: {context}
prompt: {prompt}"
"""


response = llm.invoke(full_prompt)
print(response, flush=True)

# print("\n\n",response)


end = time.time()
print(f"\n\nmodel loading time - {model_load_time-begin}")
print(f"total time take n - {end-begin}")


# query="What is 2+2"
# output = llm.predict(query)
# print("query",query)
# print("\noutput",output)

# def run_query():
#     prompt = input("\n\nEnter the prompt: ")
#     json_data = retreival.query_hp_book(prompt)
#     context = json_data
#     full_prompt = f""" Based on the provided context, answer questions about specific events, characters, and details mentioned in the text.
#     context: {context}
#     prompt: {prompt}
#     """

#     response = llm.invoke(full_prompt)
#     print("\n\n",response)
#     return response
