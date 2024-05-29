from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch

# Load the Falcon-7B model
model_id = "vilsonrodrigues/falcon-7b-instruct-sharded"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto",
    quantization_config=quantization_config,
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set up the pipeline for text generation
pipeline = pipeline(
    "text-generation",
    model=model_4bit,
    tokenizer=tokenizer,
    use_cache=True,
    device_map="auto",
    max_length=3296,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import retreival


# Generate the output of "2+2"
llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

def add_string(original_string, string_to_add):
    return original_string + string_to_add

query = input("\n\nEnter the prompt: ")
string=" [Name of disorder/condition], also referred to as [alternative names], is a rare genetic condition associated with [list common features e.g. developmental delay, varying degrees of learning (intellectual) disability, behavioural difference]. As is common with genetic conditions, each person is affected differently.[Name of disorder/condition] is caused by a change (variant) in the [name of gene] gene or the loss (deletion) of one copy of the [name of gene] gene. The loss of the gene may occur as part of a larger deletion that affects the chromosome on which the gene is located. [edit/delete as appropriate] "

new_query=add_string(query, string)
context = retreival.query_hp_book(new_query)
# full_prompt = f""" Give a specific answer according to the following prompt template . Find the
# answer from the context and just change what is written in the brackets by finding answers from context and keep everything same. Remove 
# the brackets and give a well framed answer. Strictly adhere to the prompt template. 
# Prompt template- [Name of disorder/condition], also referred to as [alternative names], is a rare genetic condition associated with [list common features e.g. developmental delay, varying degrees of learning (intellectual) disability, behavioural difference]. As is common with genetic conditions, each person is affected differently.
# [Name of disorder/condition] is caused by a change (variant) in the [name of gene] gene or the loss (deletion) of one copy of the [name of gene] gene. The loss of the gene may occur as part of a larger deletion that affects the chromosome on which the gene is located. [edit/delete as appropriate] 
# context: {context}
# prompt: {query}
# """

# full_prompt = f""" Based on the provided context, answer questions about specific events, characters, and details mentioned in the text. Provide the sources also from where the conclusion is drawn.
# context: {context}
# prompt: {query}
# """

# full_prompt = f"""Provide a specific answer according to the following prompt template. Find the answer from the context and just change what is written in the brackets by finding answers from context and keep everything else the same. Remove the brackets and give a well-framed answer. Strictly adhere to the prompt template.

# Prompt template - [disorder_name], also referred to as [alternative_names], is a rare genetic condition associated with [common_features]. As is common with genetic conditions, each person is affected differently.

# [disorder_name] is caused by a change (variant) in the [gene_name] gene or the loss (deletion) of one copy of the [gene_name] gene. The loss of the gene may occur as part of a larger deletion that affects the chromosome on which the gene is located. [edit/delete as appropriate]

# Context: {context}
# Prompt: {query}
# """

full_prompt = f""" 
You are a Medical Geneticist. Based on the provided context, answer question and follow the template fill the brackets regarding question and generate response limit upto 100 words. 

template: [Name of condition] is caused by specific changes (known as pathogenic variants) to the DNA sequence of the gene [name of gene] ([name of gene]). The [name of gene] gene is located in the [short/long ‘p/q’] arm of chromosome [insert number] in a region called [region] as shown in the image below.
context: {context}
question: {query}
answer:
"""



response = llm.predict(full_prompt)
print("\n\n",response)



# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
# import torch

# # Load the Falcon-7B model
# model_id = "vilsonrodrigues/falcon-7b-instruct-sharded"
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )
# model_4bit = AutoModelForCausalLM.from_pretrained(
#     model_id, 
#     device_map="auto",
#     quantization_config=quantization_config,
# )

# # Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# # Set up the pipeline for text generation
# pipeline = pipeline(
#     "text-generation",
#     model=model_4bit,
#     tokenizer=tokenizer,
#     use_cache=True,
#     device_map="auto",
#     max_length=296,
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     pad_token_id=tokenizer.eos_token_id,
# )

# # Generate the output of "2+2"
# output = pipeline("What is mpph syndrome? ")

# # Print the generated output
# # print(output["generated_text"])
# print("dtype output\n",type(output))
# print(output,"\n")
# print(output[0]['generated_text'])

