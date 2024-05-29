# import os
# import time
# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
# import retreival  # Corrected import statement

# # Load the new model directly
# new_tokenizer = AutoTokenizer.from_pretrained("Clinical-AI-Apollo/Medical-NER")
# new_model = AutoModelForTokenClassification.from_pretrained("Clinical-AI-Apollo/Medical-NER")

# # Use a pipeline for high-level token classification
# pipe = pipeline("ner", model=new_model, tokenizer=new_tokenizer)  # Updated task to "ner"

# model_load_time = time.time()  # Define model_load_time here

# # Prompt for user input
# prompt = input("\n\nEnter the prompt: ")
# context = retreival.query_hp_book(prompt)  # Corrected module name
# full_prompt = f""" Based on the provided context, answer questions about specific events, characters, and details mentioned in the text. Provide the sources also from where the conclusion is drawn.
# context: {context}
# prompt: {prompt}
# """

# begin = time.time()  # Start timing here

# response = pipe(full_prompt)

# print("\n\n", response)

# end = time.time()
# print(f"\n\nmodel loading time - {begin - model_load_time}")  # Corrected model loading time calculation
# print(f"total time taken - {end - begin}")
import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load environment variables
load_dotenv()

# Load tokenizer and model directly
tokenizer = AutoTokenizer.from_pretrained("Clinical-AI-Apollo/Medical-NER")
model = AutoModelForTokenClassification.from_pretrained("Clinical-AI-Apollo/Medical-NER")

# Set maximum length directly in the tokenizer
tokenizer.model_max_length = 512

# Create a pipeline for token classification
token_classification_pipeline = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer
)

# Define a function for token classification
def classify_tokens(text):
    output = token_classification_pipeline(text)
    return output

# Example usage
text = "Patient has a history of diabetes and hypertension."
output = classify_tokens(text)
print("Text:", text)
print("Token Classification Output:", output)
