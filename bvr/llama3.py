pip install flash-attn --no-build-isolation

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)
print(prompt)

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
output_encode= model.generate(input_ids=input_ids, pad_token_id=tokenizer.pad_token_id,max_length=920) 
output = tokenizer.decode(output_encode[0], skip_special_tokens=True)
print(output)