
from transformers import AutoModelForCausalLM, AutoTokenizer


import os
import torch
from dotenv import load_dotenv
import time
import transformers
# import retreival

# Load environment variables
load_dotenv()

# Start timing
begin = time.time()

# Model ID and authentication
model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
hf_auth = os.getenv("HF_AUTH")

# Device setup
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

# Model configuration
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

# Quantization configuration
bnb_config = transformers.BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
# Load model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    token=hf_auth
)

# Tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    token=hf_auth
)

# Pipeline setup
pipeline = transformers.pipeline(
   "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    eos_token_id=tokenizer.eos_token_id
)

messages = [
    {"role": "system", "content": '''Act as a review classifier. Your task is to extract sentences from the given review text and classify it as experience-rich text based on given classification rules.
 
        Task: You will be provided a review text. You need to follow the specified flow to classify or nominate the sentence from the review as experience-rich text. Follow the given flow:
            1. Develop a thought: The thought should be clarifying what you need to do next.
            2. Decide an action: Based on the thought that you have developed, you will decide an action.
            3. Observation: State your observation, which includes what you have observed after taking the above-decided action.
            4. Final answer: Give your final answer here.
 
        Classification rules:
            1. The sentence you extract from the review should be in accordance with the given specifications. You will not select any sentence that is not related to the given list of specs.
            2. The sentence should reflect the user experience. This means sentence refers to that comment of the user in review text that specifically mentions something that has come out because of the personal user experience of that user who has given that review.
            3. You will return None when there is no specific user experience mentioned in the review text by the user
            4. You will also return None when you can't find any specification to map for the sentence from the user review text '''},
            # example 1
    {"role": "user", "content": '''Instruction: Extract the experience-rich sentence based on given specification list:
        Specifications list: [Safety(how safe the shredder is to ease), Ease of use (how easy it is to clean the shredder), Number of sheets of shredder, Noise level (how noisy it is to operate the shredder), Maneuvering (how easy it is to carry the shredder from one place to another), Versatile (how versatile is shredder is, in terms of handling various materials), Bin size of shredder] of Shredders
        Review text: I am a huge fan of this shredder. It is super quiet compared to other shredders I have used in the past, and shreds the papers into teeny tiny pieces which is great for preventing identity theft. Safety features are great, if your finger is even remotely near the opening it will stop. Highly recommend!
        AI response:
        '''},  
    {"role": "assistant", "content": '''{
                           "Thought" : "I need to extract experience-rich sentence from user review text based on given specification list.",
                           "Action" : "Find[Finding a text sentence from the review based on specification list]",
                           "Observation" : "I could extract one sentence based on specification -- Noise level (how noisy it is to operate the shredder). The sentence i extracted is -- 'It is super quiet compared to other shredders I have used in the past'. Sentiment for feature (Noise level) in review is positive",
                           "Thought" : "I have found one sentence. I need to check if it is experience-rich",
                           "Action" : "Check experience-rich[Checking if the found sentence is experience-rich or not?]",
                           "Observation": "The user mentioned that he has come to the conclusion of saying it super quiet after using it and also using other shredders too. Thus, it can be said that user has commeted that based on his experience. So, it is experience-rich sentence.",
                           "Thought" : "I have found one experience-rich sentence. Total of 1 experience-rich sentence found till now. Before giving final answer, let me check if i can find more such sentences.",
                           "Action" : "Find[Finding a unique text sentence, a different one, but based on specification list]",
                           "Observation" : "I could extract another sentence based on specification -- Safety(how safe the shredder is to ease). The sentence i extracted is -- 'Safety features are great, if your finger is even remotely near the opening it will stop'. User seems satisfied with respect to feature (safety) so it shows positive sentiment towards safety",
                           "Thought" : "I have found another sentence. I need to check if it is experience-rich",
                           "Action" : "Check experience-rich[Checking if the found sentence is experience-rich or not?]",
                           "Observation": "The user specifically mentions that even if your finger is remotely near the opening it will stop. From this statement it seems that user has used this shredder himself and then found that how safe it is. Thus, it can be said that user has commeted based on his experience. So, it is experience-rich sentence.",
                           "Thought" : "I have found another experience-rich sentence. Total of 2 experience-rich sentence found till now. Before giving final answer, let me check if i can find more such sentences.",
                           "Action" : "Find[Finding a unique text sentence, a different one, but based on specification list]",
                           "Observation" : "I could not find any other sentence related to given specification list",
                           "Thought" : "Since, no new sentence found, i should give final answer with the previous 2 experience-rich sentences.",
                           "Final answer" : [{"feature":"Noise level", "actual_sentence_extracted":"It is super quiet compared to other shredders I have used in the past", "rephrased_it(without experience & observation)":"The user mentioned that this shredder is super quiet compared to other shredders he has used in the past", "sentiment":"positive"}, {"feature":"Safety", "actual_sentence_extracted": "Safety features are great, if your finger is even remotely near the opening it will stop", "rephrased_it(without experience & observation)":"The reviewer appreciates the safety features, specifically the mechanism that stops the shredder if a finger or any object is near the opening.", "sentiment":"positive"}]
        }'''},  # example 2
    {"role": "user", "content": '''Instruction: Extract the experience-rich sentence based on given specification list:
        Specifications list: [Safety(how safe the shredder is to ease), Ease of use (how easy it is to clean the shredder), Number of sheets of shredder, Noise level (how noisy it is to operate the shredder), Maneuvering (how easy it is to carry the shredder from one place to another), Versatile (how versatile is shredder is, in terms of handling various materials), Bin size of shredder] of Shredders
        Review text: This is a great shredder. I’ve never bought one before and needed one. I had a lot of records to shred. I shredded a lot and even got my 80 year old husband to to do it. He can’t do much anymore and this gave him something to do. He figured out how to use and operate the switch without my help., It’s pretty quick and makes a great cut that is not going to allow anyone to put anything back together., I’d recommend this to any home or a small business too.
        AI response:
        '''},  # example 2
    {"role": "assistant", "content": '''{
                           "Thought" : "I need to extract experience-rich sentence from user review text based on given specification",
                           "Action" : "Find[Finding a text sentence from the review based on specification]",
                           "Observation" : "I could not extract any sentence based on given specification. Nothing is related to Battery life",
                           "Thought" : "Since there is no sentence found based on specification, then as per rule 3 of Classification rules, i should return None",
                           "Final answer" : "None"
        }'''},  
    # example 3
    {"role": "user", "content": '''Instruction: Extract the experience-rich sentence based on given specification: 
        Specification: Performance of Tablets(How easy it is to load and run or operate heavy files)
        Review text: I have no problems loading apps. I use it everyday/ evening., The picture is very clear. It operates perfectly when I watch Netflix, or any of the other apps. I use it for my mail as well.
        AI response:
        '''},
        {"role": "assistant", "content": '''{
                           "Thought" : "I need to extract experience-rich sentence from user review text based on given specification.",
                           "Action" : "Find[Finding a text sentence from the review based on specification]",
                           "Observation" : "I could extract two sentences based on specification -- Performance of Tablets(How easy it is to load and run or operate heavy files). The sentence i extracted is -- 'I have no problems loading apps.' and 'It operates perfectly when I watch Netflix, or any of the other apps.',
                           "Thought" : "I have found two sentences. I need to check if 2 sentences are experience-rich",
                           "Action" : "Check experience-rich[Checking if the found 2 sentences are experience-rich or not?]",
                           "Observation": "In first sentence, the user mentioned that he faced no problems while loading. This has come out of user experience, but it would have been experience rich when he would have been more specific. so spefications lag here. Thus, not experience rich

                           In second sentence, the user mentioned that the tablet could easily run netflix and other apps. This has come out of user experience and it also includes the specific information like app name netflix. Thus, it is an experience rich sentence.
                           ",
                           "Thought" : "Out of 2 sentences, only one could qualify as an experience rich sentence. So total 1 experience rich sentence found.",
                           "Final answer" : [{"feature":"Performance of Tablets(How easy it is to load and run or operate heavy files)", "actual_sentence_extracted":"It operates perfectly when I watch Netflix, or any of the other apps.", "rephrased_it(without experience & observation)":"The user indicated that it functions flawlessly when they watch Netflix or any of the other applications.", "sentiment":"positive"}]
        }'''
         },
         {
           "role": "user", "content":'''Instruction: Extract the experience-rich sentence based on given specification: 
        Specification: "[brightness level,color temperature,compatibility,energy efficiency,heat dissipation,,plug-and-play installation,weather resistance]" of headlight-bulbs
        Review text: "Initially, these bulbs shine brilliantly, boasting a power draw of 7.3 Watts.

Its worth noting that the LED bulbs generate some heat.

Yet, after approximately 5 minutes, the intelligent IC driver adjusts the power draw to a more energy-efficient 4.2 Watts, resulting in a 40% reduction in brightness.

This driver serves a crucial role in safeguarding the LEDs from overheating."
        AI response:
        ''' 
         }
]
 
        # try:
        #     input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")

        #     generated_ids = model.generate(input_ids, max_new_tokens=1000, do_sample=True)
        #     final_response = tokenizer.batch_decode(generated_ids) 
        # except:
        #     final_response = "ERROR"
        #     cost_current = 0
        #     print("ERROR OPENAI API")

        # print("FINAL RESPONSE: ", final_response)

# Tokenize prompt
prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

# Define terminators for generation
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Generate text
outputs = pipeline(
    prompt,
    max_new_tokens=1000,
    eos_token_id=terminators,           
    do_sample=True,
    temperature=0.01,
    top_p=0.9,
)
print("FINAL RESPONSE" , outputs[0]["generated_text"][len(prompt):])


# class GetReviewSnipets:

#     # Function to be used in old case when we are revieving full text
#     def get_snippet_full_review(self):

#         model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'                
#         model = AutoModelForCausalLM.from_pretrained(model_name)
#         tokenizer = AutoTokenizer.from_pretrained(model_name)

#         messages = [
#     {"role": "system", "content": '''Act as a review classifier. Your task is to extract sentences from the given review text and classify it as experience-rich text based on given classification rules.
 
#         Task: You will be provided a review text. You need to follow the specified flow to classify or nominate the sentence from the review as experience-rich text. Follow the given flow:
#             1. Develop a thought: The thought should be clarifying what you need to do next.
#             2. Decide an action: Based on the thought that you have developed, you will decide an action.
#             3. Observation: State your observation, which includes what you have observed after taking the above-decided action.
#             4. Final answer: Give your final answer here.
 
#         Classification rules:
#             1. The sentence you extract from the review should be in accordance with the given specifications. You will not select any sentence that is not related to the given list of specs.
#             2. The sentence should reflect the user experience. This means sentence refers to that comment of the user in review text that specifically mentions something that has come out because of the personal user experience of that user who has given that review.
#             3. You will return None when there is no specific user experience mentioned in the review text by the user
#             4. You will also return None when you can't find any specification to map for the sentence from the user review text '''},
#             # example 1
#     {"role": "user", "content": '''Instruction: Extract the experience-rich sentence based on given specification list:
#         Specifications list: [Safety(how safe the shredder is to ease), Ease of use (how easy it is to clean the shredder), Number of sheets of shredder, Noise level (how noisy it is to operate the shredder), Maneuvering (how easy it is to carry the shredder from one place to another), Versatile (how versatile is shredder is, in terms of handling various materials), Bin size of shredder] of Shredders
#         Review text: I am a huge fan of this shredder. It is super quiet compared to other shredders I have used in the past, and shreds the papers into teeny tiny pieces which is great for preventing identity theft. Safety features are great, if your finger is even remotely near the opening it will stop. Highly recommend!
#         AI response:
#         '''},  
#     {"role": "assistant", "content": '''{
#                            "Thought" : "I need to extract experience-rich sentence from user review text based on given specification list.",
#                            "Action" : "Find[Finding a text sentence from the review based on specification list]",
#                            "Observation" : "I could extract one sentence based on specification -- Noise level (how noisy it is to operate the shredder). The sentence i extracted is -- 'It is super quiet compared to other shredders I have used in the past'. Sentiment for feature (Noise level) in review is positive",
#                            "Thought" : "I have found one sentence. I need to check if it is experience-rich",
#                            "Action" : "Check experience-rich[Checking if the found sentence is experience-rich or not?]",
#                            "Observation": "The user mentioned that he has come to the conclusion of saying it super quiet after using it and also using other shredders too. Thus, it can be said that user has commeted that based on his experience. So, it is experience-rich sentence.",
#                            "Thought" : "I have found one experience-rich sentence. Total of 1 experience-rich sentence found till now. Before giving final answer, let me check if i can find more such sentences.",
#                            "Action" : "Find[Finding a unique text sentence, a different one, but based on specification list]",
#                            "Observation" : "I could extract another sentence based on specification -- Safety(how safe the shredder is to ease). The sentence i extracted is -- 'Safety features are great, if your finger is even remotely near the opening it will stop'. User seems satisfied with respect to feature (safety) so it shows positive sentiment towards safety",
#                            "Thought" : "I have found another sentence. I need to check if it is experience-rich",
#                            "Action" : "Check experience-rich[Checking if the found sentence is experience-rich or not?]",
#                            "Observation": "The user specifically mentions that even if your finger is remotely near the opening it will stop. From this statement it seems that user has used this shredder himself and then found that how safe it is. Thus, it can be said that user has commeted based on his experience. So, it is experience-rich sentence.",
#                            "Thought" : "I have found another experience-rich sentence. Total of 2 experience-rich sentence found till now. Before giving final answer, let me check if i can find more such sentences.",
#                            "Action" : "Find[Finding a unique text sentence, a different one, but based on specification list]",
#                            "Observation" : "I could not find any other sentence related to given specification list",
#                            "Thought" : "Since, no new sentence found, i should give final answer with the previous 2 experience-rich sentences.",
#                            "Final answer" : [{"feature":"Noise level", "actual_sentence_extracted":"It is super quiet compared to other shredders I have used in the past", "rephrased_it(without experience & observation)":"The user mentioned that this shredder is super quiet compared to other shredders he has used in the past", "sentiment":"positive"}, {"feature":"Safety", "actual_sentence_extracted": "Safety features are great, if your finger is even remotely near the opening it will stop", "rephrased_it(without experience & observation)":"The reviewer appreciates the safety features, specifically the mechanism that stops the shredder if a finger or any object is near the opening.", "sentiment":"positive"}]
#         }'''},  # example 2
#     {"role": "user", "content": '''Instruction: Extract the experience-rich sentence based on given specification list:
#         Specifications list: [Safety(how safe the shredder is to ease), Ease of use (how easy it is to clean the shredder), Number of sheets of shredder, Noise level (how noisy it is to operate the shredder), Maneuvering (how easy it is to carry the shredder from one place to another), Versatile (how versatile is shredder is, in terms of handling various materials), Bin size of shredder] of Shredders
#         Review text: This is a great shredder. I’ve never bought one before and needed one. I had a lot of records to shred. I shredded a lot and even got my 80 year old husband to to do it. He can’t do much anymore and this gave him something to do. He figured out how to use and operate the switch without my help., It’s pretty quick and makes a great cut that is not going to allow anyone to put anything back together., I’d recommend this to any home or a small business too.
#         AI response:
#         '''},  # example 2
#     {"role": "assistant", "content": '''{
#                            "Thought" : "I need to extract experience-rich sentence from user review text based on given specification",
#                            "Action" : "Find[Finding a text sentence from the review based on specification]",
#                            "Observation" : "I could not extract any sentence based on given specification. Nothing is related to Battery life",
#                            "Thought" : "Since there is no sentence found based on specification, then as per rule 3 of Classification rules, i should return None",
#                            "Final answer" : "None"
#         }'''},  
#     # example 3
#     {"role": "user", "content": '''Instruction: Extract the experience-rich sentence based on given specification: 
#         Specification: Performance of Tablets(How easy it is to load and run or operate heavy files)
#         Review text: I have no problems loading apps. I use it everyday/ evening., The picture is very clear. It operates perfectly when I watch Netflix, or any of the other apps. I use it for my mail as well.
#         AI response:
#         '''},
#         {"role": "assistant", "content": '''{
#                            "Thought" : "I need to extract experience-rich sentence from user review text based on given specification.",
#                            "Action" : "Find[Finding a text sentence from the review based on specification]",
#                            "Observation" : "I could extract two sentences based on specification -- Performance of Tablets(How easy it is to load and run or operate heavy files). The sentence i extracted is -- 'I have no problems loading apps.' and 'It operates perfectly when I watch Netflix, or any of the other apps.',
#                            "Thought" : "I have found two sentences. I need to check if 2 sentences are experience-rich",
#                            "Action" : "Check experience-rich[Checking if the found 2 sentences are experience-rich or not?]",
#                            "Observation": "In first sentence, the user mentioned that he faced no problems while loading. This has come out of user experience, but it would have been experience rich when he would have been more specific. so spefications lag here. Thus, not experience rich

#                            In second sentence, the user mentioned that the tablet could easily run netflix and other apps. This has come out of user experience and it also includes the specific information like app name netflix. Thus, it is an experience rich sentence.
#                            ",
#                            "Thought" : "Out of 2 sentences, only one could qualify as an experience rich sentence. So total 1 experience rich sentence found.",
#                            "Final answer" : [{"feature":"Performance of Tablets(How easy it is to load and run or operate heavy files)", "actual_sentence_extracted":"It operates perfectly when I watch Netflix, or any of the other apps.", "rephrased_it(without experience & observation)":"The user indicated that it functions flawlessly when they watch Netflix or any of the other applications.", "sentiment":"positive"}]
#         }'''
#          },
#          {
#            "role": "user", "content":'''Instruction: Extract the experience-rich sentence based on given specification: 
#         Specification: "[brightness level,color temperature,compatibility,energy efficiency,heat dissipation,,plug-and-play installation,weather resistance]" of headlight-bulbs
#         Review text: "Initially, these bulbs shine brilliantly, boasting a power draw of 7.3 Watts.

# Its worth noting that the LED bulbs generate some heat.

# Yet, after approximately 5 minutes, the intelligent IC driver adjusts the power draw to a more energy-efficient 4.2 Watts, resulting in a 40% reduction in brightness.

# This driver serves a crucial role in safeguarding the LEDs from overheating."
#         AI response:
#         ''' 
#          }
# ]
 
#         try:
#             input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")

#             generated_ids = model.generate(input_ids, max_new_tokens=1000, do_sample=True)
#             final_response = tokenizer.batch_decode(generated_ids) 
#         except:
#             final_response = "ERROR"
#             cost_current = 0
#             print("ERROR OPENAI API")

#         print("FINAL RESPONSE: ", final_response)
 
#         return [final_response, 0]


#     # Function for new chunked review
#     def get_snippet_chunked_review(self): 

#         model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
#         model = AutoModelForCausalLM.from_pretrained(model_name)
#         tokenizer = AutoTokenizer.from_pretrained(model_name)

#         messages = [
#             {"role": "system", "content": '''Act as a review analyser, who can find if the given review is experience-rich based on the exclusive information, the user has mentioned.

#         Task: You will be provided a review text. You need to follow the specified flow to classify or nominate the sentence from the review as experience-rich text. Follow the given flow: 
#             1. Develop a thought: The thought should be clarifying what you need to do next. 
#             2. Decide an action: Based on the thought that you have developed, you will decide an action.
#             3. Observation: State your observation, which includes what you have observed after taking the above-decided action. 
#             4. Final answer: Give your final answer here. 

#         Classification rules: 
#             1. The sentence should reflect the user experience. This means sentence refers to that comment of the user in review text that specifically mentions something that has come out because of the personal user experience of that user who has given that review. The sentence should contain the exclusive information that has come out of user experience. 
#             2. You will return None when there is no specific user experience mentioned in the review text by the user
#             3. You will also return None when you can't find any specification to map for the sentence from the user review text
#             4. You should only classify the review as an experience rich sentence only when you find it to have specified information or exclusive information.'''
#              },
#             {"role": "user", "content": '''Instruction: Extract the experience-rich sentence based on given specification: 
#         Specification: Battery life of Tablets
#         Review text: Battery life varies given the task being performed of course, but is very fair and close to expectations. Overall a very good tablet well worth the purchase price.
#         AI response:
#         '''},  # example 1
#             {"role": "assistant", "content": '''{
#                            "Thought" : "I need to extract experience-rich sentence from user review text based on given specification.",
#                            "Action" : "Find[Finding a text sentence from the review based on specification]",
#                            "Observation" : "I could extract one sentence based on specification -- Battery life. The sentence i extracted is -- 'Battery life varies given the task being performed of course, but is very fair and close to expectations'. Sentiment for feature (Battery life) in review is positive",
#                            "Thought" : "I have found one sentence. I need to check if it is experience-rich",
#                            "Action" : "Check experience-rich[Checking if the found sentence is experience-rich or not?]",
#                            "Observation": "In first sentence, the user mentioned that battery life will vary for differnt tasks, he also mentioned that it is close to expectations. This has come out of user experience, but it has no specifications. Despite having no specification, it seems to be exclsuive, since the user said it is close to expectation. So, as per rule 4 of classification rules it has exclusive information so it is experience rich.",
#                            "Thought" : "I have found one experience-rich sentence. I should give final answer with the 1 experience-rich sentence.",
#                            "Final answer" : [{"feature":"Battery life", "actual_sentence_extracted":"Battery life varies given the task being performed of course, but is very fair and close to expectations", "rephrased_it(without experience & observation)":"The user mentioned that the battery life varies depending on the task at hand. However, it is close to his expectations.", "sentiment":"positive"}]
#         }'''},  # example 1
#             {"role": "user", "content": '''Instruction: Extract the experience-rich sentence based on given specification: 
#         Specification: Battery life of Tablets
#         Review text: 10 years ago this would have been okay but your phone is faster. I couldnt recommend this except for the simplest of applications like reading
#         AI response:
#         '''},  # example 2
#             {"role": "assistant", "content": '''{
#                            "Thought" : "I need to extract experience-rich sentence from user review text based on given specification",
#                            "Action" : "Find[Finding a text sentence from the review based on specification]",
#                            "Observation" : "I could not extract any sentence based on given specification. Nothing is related to Battery life",
#                            "Thought" : "Since there is no sentence found based on specification, then as per rule 3 of Classification rules, i should return None",
#                            "Final answer" : "None"
#         }'''},  # example 2
#             {"role": "user", "content": '''Instruction: Extract the experience-rich sentence based on given specification: 
#         Specification: Performance of Tablets(How easy it is to load and run or operate heavy files)
#         Review text: I have no problems loading apps. I use it everyday/ evening., The picture is very clear. It operates perfectly when I watch Netflix, or any of the other apps. I use it for my mail as well.
#         AI response:
#         '''},
# {"role": "assistant", "content":  '''{
#                            "Thought" : "I need to extract experience-rich sentence from user review text based on given specification.",
#                            "Action" : "Find[Finding a text sentence from the review based on specification]",
#                            "Observation" : "I could extract two sentences based on specification -- Performance of Tablets(How easy it is to load and run or operate heavy files). The sentence i extracted is -- 'I have no problems loading apps.' and 'It operates perfectly when I watch Netflix, or any of the other apps.',
#                            "Thought" : "I have found two sentences. I need to check if 2 sentences are experience-rich",
#                            "Action" : "Check experience-rich[Checking if the found 2 sentences are experience-rich or not?]",
#                            "Observation": "In first sentence, the user mentioned that he faced no problems while loading. This has come out of user experience, but it would have been experience rich when he would have been more specific. so spefications lag here. Thus, not experience rich

#                            In second sentence, the user mentioned that the tablet could easily run netflix and other apps. This has come out of user experience and it also includes the specific information like app name netflix. Thus, it is an experience rich sentence.
#                            ",
#                            "Thought" : "Out of 2 sentences, only one could qualify as an experience rich sentence. So total 1 experience rich sentence found.",
#                            "Final answer" : [{"feature":"Performance of Tablets(How easy it is to load and run or operate heavy files)", "actual_sentence_extracted":"It operates perfectly when I watch Netflix, or any of the other apps.", "rephrased_it(without experience & observation)":"The user indicated that it functions flawlessly when they watch Netflix or any of the other applications.", "sentiment":"positive"}]
#         }'''},
#         {"role": "user", "content":'''Instruction: Extract the experience-rich sentence based on given specification: 
#         Specification: "[brightness level,color temperature,compatibility,energy efficiency,heat dissipation,,plug-and-play installation,weather resistance]" of headlight-bulbs
#         Review text: "Initially, these bulbs shine brilliantly, boasting a power draw of 7.3 Watts.

# Its worth noting that the LED bulbs generate some heat.

# Yet, after approximately 5 minutes, the intelligent IC driver adjusts the power draw to a more energy-efficient 4.2 Watts, resulting in a 40% reduction in brightness.

# This driver serves a crucial role in safeguarding the LEDs from overheating."
#         AI response:
#         '''}
       
#         ]

#         try:
#             input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
#             generated_ids = model.generate(input_ids, max_new_tokens=1000, do_sample=True)
#             final_response = tokenizer.batch_decode(generated_ids)

#         except:
#             final_response = "ERROR"
#             cost_current = 0

#         return [final_response, 0]


# obj = GetReviewSnipets()

# # obj.get_snippet_chunked_review()
# obj.get_snippet_full_review()