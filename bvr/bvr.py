
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        Review text: This is a great shredder. I have never bought one before and needed one. I had a lot of records to shred. I shredded a lot and even got my 80 year old husband to to do it. He can’t do much anymore and this gave him something to do. He figured out how to use and operate the switch without my help., It’s pretty quick and makes a great cut that is not going to allow anyone to put anything back together., I’d recommend this to any home or a small business too.
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
        }'''}
]



input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")

generated_ids = model.generate(input_ids, max_new_tokens=1000, do_sample=True)
outputs = tokenizer.batch_decode(generated_ids)
print(outputs)