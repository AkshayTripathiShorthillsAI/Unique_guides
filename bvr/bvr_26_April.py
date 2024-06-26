import os, glob, time, ast
from dotenv import load_dotenv
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
)

import os
import torch
from dotenv import load_dotenv
import time

import torch

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

load_dotenv()

PIDS_FILE = "/home/ubuntu/rag_llama/akshay/bvr/ASIN_values_2024-04-24.csv"  
df_pid = pd.read_csv(PIDS_FILE)

features_df = pd.read_csv("/home/ubuntu/rag_llama/akshay/bvr/features_output.csv")   

# Model ID and authentication
hf_auth = os.getenv("HF_AUTH")

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
          model_name ,
          quantization_config=quant_config,
          device_map="auto",
        # attn_implementation="flash_attention_2",
          torch_dtype=torch.bfloat16
          )

tokenizer = AutoTokenizer.from_pretrained(model_name , trust_remote_code=True)


tokenizer.eos_token ='<|eot_id|>'

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# peft_model = PeftModel.from_pretrained(model, model_checkpoint)

model
#5888

class GetReviewSnipets:

    # Function to be used in old case when we are revieving full text
    def get_snippet_full_review(self, review_text, category_features, category_name):
        print("in the get snippet full review function")

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
           {"role": "user", "content":f'''Instruction: Extract the experience-rich sentence based on given specification list:
           Specifications list: {category_features} of {category_name}
           Review text: {review_text}
           ''' }
        ]
 
        prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
        )
        # print(prompt)

        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
        output_encode= model.generate(input_ids=input_ids, pad_token_id=tokenizer.pad_token_id,max_new_tokens=3000) 
        # output_encode= ""
        output = tokenizer.decode(output_encode[0], skip_special_tokens=False)
        
        print("Output :" ,output)
            
        return [output, 0]


    # Function for new chunked review
    def get_snippet_chunked_review(self, review_text, category_features, category_name): 

        messages = [
            {"role": "system", "content": '''Act as a review analyser, who can find if the given review is experience-rich based on the exclusive information, the user has mentioned.

            Task: You will be provided a review text. You need to follow the specified flow to classify or nominate the sentence from the review as experience-rich text. Follow the given flow: 
            1. Develop a thought: The thought should be clarifying what you need to do next. 
            2. Decide an action: Based on the thought that you have developed, you will decide an action.
            3. Observation: State your observation, which includes what you have observed after taking the above-decided action. 
            4. Final answer: Give your final answer here. 

            Classification rules: 
            1. The sentence should reflect the user experience. This means sentence refers to that comment of the user in review text that specifically mentions something that has come out because of the personal user experience of that user who has given that review. The sentence should contain the exclusive information that has come out of user experience. 
            2. You will return None when there is no specific user experience mentioned in the review text by the user
            3. You will also return None when you can't find any specification to map for the sentence from the user review text
            4. You should only classify the review as an experience rich sentence only when you find it to have specified information or exclusive information.'''
            },
            {"role": "user", "content": '''Instruction: Extract the experience-rich sentence based on given specification: 
             Specification: Battery life of Tablets
             Review text: Battery life varies given the task being performed of course, but is very fair and close to expectations. Overall a very good tablet well worth the purchase price.
            '''},  
            # example 1
            {"role": "assistant", "content": '''
                           "Thought" : "I need to extract experience-rich sentence from user review text based on given specification.",
                           "Action" : "Find[Finding a text sentence from the review based on specification]",
                           "Observation" : "I could extract one sentence based on specification -- Battery life. The sentence i extracted is -- 'Battery life varies given the task being performed of course, but is very fair and close to expectations'. Sentiment for feature (Battery life) in review is positive",
                           "Thought" : "I have found one sentence. I need to check if it is experience-rich",
                           "Action" : "Check experience-rich[Checking if the found sentence is experience-rich or not?]",
                           "Observation": "In first sentence, the user mentioned that battery life will vary for differnt tasks, he also mentioned that it is close to expectations. This has come out of user experience, but it has no specifications. Despite having no specification, it seems to be exclsuive, since the user said it is close to expectation. So, as per rule 4 of classification rules it has exclusive information so it is experience rich.",
                           "Thought" : "I have found one experience-rich sentence. I should give final answer with the 1 experience-rich sentence.",
                           "Final answer" : [{"feature":"Battery life", "actual_sentence_extracted":"Battery life varies given the task being performed of course, but is very fair and close to expectations", "rephrased_it(without experience & observation)":"The user mentioned that the battery life varies depending on the task at hand. However, it is close to his expectations.", "sentiment":"positive"}]
            '''},

            {"role": "user", "content": '''Instruction: Extract the experience-rich sentence based on given specification: 
            Specification: Battery life of Tablets
            Review text: 10 years ago this would have been okay but your phone is faster. I couldnt recommend this except for the simplest of applications like reading
            
            '''},  
            # example 2
            {"role": "assistant", "content": '''
                           "Thought" : "I need to extract experience-rich sentence from user review text based on given specification",
                           "Action" : "Find[Finding a text sentence from the review based on specification]",
                           "Observation" : "I could not extract any sentence based on given specification. Nothing is related to Battery life",
                           "Thought" : "Since there is no sentence found based on specification, then as per rule 3 of Classification rules, i should return None",
                           "Final answer" : "None"
            '''}, 
            {"role": "user", "content": '''Instruction: Extract the experience-rich sentence based on given specification: 
            Specification: Performance of Tablets(How easy it is to load and run or operate heavy files)
            Review text: I have no problems loading apps. I use it everyday/ evening., The picture is very clear. It operates perfectly when I watch Netflix, or any of the other apps. I use it for my mail as well.
            '''},
            {"role": "assistant", "content":  '''{
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
            }'''},
            {"role": "user", "content":f'''Instruction: Extract the experience-rich sentence based on given specification: 
            Specification: {category_features} of {category_name}
            Review text: {review_text}
            '''}
        ]

     
        prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
        )
        # print(prompt)

        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
        output_encode= model.generate(input_ids=input_ids, pad_token_id=tokenizer.pad_token_id,
                                      max_new_tokens=1000
                                      ) 
        # output_encode= ""
        output = tokenizer.decode(output_encode[0], skip_special_tokens=True)

        final_response = output[len(prompt):]
        print(output)
            
        return [final_response, 0]


def start_process(reviews, is_helpful, rating, category_features, frequency_list, category, asin_value):
    run = GetReviewSnipets()
    reviews_list = []
    all_reviews = reviews
    cost_for_all_reviews = []
    rephrased_snippets_for_all_reviews = []
    actual_response_for_all_reviews = []
    time_taken_list = []
    index = 0
    is_helpfuls_list = []
    ratings_list = []
    review_number = []

    for each_review_text, frequency in zip(all_reviews, frequency_list):
        st = time.time()
        output = []

        # Hardcoded
        if frequency == 1:
            output = run.get_snippet_chunked_review(each_review_text, category_features, category)
        else:
            output = run.get_snippet_full_review(each_review_text, category_features, category)

        tt = time.time() - st

        actual_response = output[0]
        cost_for_each_review = output[1]
        snips = []
        rephrased_snippets = ""

        if actual_response == "ERROR":
            rephrased_snippets = ""
        else:
            try:
                rephrased_snippets = output[0].split('Final answer" : ')[1]
                print("Rephrased Snippets" , rephrased_snippets)

            except Exception as e:
                print("Error in rephrasing snip" + str(e))
                rephrased_snippets = ""

            t = rephrased_snippets[:-1]
            t = t.strip()

            try:
                snips = ast.literal_eval(t)
            except:
                rephrased_snippets = t

        if type(snips) == list:
            flag = True
            for snip in snips:
                reviews_list.append(each_review_text)
                is_helpfuls_list.append(is_helpful[index])
                ratings_list.append(rating[index])
                rephrased_snippets_for_all_reviews.append(snip)
                actual_response_for_all_reviews.append(actual_response)
                review_number.append(index + 1)

                if flag:
                    cost_for_all_reviews.append(cost_for_each_review)
                    time_taken_list.append(tt)
                    flag = False
                else:
                    cost_for_all_reviews.append("")
                    time_taken_list.append("")
        else:
            reviews_list.append(each_review_text)
            is_helpfuls_list.append(is_helpful[index])
            ratings_list.append(rating[index])

            if snips.lower() == "none":
                rephrased_snippets = ""

            rephrased_snippets_for_all_reviews.append(rephrased_snippets)
            actual_response_for_all_reviews.append(actual_response)
            cost_for_all_reviews.append(cost_for_each_review)
            time_taken_list.append(tt)
            review_number.append(index + 1)

        # Access the corresponding element in asin_value based on the index
        current_asin_value = asin_value[index]
        index = index + 1
        time.sleep(2)

    final_data = {'Review text': reviews_list,
                  'is_helpful': is_helpfuls_list,
                  'rating': ratings_list,
                  'Actual Response': actual_response_for_all_reviews,
                  'Review Number': review_number,
                  'Rephrased Snippets': rephrased_snippets_for_all_reviews,
                  'Cost per review': cost_for_all_reviews,
                  'Time Taken': time_taken_list,
                  'amazon_page': current_asin_value}

    print(final_data['amazon_page'],final_data['Review text'])

    final_dataframe = pd.DataFrame(final_data)

    return final_dataframe

desired_column_order = ["is_helpful", "rating", "Actual Response", "Review Number", "Cost per review",	"Time Taken", "amazon_page", "Review text", "Rephrased Snippets", "aspect", "QA_Content", "Accept"]

def extract_rephrased(data):
    #print(data)
    if type(data) != dict:
         return ""
    try:
        return data.get('rephrased_it(without experience & observation)',"")
        #data = json.loads(json_data)
        #exit()
    except Exception as e:
        #print("Unable to parse json file returning it as it is")
        return data

# All categories
CATEGORY = df_pid['category_slug'].unique()
# CATEGORY = ["bullet-surveillance-cameras"]
#CATEGORY_2 = ["craft-scissors", "glue-sticks"]
#CATEGORY_3 = ["jump-starters", "stand-up-paddleboards"]
#CATEGORY_4 = ["tattoo-kits", "whey-protein-powders"]

for category in CATEGORY:
    
    files_and_folders = glob.glob(f'{"/home/ubuntu/rag_llama/akshay/bvr/filtered_12_march_2024"}/{category}/**', recursive=True)    # Dumping folder of filtered features
    files = [f for f in files_and_folders if f.endswith('.xlsx')]
    print(files)
    #category_features = str(list(features_df[features_df["category_slug"]==category]["features"]))
    category_features = list(features_df[features_df["category_slug"]==category]["features"])
    print(category_features)
    if category_features == "":
        print(f"Category Features missing for catrgory {category} in csv file")
        exit()
    
    category_snippet_folder= f'{os.getenv("CATEGORY_SNIPPET_FOLDER")}/{category}'      # Dumping folder of final Snippets 
    if not os.path.exists(category_snippet_folder):
        os.makedirs(category_snippet_folder)
        
    for file in files:
        listing_id = file.split("/")[-1].split(".")[0]
        
        print(file)
        df = pd.read_excel(file)

        combined_data_frame = pd.DataFrame()

        for aspect in category_features:
            asin_value=df[df["aspect"]==aspect]['asin'].to_list()
            review_list = df[df["aspect"]==aspect]['openAiText'].to_list()
            is_helpful = df[df["aspect"]==aspect]['is_Helpful'].to_list()
            rating = df[df["aspect"]==aspect]["rating"].to_list()
            frequency = (df[df["aspect"]==aspect]["frequency"]).to_list()
            
            # print(asin_value)
            # print(asin_value[0])
            
            
            if len(review_list)>0:
                final_dataframe = start_process(review_list, is_helpful,rating, aspect,frequency, category,asin_value)
                # final_dataframe["amazon_page"]=f"https://amazon.com/dp/{asin_value[0]}"
                final_dataframe["amazon_page"]="https://amazon.com/dp/B007F9XHAY"
                final_dataframe["aspect"] = aspect
                for column in desired_column_order:
                    if column not in final_dataframe.columns:
                        final_dataframe[column] = ''
                final_dataframe['QA_Content'] = final_dataframe['Rephrased Snippets'].apply(extract_rephrased)
                final_dataframe = final_dataframe[desired_column_order]
                combined_data_frame = pd.concat([combined_data_frame, final_dataframe], ignore_index=True)
                
        combined_data_frame.to_excel(f"{category_snippet_folder}/{listing_id}_snippets.xlsx")


