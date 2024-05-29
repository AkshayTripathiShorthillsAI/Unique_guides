# Import statements
import os
import glob
import time
import ast
from dotenv import load_dotenv
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig

class ReviewSnippets:
    def __init__(self):
        self.df_pid = None
        self.features_df = None
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        load_dotenv()


    def load_configuration(self):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
    
    def load_dataframes(self):
        self.df_pid = pd.read_csv("/home/ubuntu/rag_llama/akshay/bvr/ASIN_values_2024-04-24.csv")
        self.features_df = pd.read_csv("/home/ubuntu/rag_llama/akshay/bvr/features_output.csv")   

    def load_model(self):
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        hf_auth = os.getenv("HF_AUTH")

        # Device setup
        # device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

        # Model configuration
        model_config = AutoConfig.from_pretrained(
            model_name,
            use_auth_token=hf_auth
        )

        # Quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            token=hf_auth
        )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_auth
        )

        # Pipeline setup
        self.pipeline = transformers.pipeline(
        "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            eos_token_id=self.tokenizer.eos_token_id
        )


    # Function to be used in old case when we are revieving full text
    def get_snippet_full_review(self, review_text, category_features, category_name):

        messages = [
            {
                "role": "system",
                "content": '''As a review analyzer, your task is to identify experience-rich sentences based on the user's provided information. Follow these steps:
                
                1. Develop a thought: Clarify your next steps.
                2. Decide an action: Determine how to proceed based on your thought.
                3. Observation: State what you observe after taking the action.
                4. Final answer: Provide your conclusion.
                
                Classification rules:
                1. The sentence should reflect user experience, containing exclusive information.
                2. Return None if no specific user experience is mentioned.
                3. Return None if no relevant information can be found.
                4. Classify as experience-rich only if it contains specified or exclusive information.
                '''
            },
            {
                "role": "user",
                "content": '''Extract the experience-rich sentence based on the given specification:
                Specification: Battery life of Tablets
                Review text: Battery life varies given the task being performed of course, but is very fair and close to expectations. Overall a very good tablet well worth the purchase price.
                '''
            },
            {
                "role": "assistant",
                "content": '''
                "Thought": "Identify an experience-rich sentence based on the specified criteria."
                "Action": "Find a sentence related to the battery life."
                "Observation": "Extracted sentence: 'Battery life varies given the task being performed of course, but is very fair and close to expectations.' Positive sentiment observed."
                "Thought": "Check if the extracted sentence is experience-rich."
                "Action": "Assess if the sentence reflects exclusive user experience."
                "Observation": "Although no specific specification is mentioned, the user's satisfaction indicates a personalized experience. Hence, it qualifies as experience-rich."
                "Final answer": [{"feature": "Battery life", "actual_sentence_extracted": "Battery life varies given the task being performed of course, but is very fair and close to expectations", "rephrased_it(without experience & observation)": "The user mentioned that the battery life varies depending on the task at hand. However, it meets expectations.", "sentiment": "positive"}]
                '''
            },
            {
                "role": "user",
                "content": '''Extract the experience-rich sentence based on the given specification:
                Specification: Battery life of Tablets
                Review text: 10 years ago this would have been okay but your phone is faster. I couldn't recommend this except for the simplest of applications like reading
                '''
            },
            {
                "role": "assistant",
                "content": '''
                "Thought": "Identify an experience-rich sentence based on the specified criteria."
                "Action": "Search for a sentence related to battery life."
                "Observation": "No sentence found relevant to battery life."
                "Thought": "As per classification rules, return None."
                "Final answer": "None"
                '''
            },
            {
                "role": "user",
                "content": '''Extract the experience-rich sentence based on the given specification:
                Specification: Performance of Tablets (How easy it is to load and run or operate heavy files)
                Review text: I have no problems loading apps. I use it every day/evening. The picture is very clear. It operates perfectly when I watch Netflix or any of the other apps. I use it for my mail as well.
                '''
            },
            {
                "role": "assistant",
                "content": '''
                "Thought": "Identify an experience-rich sentence based on the specified criteria."
                "Action": "Find sentences related to tablet performance."
                "Observation": "Two sentences extracted: 'I have no problems loading apps.' and 'It operates perfectly when I watch Netflix or any of the other apps.'"
                "Thought": "Check if the extracted sentences are experience-rich."
                "Action": "Assess if each sentence reflects exclusive user experience."
                "Observation": "First sentence lacks specificity; second sentence mentions the tablet running Netflix smoothly, which is an experience-rich detail."
                "Thought": "Out of 2 sentences, only 1 qualifies as experience-rich."
                "Final answer": [{"feature": "Performance of Tablets (How easy it is to load and run or operate heavy files)", "actual_sentence_extracted": "It operates perfectly when I watch Netflix, or any of the other apps.", "rephrased_it(without experience & observation)": "The user indicated that it functions flawlessly when they watch Netflix or any of the other applications.", "sentiment": "positive"}]
                '''
            },
            {
                "role": "user",
                "content": f'''Extract the experience-rich sentence based on the given specification:
                Specification: {category_features} of {category_name}
                Review text: {review_text}
                '''
            }
        ]

        try:   
            # Tokenize prompt
            prompt = self.pipeline.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
            )

            # Define terminators for generation
            terminators = [
                self.pipeline.tokenizer.eos_token_id,
                self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            # Generate text
            outputs = self.pipeline(
                prompt,
                max_new_tokens=1000,
                eos_token_id=terminators,           
                do_sample=True,
                temperature=0.01,
                top_p=0.9,
            )
            # print("FINAL RESPONSE" , outputs[0]["generated_text"][len(prompt):])
            final_response = outputs[0]["generated_text"][len(prompt):]
        except:
            final_response = "ERROR"
            cost_current = 0
            print("ERROR OPENAI API")

        print("FINAL RESPONSE: ", outputs)
 
        return [final_response, 0]


    # Function for new chunked review
    def get_snippet_chunked_review(self, review_text, category_features, category_name): 

        messages = [
            {
                "role": "system",
                "content": '''As a review analyzer, your task is to identify experience-rich sentences based on the user's provided information. Follow these steps:
                
                1. Develop a thought: Clarify your next steps.
                2. Decide an action: Determine how to proceed based on your thought.
                3. Observation: State what you observe after taking the action.
                4. Final answer: Provide your conclusion.
                
                Classification rules:
                1. The sentence should reflect user experience, containing exclusive information.
                2. Return None if no specific user experience is mentioned.
                3. Return None if no relevant information can be found.
                4. Classify as experience-rich only if it contains specified or exclusive information.
                '''
            },
            {
                "role": "user",
                "content": '''Extract the experience-rich sentence based on the given specification:
                Specification: Battery life of Tablets
                Review text: Battery life varies given the task being performed of course, but is very fair and close to expectations. Overall a very good tablet well worth the purchase price.
                '''
            },
            {
                "role": "assistant",
                "content": '''
                "Thought": "Identify an experience-rich sentence based on the specified criteria."
                "Action": "Find a sentence related to the battery life."
                "Observation": "Extracted sentence: 'Battery life varies given the task being performed of course, but is very fair and close to expectations.' Positive sentiment observed."
                "Thought": "Check if the extracted sentence is experience-rich."
                "Action": "Assess if the sentence reflects exclusive user experience."
                "Observation": "Although no specific specification is mentioned, the user's satisfaction indicates a personalized experience. Hence, it qualifies as experience-rich."
                "Final answer": [{"feature": "Battery life", "actual_sentence_extracted": "Battery life varies given the task being performed of course, but is very fair and close to expectations", "rephrased_it(without experience & observation)": "The user mentioned that the battery life varies depending on the task at hand. However, it meets expectations.", "sentiment": "positive"}]
                '''
            },
            {
                "role": "user",
                "content": '''Extract the experience-rich sentence based on the given specification:
                Specification: Battery life of Tablets
                Review text: 10 years ago this would have been okay but your phone is faster. I couldn't recommend this except for the simplest of applications like reading
                '''
            },
            {
                "role": "assistant",
                "content": '''
                "Thought": "Identify an experience-rich sentence based on the specified criteria."
                "Action": "Search for a sentence related to battery life."
                "Observation": "No sentence found relevant to battery life."
                "Thought": "As per classification rules, return None."
                "Final answer": "None"
                '''
            },
            {
                "role": "user",
                "content": '''Extract the experience-rich sentence based on the given specification:
                Specification: Performance of Tablets (How easy it is to load and run or operate heavy files)
                Review text: I have no problems loading apps. I use it every day/evening. The picture is very clear. It operates perfectly when I watch Netflix or any of the other apps. I use it for my mail as well.
                '''
            },
            {
                "role": "assistant",
                "content": '''
                "Thought": "Identify an experience-rich sentence based on the specified criteria."
                "Action": "Find sentences related to tablet performance."
                "Observation": "Two sentences extracted: 'I have no problems loading apps.' and 'It operates perfectly when I watch Netflix or any of the other apps.'"
                "Thought": "Check if the extracted sentences are experience-rich."
                "Action": "Assess if each sentence reflects exclusive user experience."
                "Observation": "First sentence lacks specificity; second sentence mentions the tablet running Netflix smoothly, which is an experience-rich detail."
                "Thought": "Out of 2 sentences, only 1 qualifies as experience-rich."
                "Final answer": [{"feature": "Performance of Tablets (How easy it is to load and run or operate heavy files)", "actual_sentence_extracted": "It operates perfectly when I watch Netflix, or any of the other apps.", "rephrased_it(without experience & observation)": "The user indicated that it functions flawlessly when they watch Netflix or any of the other applications.", "sentiment": "positive"}]
                '''
            },
            {
                "role": "user",
                "content": f'''Extract the experience-rich sentence based on the given specification:
                Specification: {category_features} of {category_name}
                Review text: {review_text}
                '''
            }
        ]

        try:   
            # Tokenize prompt
            prompt = self.pipeline.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
            )

            # Define terminators for generation
            terminators = [
                self.pipeline.tokenizer.eos_token_id,
                self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            # Generate text
            outputs = self.pipeline(
                prompt,
                max_new_tokens=1000,
                eos_token_id=terminators,           
                do_sample=True,
                temperature=0.01,
                top_p=0.9,
            )
            # print("FINAL RESPONSE" , outputs[0]["generated_text"][len(prompt):])
            final_response = outputs[0]["generated_text"][len(prompt):]
        except:
            final_response = "ERROR"
            cost_current = 0
            print("ERROR OPENAI API")

        print("FINAL RESPONSE: ", outputs)
 
        return [final_response, 0]


    def start_extraction_process(self,reviews, is_helpful, rating, category_features, frequency_list, category, asin_value):
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
                output = self.get_snippet_chunked_review(each_review_text, category_features, category)
            else:
                output = self.get_snippet_full_review(each_review_text, category_features, category)

            tt = time.time() - st
            print("Total Time:" , tt)

            actual_response = output[0]
            cost_for_each_review = output[1]
            snips = []
            rephrased_snippets = ""

            if actual_response == "ERROR":
                rephrased_snippets = ""
            else:
                try:
                    rephrased_snippets = output[0].split('Final answer":')[1]
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

    # Function to extract rephrased string from Final Answer
    def extract_rephrased(self,data):
        if type(data) != dict:
            return ""
        try:
            return data.get('rephrased_it(without experience & observation)',"")
        except Exception as e:
            return data
        
    
    # Function to save snippets in excel
    def save_snippets(self):
        desired_column_order = ["is_helpful", "rating", "Actual Response", "Review Number", "Cost per review",	"Time Taken", "amazon_page", "Review text", "Rephrased Snippets", "aspect", "QA_Content", "Accept"]
        CATEGORY = self.df_pid['category_slug'].unique()

        for category in CATEGORY:
            
            files_and_folders = glob.glob(f'{"/home/ubuntu/rag_llama/akshay/bvr/filretered_12_march_2024"}/{category}/**', recursive=True)   
            files = [f for f in files_and_folders if f.endswith('.xlsx')]
            print(files)
            category_features = list(self.features_df[self.features_df["category_slug"]==category]["features"])
            print(category_features)
            if category_features == "":
                print(f"Category Features missing for catrgory {category} in csv file")
                exit()
            
            category_snippet_folder= f'{os.getenv("CATEGORY_SNIPPET_FOLDER")}/{category}'   
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
                        
                    if len(review_list)>0:
                        final_dataframe = self.start_extraction_process(review_list, is_helpful,rating, aspect,frequency, category,asin_value)
                        # final_dataframe["amazon_page"]=f"https://amazon.com/dp/{asin_value[0]}"
                        final_dataframe["amazon_page"]="https://amazon.com/dp/B007F9XHAY"
                        final_dataframe["aspect"] = aspect
                        for column in desired_column_order:
                            if column not in final_dataframe.columns:
                                final_dataframe[column] = ''
                        final_dataframe['QA_Content'] = final_dataframe['Rephrased Snippets'].apply(self.extract_rephrased)
                        final_dataframe = final_dataframe[desired_column_order]
                        combined_data_frame = pd.concat([combined_data_frame, final_dataframe], ignore_index=True)
                        
                combined_data_frame.to_excel(f"{category_snippet_folder}/{listing_id}_snippets.xlsx")


if __name__ == "__main__":
    extractor = ReviewSnippets()
    extractor.load_dataframes()
    extractor.load_configuration()
    extractor.load_model()
    extractor.save_snippets()
