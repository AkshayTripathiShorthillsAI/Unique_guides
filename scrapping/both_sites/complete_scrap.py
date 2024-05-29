import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import time 


start_time = time.time()
def scrape_medlineplus(url):
    if not url:
        return None
    if not url.startswith("http"):
        url = "https://" + url
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    content_div = soup.find('div', {'id': 'mplus-content'})
    if content_div:
        content_text = content_div.get_text()
        return content_text
    else:
        return None

def scrape_gene_review(url):
    if not url:
        return None
    if not url.startswith("http"):
        url = "https://" + url
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    main_content_div = soup.find('div', class_='jig-ncbiinpagenav body-content whole_rhythm')
    if main_content_div:
        pdf_content = ""
        sub_divs = main_content_div.find_all('div')
        for div in sub_divs:
            content_text = div.get_text().strip()
            pdf_content += content_text + "\n\n"
        return pdf_content
    else:
        return None

def scrape_pubmed_abstract(url):
    if not url:
        return None
    if not url.startswith("http"):
        url = "https://" + url
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    abstract_div = soup.find('div', class_='abstract-content selected')
    if abstract_div:
        abstract_text = abstract_div.get_text(separator='\n').strip()
        return abstract_text
    else:
        return None

# excel_file = "temp_links.xlsx"
excel_file = "/home/shtlp_0015/Desktop/Unique/Single_gene_disorder_data.xlsx"
df = pd.read_excel(excel_file)

# Create a dictionary to store scraped data
scraped_data = {}

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    syndrome_name = row[0]  # Disease name from column 1
    medlineplus_links = str(row[1]).split(',') if not pd.isna(row[1]) else []  # Medline Plus links from column 2
    gene_review_links = str(row[2]).split(',') if not pd.isna(row[2]) else []  # Gene review links from column 3
    pubmed_links = str(row[3]).split(',') if not pd.isna(row[3]) else []  # PubMed links from column 4
    
    syndrome_data = {}
    
    # Scraping content from MedlinePlus links
    medlineplus_content = {}
    for link in medlineplus_links:
        content = scrape_medlineplus(link.strip())
        if content:
            medlineplus_content[link.strip()] = content
    
    # Scraping content from gene review links
    gene_review_content = {}
    for link in gene_review_links:
        content = scrape_gene_review(link.strip())
        if content:
            gene_review_content[link.strip()] = content
    
    # Scraping PubMed abstracts
    pubmed_abstracts = {}
    for link in pubmed_links:
        abstract = scrape_pubmed_abstract(link.strip())
        if abstract:
            pubmed_abstracts[link.strip()] = abstract
    
    # Combine all the data into a single dictionary for the syndrome
    syndrome_data[syndrome_name] = {}
    syndrome_data[syndrome_name].update(medlineplus_content)
    syndrome_data[syndrome_name].update(gene_review_content)
    syndrome_data[syndrome_name].update(pubmed_abstracts)
    scraped_data.update(syndrome_data)

output_json_file = "scraped_data.json"
with open(output_json_file, 'w') as json_file:
    json.dump(scraped_data, json_file, indent=4)
print(f"Scraped data saved successfully in '{output_json_file}'.")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")


# import requests
# from bs4 import BeautifulSoup
# import json
# import pandas as pd

# # Function to scrape content from MedlinePlus and save as JSON
# def scrape_medlineplus(url):
#     if not url:
#         return None
#     if not url.startswith("http"):
#         url = "https://" + url
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     content_div = soup.find('div', {'id': 'mplus-content'})
#     if content_div:
#         content_text = content_div.get_text()
#         return content_text
#     else:
#         return None

# # Function to scrape content from gene review site and save as PDF
# def scrape_gene_review(url):
#     if not url:
#         return None
#     if not url.startswith("http"):
#         url = "https://" + url
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     main_content_div = soup.find('div', class_='jig-ncbiinpagenav body-content whole_rhythm')
#     if main_content_div:
#         pdf_content = ""
#         sub_divs = main_content_div.find_all('div')
#         for div in sub_divs:
#             content_text = div.get_text().strip()
#             pdf_content += content_text + "\n\n"
#         return pdf_content
#     else:
#         return None

# # Function to scrape PubMed abstract from URL
# def scrape_pubmed_abstract(url):
#     if not url:
#         return None
#     if not url.startswith("http"):
#         url = "https://" + url
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, 'html.parser')
#     abstract_div = soup.find('div', class_='abstract-content selected')
#     if abstract_div:
#         abstract_text = abstract_div.get_text(separator='\n').strip()
#         return abstract_text
#     else:
#         return None

# # Load Excel sheet into DataFrame
# excel_file = "temp_links.xlsx"
# df = pd.read_excel(excel_file)

# # Create a list to store scraped data
# scraped_data = []

# # Iterate through each row in the DataFrame
# for index, row in df.iterrows():
#     syndrome_name = row[0]  # Disease name from column 1
#     medlineplus_links = str(row[1]).split(',') if not pd.isna(row[1]) else []  # Medline Plus links from column 2
#     gene_review_links = str(row[2]).split(',') if not pd.isna(row[2]) else []  # Gene review links from column 3
#     pubmed_links = str(row[3]).split(',') if not pd.isna(row[3]) else []  # PubMed links from column 4
    
#     syndrome_data = {
#         "syndrome_name": syndrome_name,
#         "MedlinePlus": [],
#         "GeneReview": [],
#         "PubMed": []
#     }
    
#     # Scraping content from MedlinePlus links
#     for link in medlineplus_links:
#         content = scrape_medlineplus(link.strip())
#         if content:
#             syndrome_data["MedlinePlus"].append({
#                 "link": link.strip(),
#                 "content": content
#             })
    
#     # Scraping content from gene review links
#     for link in gene_review_links:
#         content = scrape_gene_review(link.strip())
#         if content:
#             syndrome_data["GeneReview"].append({
#                 "link": link.strip(),
#                 "content": content
#             })
    
#     # Scraping PubMed abstracts
#     for link in pubmed_links:
#         abstract = scrape_pubmed_abstract(link.strip())
#         if abstract:
#             syndrome_data["PubMed"].append({
#                 "link": link.strip(),
#                 "abstract": abstract
#             })
    
#     scraped_data.append(syndrome_data)

# # Save the scraped data as JSON
# output_json_file = "scraped_data.json"
# with open(output_json_file, 'w') as json_file:
#     json.dump(scraped_data, json_file, indent=4)

# print(f"Scraped data saved successfully in '{output_json_file}'.")



# import os
# import requests
# from bs4 import BeautifulSoup
# from fpdf import FPDF
# import json
# import pandas as pd
# import numpy as np

# # Function to scrape content from MedlinePlus and save as JSON
# def scrape_medlineplus(url):
#     if not url:
#         return None
#     if not url.startswith("http"):
#         url = "https://" + url
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     content_div = soup.find('div', {'id': 'mplus-content'})
#     if content_div:
#         content_text = content_div.get_text()
#         return content_text
#     else:
#         return None

# # Function to scrape content from gene review site and save as PDF
# def scrape_gene_review(url):
#     if not url:
#         return None
#     if not url.startswith("http"):
#         url = "https://" + url
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     main_content_div = soup.find('div', class_='jig-ncbiinpagenav body-content whole_rhythm')
#     if main_content_div:
#         pdf_content = ""
#         sub_divs = main_content_div.find_all('div')
#         for div in sub_divs:
#             content_text = div.get_text().strip()
#             pdf_content += content_text + "\n\n"
#         return pdf_content
#     else:
#         return None


# # Function to scrape PubMed abstract from URL

# # Function to scrape PubMed abstract from URL
# def scrape_pubmed_abstract(url):
#     # print("Processing PubMed URL:", url)  # Debugging line
#     if not url:
#         return None
#     if not url.startswith("http"):
#         url = "https://" + url
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, 'html.parser')
#     abstract_div = soup.find('div', class_='abstract-content selected')
#     if abstract_div:
#         abstract_text = abstract_div.get_text(separator='\n').strip()
#         return abstract_text
#     else:
#         return None


# def scrape_pubmed_abstract(url):
#     if not url:
#         return None
#     if not url.startswith("http"):
#         url = "https://" + url
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, 'html.parser')
#     abstract_div = soup.find('div', class_='abstract-content selected')
#     if abstract_div:
#         abstract_text = abstract_div.get_text(separator='\n').strip()
#         return abstract_text
#     else:
#         return None
    

# # Load Excel sheet into DataFrame
# excel_file = "temp_links.xlsx" 
# # excel_file = "/home/shtlp_0015/Desktop/Unique/Single_gene_disorder_data.xlsx"
# df = pd.read_excel(excel_file)

# # Create a dictionary to store scraped data
# scraped_data = {}

# # Iterate through each row in the DataFrame
# for index, row in df.iterrows():
#     syndrome_name = row[0]  # Disease name from column 1
#     medlineplus_links = str(row[1]).split(',') if not pd.isna(row[1]) else []  # Medline Plus links from column 2
#     gene_review_links = str(row[2]).split(',') if not pd.isna(row[2]) else []  # Gene review links from column 3
#     pubmed_links = str(row[3]).split(',') if not pd.isna(row[3]) else []  # PubMed links from column 4
    
#     syndrome_data = {}
    
#     # Scraping content from MedlinePlus links
#     medlineplus_content = {}
#     for link in medlineplus_links:
#         content = scrape_medlineplus(link.strip())
#         if content:
#             medlineplus_content[link.strip()] = content
    
#     # Scraping content from gene review links
#     gene_review_content = {}
#     for link in gene_review_links:
#         content = scrape_gene_review(link.strip())
#         if content:
#             gene_review_content[link.strip()] = content
    
#     # Scraping PubMed abstracts
#     pubmed_abstracts = {}
#     for link in pubmed_links:
#         abstract = scrape_pubmed_abstract(link.strip())
#         if abstract:
#             pubmed_abstracts[link.strip()] = abstract
    
#     syndrome_data['MedlinePlus'] = medlineplus_content
#     syndrome_data['GeneReview'] = gene_review_content
#     syndrome_data['PubMed'] = pubmed_abstracts
    
#     scraped_data[syndrome_name] = syndrome_data

# # Save the scraped data as JSON
# output_json_file = "scraped_data.json"
# with open(output_json_file, 'w') as json_file:
#     json.dump(scraped_data, json_file, indent=4)

# print(f"Scraped data saved successfully in '{output_json_file}'.")



# import os
# import requests
# from bs4 import BeautifulSoup
# from fpdf import FPDF
# import json
# import pandas as pd

# # Function to scrape content from MedlinePlus and save as JSON
# def scrape_medlineplus(url):
#     if not url:
#         return None
#     if not url.startswith("http"):
#         url = "https://" + url
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     content_div = soup.find('div', {'id': 'mplus-content'})
#     if content_div:
#         content_text = content_div.get_text()
#         return content_text
#     else:
#         return None

# # Function to scrape content from gene review site and save as PDF
# def scrape_gene_review(url):
#     if not url:
#         return None
#     if not url.startswith("http"):
#         url = "https://" + url
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     main_content_div = soup.find('div', class_='jig-ncbiinpagenav body-content whole_rhythm')
#     if main_content_div:
#         pdf_content = ""
#         sub_divs = main_content_div.find_all('div')
#         for div in sub_divs:
#             content_text = div.get_text().strip()
#             pdf_content += content_text + "\n\n"
#         return pdf_content
#     else:
#         return None


# # Function to scrape PubMed abstract from URL
# def scrape_pubmed_abstract(url):
#     if not url:
#         return None
#     if not url.startswith("http"):
#         url = "https://" + url
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, 'html.parser')
#     abstract_div = soup.find('div', class_='abstract-content selected')
#     if abstract_div:
#         abstract_text = abstract_div.get_text(separator='\n').strip()
#         return abstract_text
#     else:
#         return None
    

# # Load Excel sheet into DataFrame
# # excel_file = "temp_links.xlsx" 
# excel_file = "/home/shtlp_0015/Desktop/Unique/Single_gene_disorder_data.xlsx"
# df = pd.read_excel(excel_file)

# # Create a dictionary to store scraped data
# scraped_data = {}

# # Iterate through each row in the DataFrame
# for index, row in df.iterrows():
#     syndrome_name = row[0]  # Disease name from column 1
#     medlineplus_links = row[1].split(',')  # Medline Plus links from column 2
#     gene_review_links = row[2].split(',')  # Gene review links from column 3
#     pubmed_links = row[3].split(',')  # PubMed links from column 4
    
#     syndrome_data = {}
    
#     # Scraping content from MedlinePlus links
#     medlineplus_content = {}
#     for link in medlineplus_links:
#         content = scrape_medlineplus(link.strip())
#         if content:
#             medlineplus_content[link.strip()] = content
    
#     # Scraping content from gene review links
#     gene_review_content = {}
#     for link in gene_review_links:
#         content = scrape_gene_review(link.strip())
#         if content:
#             gene_review_content[link.strip()] = content
    
#     # Scraping PubMed abstracts
#     pubmed_abstracts = {}
#     for link in pubmed_links:
#         abstract = scrape_pubmed_abstract(link.strip())
#         if abstract:
#             pubmed_abstracts[link.strip()] = abstract
    
#     syndrome_data['MedlinePlus'] = medlineplus_content
#     syndrome_data['GeneReview'] = gene_review_content
#     syndrome_data['PubMed'] = pubmed_abstracts
    
#     scraped_data[syndrome_name] = syndrome_data

# # Save the scraped data as JSON
# output_json_file = "scraped_data.json"
# with open(output_json_file, 'w') as json_file:
#     json.dump(scraped_data, json_file, indent=4)

# print(f"Scraped data saved successfully in '{output_json_file}'.")
