import os
import requests
from bs4 import BeautifulSoup
import json

# URL of the website to scrape
url = "https://medlineplus.gov/genetics/condition/megalencephaly-polymicrogyria-polydactyly-hydrocephalus-syndrome/"

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Find the div with id "mplus-content" and extract its content
content_div = soup.find('div', {'id': 'mplus-content'})

# Extract text content from the div
if content_div:
    content_text = content_div.get_text()

    # Create the directory if it doesn't exist
    output_directory = "scraped_data"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save the content as JSON
    json_path = os.path.join(output_directory, 'scraped_content.json')

    # Create a dictionary to store the data
    data = {
        'source': url,
        'content': content_text
    }

    # Save the data as JSON
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    print(f"JSON saved successfully in '{json_path}'.")
else:
    print("Content div not found.")
