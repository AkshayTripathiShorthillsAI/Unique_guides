import requests
from bs4 import BeautifulSoup
import json

# URL of the webpage
url = "https://pubmed.ncbi.nlm.nih.gov/27854409/"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the div with class "abstract-content selected"
    abstract_div = soup.find('div', class_='abstract-content selected')

    if abstract_div:
        # Extract the text from the div
        abstract_text = abstract_div.get_text(separator='\n').strip()
        
        # Create a dictionary to hold the data
        data = {
            "url": url,
            "abstract_text": abstract_text
        }
        
        # Convert the dictionary to JSON format
        json_data = json.dumps(data, indent=4)
        
        # Print the JSON data
        print(json_data)
    else:
        print("Abstract content not found on the page.")
else:
    print("Failed to retrieve the webpage.")
