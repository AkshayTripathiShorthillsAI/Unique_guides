import os
import requests
from bs4 import BeautifulSoup
import json

def scrape_content_from_link(link):
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')
    content_div = soup.find('div', {'id': 'mplus-content'})
    if content_div:
        content_text = content_div.get_text()
        return content_text
    else:
        return None

# Initialize an empty list to store the data
data_list = []

# Iterate over each alphabet from 'b' to 'z'
for char in range(ord('b'), ord('z')+1):
    # Construct the URL
    url = f"https://medlineplus.gov/genetics/condition-{chr(char)}/"
    
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    ul_element = soup.find('ul', class_='withident breaklist')
    
    # Iterate over list disease on page
    for li in ul_element.find_all('li'):
        text = li.get_text(strip=True)
        link = li.find('a')['href']
        content = scrape_content_from_link(link)
        data_list.append({'text': text, 'link': link, 'content': content})

# Create the directory if it doesn't exist
output_directory = "scraped_data"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Save the data to a JSON file
json_path = os.path.join(output_directory, 'scraped_content.json')
with open(json_path, 'w') as f:
    json.dump(data_list, f, indent=4)

print(f"Data has been scraped and saved to '{json_path}'.")



# import os
# import requests
# from bs4 import BeautifulSoup
# import json

# def scrape_content_from_link(link):
#     response = requests.get(link)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     content_div = soup.find('div', {'id': 'mplus-content'})
#     if content_div:
#         content_text = content_div.get_text()
#         return content_text
#     else:
#         return None

# # Send a GET request to the URL
# url = "https://medlineplus.gov/genetics/condition/"
# response = requests.get(url)

# # Parse the HTML content
# soup = BeautifulSoup(response.text, 'html.parser')

# # Find the ul element with class 'withident breaklist'
# ul_element = soup.find('ul', class_='withident breaklist')

# # Initialize an empty list to store the data
# data_list = []

# # Iterate over each li element within the ul
# for li in ul_element.find_all('li'):
#     # Extract the text and link
#     text = li.get_text(strip=True)
#     link = li.find('a')['href']
    
#     # Scrape content from the link
#     content = scrape_content_from_link(link)
    
#     # Append the data to the list
#     data_list.append({'text': text, 'link': link, 'content': content})

# # Create the directory if it doesn't exist
# output_directory = "scraped_data"
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)

# # Save the data to a JSON file
# json_path = os.path.join(output_directory, 'scraped_content.json')
# with open(json_path, 'w') as f:
#     json.dump(data_list, f, indent=4)

# print(f"Data has been scraped and saved to '{json_path}'.")


# import os
# import requests
# from bs4 import BeautifulSoup
# import json

# def scrape_data_from_link(link):
#     response = requests.get(link)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     content_div = soup.find('div', {'id': 'mplus-content'})
#     if content_div:
#         content_text = content_div.get_text()
#         return content_text
#     else:
#         return None

# # Send a GET request to the URL
# url = "https://medlineplus.gov/genetics/condition/"
# response = requests.get(url)

# # Parse the HTML content
# soup = BeautifulSoup(response.text, 'html.parser')

# # Find the ul element with class 'withident breaklist'
# ul_element = soup.find('ul', class_='withident breaklist')

# # Initialize an empty list to store the data
# data_list = []

# # Iterate over each li element within the ul
# for li in ul_element.find_all('li'):
#     # Extract the text and link
#     text = li.get_text(strip=True)
#     link = li.find('a')['href']
    
#     # Scrape data from the link
#     scraped_data = scrape_data_from_link(link)
    
#     # Append the data to the list
#     data_list.append({'text': text, 'link': link, 'data': scraped_data})

# # Create the directory if it doesn't exist
# output_directory = "scraped_data"
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)

# # Save the data to a JSON file
# json_path = os.path.join(output_directory, 'scraped_data.json')
# with open(json_path, 'w') as f:
#     json.dump(data_list, f, indent=4)

# print(f"Data has been scraped and saved to '{json_path}'.")



# import requests
# from bs4 import BeautifulSoup
# import json

# # Send a GET request to the URL
# url = "https://medlineplus.gov/genetics/condition/"
# response = requests.get(url)

# # Parse the HTML content
# soup = BeautifulSoup(response.text, 'html.parser')

# # Find the ul element with class 'withident breaklist'
# ul_element = soup.find('ul', class_='withident breaklist')

# # Initialize an empty list to store the data
# data_list = []

# # Iterate over each li element within the ul
# for li in ul_element.find_all('li'):
#     # Extract the text and link
#     text = li.get_text(strip=True)
#     link = li.find('a')['href']
    
#     # Append the data to the list
#     data_list.append({'text': text, 'link': link})

# # Save the data to a JSON file
# with open('scraped_data.json', 'w') as f:
#     json.dump(data_list, f, indent=4)

# print("Data has been scraped and saved to scraped_data.json")
