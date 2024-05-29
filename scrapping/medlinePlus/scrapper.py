# import os
# import requests
# from bs4 import BeautifulSoup
# from fpdf import FPDF

# def scrape_website(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     content_div = soup.find('div', {'id': 'mplus-content'})
#     if content_div:
#         return content_div.get_text()
#     else:
#         return None

# def save_as_pdf(content_text, output_directory, pdf_filename):
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)

#     pdf_path = os.path.join(output_directory, pdf_filename)

#     pdf = FPDF()

#     pdf.add_page()

#     pdf.set_font('Arial', size=12)

#     pdf.multi_cell(0, 10, content_text.encode('latin-1', 'replace').decode('latin-1'))

#     pdf.output(pdf_path)
#     return pdf_path

# def main():
#     url = "https://medlineplus.gov/genetics/condition/megalencephaly-polymicrogyria-polydactyly-hydrocephalus-syndrome/"

#     content_text = scrape_website(url)

#     if content_text:
#         output_directory = "scraped_data"
#         pdf_filename = 'scraped_content.pdf'
#         pdf_path = save_as_pdf(content_text, output_directory, pdf_filename)
#         print(f"PDF saved successfully in '{pdf_path}'.")
#     else:
#         print("Content div not found.")

# if __name__ == "__main__":
#     main()



import os
import requests
from bs4 import BeautifulSoup
from fpdf import FPDF

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

    # Save the PDF in the directory
    pdf_path = os.path.join(output_directory, 'scraped_content.pdf')

    # Create a new FPDF object
    pdf = FPDF()

    # Add a new page to the PDF
    pdf.add_page()

    # Set the font and font size
    pdf.set_font('Arial', size=12)

    # Write the text to the PDF (encode to UTF-8)
    pdf.multi_cell(0, 10, content_text.encode('latin-1', 'replace').decode('latin-1'))

    # Save the PDF
    pdf.output(pdf_path)
    print(f"PDF saved successfully in '{pdf_path}'.")
else:
    print("Content div not found.")


# import os
# import requests
# from bs4 import BeautifulSoup
# from fpdf import FPDF

# # Function to scrape individual link and save as PDF
# def scrape_link(url, output_directory):
#     # Send a GET request to the URL
#     response = requests.get(url)

#     # Parse the HTML content
#     soup = BeautifulSoup(response.text, 'html.parser')

#     # Find the div with id "mplus-content" and extract its content
#     content_div = soup.find('div', {'id': 'mplus-content'})

#     # Extract text content from the div
#     if content_div:
#         content_text = content_div.get_text()

#         # Save the PDF in the directory
#         pdf_path = os.path.join(output_directory, f'{url.split("/")[-2]}.pdf')

#         # Create a new FPDF object
#         pdf = FPDF()

#         # Add a new page to the PDF
#         pdf.add_page()

#         # Set the font and font size
#         pdf.set_font('Arial', size=12)

#         # Write the text to the PDF (encode to UTF-8)
#         pdf.multi_cell(0, 10, content_text.encode('latin-1', 'replace').decode('latin-1'))

#         # Save the PDF
#         pdf.output(pdf_path)
#         print(f"PDF saved successfully for {url} in '{pdf_path}'.")
#     else:
#         print(f"Content div not found for {url}.")


# # URL of the website to scrape
# url = "https://medlineplus.gov/genetics/condition/megalencephaly-polymicrogyria-polydactyly-hydrocephalus-syndrome/"

# # Send a GET request to the URL
# response = requests.get(url)

# # Parse the HTML content
# soup = BeautifulSoup(response.text, 'html.parser')

# # Find all hyperlinks in the content
# links = soup.find_all('a')

# # Create the directory if it doesn't exist
# output_directory = "scraped_data"
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)

# # Iterate through each link and scrape
# for link in links:
#     href = link.get('href')
#     if href and href.startswith("https://medlineplus.gov/genetics/condition/"):
#         scrape_link(href, output_directory)


# import os
# import requests
# from bs4 import BeautifulSoup
# from fpdf import FPDF
# from PyPDF2 import PdfReader

# # Function to scrape individual link and save as PDF
# def scrape_link(url, output_directory):
#     # Send a GET request to the URL
#     response = requests.get(url)

#     # Parse the HTML content
#     soup = BeautifulSoup(response.text, 'html.parser')

#     # Find the div with id "mplus-content" and extract its content
#     content_div = soup.find('div', {'id': 'mplus-content'})

#     # Extract text content from the div
#     if content_div:
#         content_text = content_div.get_text()

#         # Save the PDF in the directory
#         pdf_path = os.path.join(output_directory, f'{url.split("/")[-2]}.pdf')

#         # Create a new FPDF object
#         pdf = FPDF()

#         # Add a new page to the PDF
#         pdf.add_page()

#         # Set the font and font size
#         pdf.set_font('Arial', size=12)

#         # Write the text to the PDF (encode to UTF-8)
#         pdf.multi_cell(0, 10, content_text.encode('latin-1', 'replace').decode('latin-1'))

#         # Save the PDF
#         pdf.output(pdf_path)
#         print(f"PDF saved successfully for {url} in '{pdf_path}'.")
#     else:
#         print(f"Content div not found for {url}.")

# # Function to extract links from text
# def extract_links_from_text(text):
#     soup = BeautifulSoup(text, 'html.parser')
#     links = [a['href'] for a in soup.find_all('a', href=True)]
#     return links

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     with open(pdf_path, 'rb') as f:
#         reader = PdfReader(f)
#         text = ''
#         for page_num in range(len(reader.pages)):
#             text += reader.getPage(page_num).extractText()
#     return text

# # Input PDF file path
# input_pdf_path = "input/MPPH Syndrome 1.pdf"

# # Extract text from input PDF
# input_text = extract_text_from_pdf(input_pdf_path)

# # Extract links from the text
# links = extract_links_from_text(input_text)

# # Create the directory if it doesn't exist
# output_directory = "scraped_data"
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)

# # Scrape each link separately
# for link in links:
#     scrape_link(link, output_directory)


