import os
import requests
from bs4 import BeautifulSoup
from fpdf import FPDF

# Function to scrape content and save as PDF
def scrape_and_save_as_pdf(url, output_filename):
    # Send a GET request to the URL
    response = requests.get(url)

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the main content div with the specified class
    main_content_div = soup.find('div', class_='jig-ncbiinpagenav body-content whole_rhythm')

    # Check if the main content div exists
    if main_content_div:
        # Initialize PDF content string
        pdf_content = ""

        # Find all divs within the main content div
        sub_divs = main_content_div.find_all('div')

        # Extract text content from each sub div and add it to the PDF content
        for div in sub_divs:
            # Extract text content from the div and append to the PDF content
            content_text = div.get_text().strip()
            pdf_content += content_text + "\n\n"

        # Save the PDF content to a file with UTF-8 encoding
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(pdf_content)

        print(f"PDF saved successfully in '{output_filename}'.")
    else:
        print("Main content div not found.")

# URL of the website to scrape
url = "https://www.ncbi.nlm.nih.gov/books/NBK396098/"

# Output PDF filename
output_filename = "scraped_content.pdf"

# Scrape and save as PDF
scrape_and_save_as_pdf(url, output_filename)
