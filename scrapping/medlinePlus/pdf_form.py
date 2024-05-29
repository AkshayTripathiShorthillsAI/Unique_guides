import os
import requests
from bs4 import BeautifulSoup
from fpdf import FPDF

# URL of the website to scrape
url = "https://medlineplus.gov/genetics/condition/satb2-associated-syndrome/"

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

