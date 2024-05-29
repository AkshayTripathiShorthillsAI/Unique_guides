import os
import json
from fpdf import FPDF

with open("scraped_data.json", "r", encoding='utf-8') as json_file:
    data = json.load(json_file)

if not os.path.exists("PDF_files"):
    os.makedirs("PDF_files")

def create_pdf(syndrome_name, url, content, count):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, syndrome_name, ln=True, align="C")

    pdf.set_font("Arial", style='I', size=12)
    pdf.cell(200, 10, url, ln=True, align="C")
    pdf.ln(10)

    content_escaped = content.encode('ascii', 'xmlcharrefreplace').decode('ascii')

    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, content_escaped, align="L")

    pdf_file_path = os.path.join("PDF_files", f"{syndrome_name}_{os.path.basename(url)}_{count}.pdf")
    try:
        pdf.output(pdf_file_path)
        print(f"PDF generated for {syndrome_name}: {url}.")
    except Exception as e:
        print(f"Error generating PDF for {syndrome_name}: {url}. {e}")

for syndrome_name, syndrome_data in data.items():
    for index, (url, content) in enumerate(syndrome_data.items(), start=1):
        create_pdf(syndrome_name, url, content, index)


# import os
# import json
# from fpdf import FPDF

# # Load JSON data
# with open("scraped_data.json", "r", encoding='utf-8') as json_file:
#     data = json.load(json_file)

# # Create a directory to store PDF files
# if not os.path.exists("PDF_files"):
#     os.makedirs("PDF_files")

# # Function to create PDF files
# def create_pdf(syndrome_name, url, content):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_auto_page_break(auto=True, margin=15)
#     pdf.set_font("Arial", size=12)

#     # Add syndrome name as title
#     pdf.set_font("Arial", style='B', size=16)
#     pdf.cell(200, 10, syndrome_name, ln=True, align="C")

#     # Add URL as subtitle
#     pdf.set_font("Arial", style='I', size=12)
#     pdf.cell(200, 10, url, ln=True, align="C")
#     pdf.ln(10)

#     # Replace characters outside the ASCII range with Unicode escape sequences
#     content_escaped = content.encode('ascii', 'xmlcharrefreplace').decode('ascii')

#     # Add content to PDF
#     pdf.set_font("Arial", size=12)
#     pdf.multi_cell(0, 10, content_escaped, align="L")

#     # Save the PDF file
#     pdf_file_path = os.path.join("PDF_files", f"{syndrome_name}_{os.path.basename(url)}.pdf")
#     pdf.output(pdf_file_path)
#     print(f"PDF generated for {syndrome_name}: {url}.")

# # Iterate over syndromes and their data
# count =0
# for syndrome_name, syndrome_data in data.items():
#     for url, content in syndrome_data.items():
#         count = count +1
#         create_pdf(syndrome_name, url, content)

# print(count)



# import os
# import json
# from fpdf import FPDF

# # Load JSON data
# with open("scraped_data.json", "r", encoding='utf-8') as json_file:
#     data = json.load(json_file)

# # Create a directory to store PDF files
# if not os.path.exists("PDF_files"):
#     os.makedirs("PDF_files")

# # Function to create PDF files
# def create_pdf(syndrome_name, url, content):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_auto_page_break(auto=True, margin=15)
#     pdf.set_font("Arial", size=12)

#     # Add syndrome name as title
#     pdf.set_font("Arial", style='B', size=16)
#     pdf.cell(200, 10, syndrome_name, ln=True, align="C")

#     # Add URL as subtitle
#     pdf.set_font("Arial", style='I', size=12)
#     pdf.cell(200, 10, url, ln=True, align="C")
#     pdf.ln(10)

#     # Replace characters outside the ASCII range with Unicode escape sequences
#     content_escaped = content.encode('ascii', 'xmlcharrefreplace').decode('ascii')

#     # Add content to PDF
#     pdf.set_font("Arial", size=12)
#     pdf.multi_cell(0, 10, content_escaped, align="L")

#     # Save the PDF file
#     pdf_file_path = os.path.join("PDF_files", f"{syndrome_name}_{os.path.basename(url)}.pdf")
#     pdf.output(pdf_file_path)
#     print(f"PDF generated for {syndrome_name}: {url}.")

# # Iterate over syndromes and their data
# for syndrome_name, syndrome_data in data.items():
#     for url, content in syndrome_data.items():
#         create_pdf(syndrome_name, url, content)
 