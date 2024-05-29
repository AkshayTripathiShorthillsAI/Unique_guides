import weaviate
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader



# Connect to Weaviate
WEAVIATE_URL = "http://localhost:8080"

client = weaviate.Client(url=WEAVIATE_URL)


schema = {
    "class": "Mpph_1",
    "vectorizer": "text2vec-transformers",
    "moduleConfig": {
    "text2vec-huggingface": {
      "model": "sentence-transformers/all-MiniLM-L12-v2-onnx", 
      "options": {
        "waitForModel": True,
      }
     }
    }
}


# client.schema.create_class(schema)
data_objects = []

PDF_STORAGE_PATH = "/home/ubuntu/rag_llama/akshay/Unique_llama2/pdf"
def process_pdfs(pdf_storage_path: str):
    pdf_directory = Path(pdf_storage_path)
    docs = []  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    for pdf_path in pdf_directory.glob("*.pdf"):
        loader = PyMuPDFLoader(str(pdf_path))
        # print(loader)
        documents = loader.load()
        docs += text_splitter.split_documents(documents)
    return docs

data_objects = process_pdfs(PDF_STORAGE_PATH)

print(len(data_objects))
# for i in data_objects:
#     print(i)
#     print("&&&&&&&&&&&&&&")
#     print(i.page_content)
#     print(i.metadata['source'])
#     print(i.metadata['page'])
#     print("***********************")


# adding chunks to weavaite while configuring the schema itself here
client.batch.configure(batch_size=100)  
with client.batch as batch:
    for data_object in data_objects:
        chunk_object = {
            "source": data_object.metadata['source'],
            "text": data_object.page_content,
            "page": data_object.metadata['page']
        }
        batch.add_data_object(data_object=chunk_object, class_name="Mpph_1")