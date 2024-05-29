# from langchain_community.vectorstores import Weaviate
from weaviate.client import Client
import json


# Connect to Weaviate
WEAVIATE_URL = "http://localhost:8080"
client = Client(WEAVIATE_URL)


def query_pdf(keyword):
    response = (
        client.query
        # .get("Genetics_512", ["source", "text"])
        .get("Mpph_1", ["source", "text"])
        .with_bm25(query=keyword)
        # .with_near_text({"concepts":[keyword]})
        .with_limit(10)
        .do()
    )
    return response['data']['Get']['Mpph_1']