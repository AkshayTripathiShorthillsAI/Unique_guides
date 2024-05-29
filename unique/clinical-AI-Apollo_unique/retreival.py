# from langchain_community.vectorstores import Weaviate
from weaviate.client import Client
import json


# Connect to Weaviate
WEAVIATE_URL = "http://localhost:8080"
client = Client(WEAVIATE_URL)


def query_hp_book(keyword):
    response = (
        client.query
        .get("Genetics_512", ["source", "text"])
        .with_bm25(query=keyword)
        # .with_near_text({"concepts":[keyword]})
        .with_limit(10)
        .do()
    )
    return response['data']['Get']['Genetics_512']