import time
import os
import logging
import numpy as np
from opensearchpy import OpenSearch, RequestsHttpConnection
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from embeddig_model import E5Embeddings
from config import CONFIG
from dotenv import load_dotenv
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenSearchDB():
    def __init__(self,embedding):
        self.embedding = embedding
        self.client = self.get_opensearch_client()
    def get_opensearch_client(self):
        load_dotenv()
        admin_key = os.environ.get("OPENSEARCH_KEY")
        try:
            client = OpenSearch(
                hosts=[{"host": "localhost", "port": 9200}],
                http_compress=True,
                http_auth=("admin", admin_key),
                use_ssl=True,
                verify_certs=False,
                ssl_assert_hostname=False,
                ssl_show_warn=False,
                connection_class=RequestsHttpConnection
            )
            logger.info("connected to OpenSearch")
            return client
        except Exception as e:
            logger.error(f" Failed to connect to OpenSearch: {e}")
            exit(1)

    def query_documents(self,question,index_name, k=3):
        question_embedding = self.embedding.embed_query(question)
        search_query = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": question_embedding[0],
                        "k": k
                    }
                }
            }
        }

        try:
            response = self.client.search(index=index_name, body=search_query)
            return response["hits"]["hits"]
        except Exception as e:
            logger.error(f" Query failed: {e}")
            return []
    def create_body(self,index_name):
        embedding_dim = self.embedding.dim
        logger.info(f"✅ Embedding Dimension: {embedding_dim}")
        index_body = {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": CONFIG['retriever']['opensearch']['number_of_shards'],
                "number_of_replicas": CONFIG['retriever']['opensearch']['number_of_replicas']
            }
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": embedding_dim,
                    "method": {
                        "name": CONFIG['retriever']['opensearch']['knn_method'],
                        "engine": CONFIG['retriever']['opensearch']['knn_engine'],
                        "space_type": CONFIG['retriever']['opensearch']['space_type']
                    }
                },
                    "metadata": {
                    "properties": {
                    "document_name": {"type": "text"},
                    "page": {"type": "integer"},
                    "total_page": {"type": "integer"},
                    "author": {"type": "text"}
                    }
                }
                }
            }
        }
        if self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)
        self.client.indices.create(index=index_name, body=index_body)
        logger.info(f"✅ Created index: {index_name} with k-NN enabled and dimension {embedding_dim}")
