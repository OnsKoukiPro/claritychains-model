import os
from langchain.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import Document

class Retriever:
    def __init__(self):
        self.enabled = os.environ.get("USE_RETRIEVER", "false").lower() == "true"
        self.vector_store = None
        self.embeddings = None
        if self.enabled:
            try:
                self.embeddings = AzureOpenAIEmbeddings(
                    azure_deployment="text-embedding-ada-002",
                    api_version="2024-08-01-preview",
                )
            except Exception as e:
                print(f"Error creating AzureOpenAIEmbeddings: {e}")
                self.enabled = False

    def create_retriever(self, documents):
        if not self.enabled:
            return
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def add_to_retriever(self, documents):
        if not self.enabled:
            return
        if self.vector_store:
            self.vector_store.add_documents(documents)
        else:
            self.create_retriever(documents)

    def get_retriever(self):
        if not self.enabled or not self.vector_store:
            return None
        return self.vector_store.as_retriever()