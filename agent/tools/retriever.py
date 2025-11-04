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
                print("✅ Azure embeddings configured successfully")
            except Exception as e:
                print(f"❌ Error creating AzureOpenAIEmbeddings: {e}")
                print("⚠️  Disabling retriever functionality")
                self.enabled = False

    def create_retriever(self, documents):
        if not self.enabled:
            print("⚠️  Retriever disabled, skipping create_retriever")
            return
        try:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            print("✅ Retriever created successfully")
        except Exception as e:
            print(f"❌ Error creating retriever: {e}")
            self.enabled = False

    def add_to_retriever(self, documents):
        if not self.enabled:
            return
        try:
            if self.vector_store:
                self.vector_store.add_documents(documents)
            else:
                self.create_retriever(documents)
        except Exception as e:
            print(f"❌ Error adding to retriever: {e}")
            self.enabled = False

    def get_retriever(self):
        if not self.enabled or not self.vector_store:
            return None
        return self.vector_store.as_retriever()