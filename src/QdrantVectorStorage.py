from langchain_qdrant import QdrantVectorStore
from langchain_core.documents.base import Document
from typing import List
from VectorStorage import VectorStorage
import os

class QdrantVectorStorage(VectorStorage):
  def __init__(self, api_key: str, url: str, chunks: List[Document], embeddings) -> None:
    self.api_key = os.getenv('QDRANT_API_KEY')
    self.url = os.getenv('QDRANT_URL')
    self.collection_name = "catholic-docs"
    super().__init__(chunks=chunks, embeddings=embeddings)
  
  def _load_storage(self) -> None:
    print("Loading documents into vectorstore...")
    self.vector_store = QdrantVectorStore.from_documents(
        documents=self.chunks, embedding=self.embeddings, url=self.url, api_key=self.api_key, collection_name=self.collection_name
    )
    
  def get_vector_store(self):
    if self.vector_store == None:
      self.vector_store = QdrantVectorStore.from_existing_collection(
        collection_name=self.collection_name,
        embeddings=self.embeddings,
        url=self.url,
        api_key=self.api_key)

    return self.vector_store
