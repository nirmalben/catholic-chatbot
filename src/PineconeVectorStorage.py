from langchain_pinecone import PineconeVectorStore
from langchain_core.documents.base import Document
from pinecone import Pinecone, ServerlessSpec
from typing import List
from VectorStorage import VectorStorage
import os

class PineconeVectorStorage(VectorStorage):
  def __init__(self, chunks: List[Document], embeddings) -> None:
    self.api_key = os.getenv('PINECONE_API_KEY')
    self.index_name = "catholic-chatbot-index"
    super().__init__(chunks=chunks, embeddings=embeddings)
  
  def _load_storage(self) -> None:
    pc = Pinecone(api_key=self.api_key)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if self.index_name not in existing_indexes:
      pc.create_index(
        name=self.index_name,
        dimension=1024,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        metric="cosine")
      PineconeVectorStore.from_documents(self.chunks, self.embeddings, index_name=self.index_name)
    
    index = pc.Index(self.index_name)      
    self.vector_store = PineconeVectorStore(index=index, embedding=self.embeddings)
  
  # Retriever
  # retriever = vector_store.as_retriever(
  #   search_type="similarity_score_threshold",
  #   search_kwargs={"k": 4, "score_threshold": 0.8},
  # )
  # print(retriever.invoke("Write a code to reverse a string in python"))
    