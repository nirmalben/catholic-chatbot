from abc import ABC, abstractmethod

from langchain_core.documents.base import Document
from typing import List

class VectorStorage(ABC):
  def __init__(self, chunks: List[Document], embeddings) -> None:
    self.chunks = chunks
    self.embeddings = embeddings
    self.vector_store = None
    self._load_storage()
  
  @abstractmethod
  def _load_storage(self) -> None:
    # Populates self.vector_store
    pass

  def get_vector_store(self):
    return self.vector_store