from abc import ABC, abstractmethod

from langchain_core.documents.base import Document
from typing import List

class DocumentLoader(ABC):
  def __init__(self, file_path: str) -> None:
    self.file_path = file_path
    self.documents = []
  
  def _load(self) -> None:
    # Populates self.documents
    pass

  def get_documents(self) -> List[Document]:
    return self.documents