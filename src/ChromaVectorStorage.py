__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from chromadb.utils.batch_utils import create_batches
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from pathlib import Path
from typing import List
from VectorStorage import VectorStorage
import chromadb
import uuid

class ChromaVectorStorage(VectorStorage):
  def __init__(self, chunks: List[Document], embeddings) -> None:
    self.collection_name = "catholic"
    persist_directory = Path(__file__).resolve().parent.parent.joinpath("data", "vector_stores").as_posix()
    self.chroma_client = chromadb.PersistentClient(path=persist_directory)
    super().__init__(chunks=chunks, embeddings=embeddings)

  def _load_storage(self) -> None:
    # self.chroma_client.delete_collection(name=self.collection_name)
    is_existing_collection = self.collection_name in [c.name for c in self.chroma_client.list_collections()]
  
    collection = self.chroma_client.get_or_create_collection(self.collection_name)
    
    if not is_existing_collection:
      for batch in create_batches(
        api=self.chroma_client,
        ids=[str(uuid.uuid4()) for _ in range(len(self.chunks))],
        metadatas=[c.metadata for c in self.chunks],
        documents=[c.page_content for c in self.chunks]
      ):
        print(f"LOAD Documents of batch {len(batch[0])} into vectorstore...")
        collection.add(*batch)
    else:
      print("Collection already exist.")
    
    self.vector_store = Chroma(
      client=self.chroma_client,
      collection_name=self.collection_name,
      embedding_function=self.embeddings
    )
