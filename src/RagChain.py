from dotenv import load_dotenv
from JsonDocumentLoader import JsonDocumentLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from typing import List

import itertools
import os

PINECONE_INDEX_NAME = "catholic-chatbot-index"

class RagChain():
  def __init__(self) -> None:
    load_dotenv()
    self._HUGGING_FACE_API_KEY = os.getenv('HUGGING_FACE_API_KEY')
    self._PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    self._build()

  def _load_bible(self) -> List[Document]:
    def metadata_func(record: dict, metadata: dict) -> dict:
      metadata["book"] = record.get("book")
      metadata["chapter"] = record.get("chapter")
      metadata["verse"] = record.get("verse")
      return metadata
    
    return JsonDocumentLoader(
      json_file_path='./data/nabre.json',
      metadata_func=metadata_func,
      jq_schema='.[] | . as $grandparent | .chapters[] | . as $parent | .verses[] | {book: $grandparent.book, chapter: $parent.chapter, verse: .verse, text: .text}'
    ).get_documents()

  def _load_documents(self) -> List[Document]:
    nabre_bible_doc = self._load_bible()
    
    catechism_doc = JsonDocumentLoader(json_file_path='./data/catechism.json').get_documents()

    canon_doc = JsonDocumentLoader(
      json_file_path='./data/canon.json',
      jq_schema='.[] | if has("sections") then . as $section | .sections[] | {id: [($section.id | tostring), (.id | tostring)] | join(".") | tonumber, text: .text} else {id: (.id | tonumber), text: .text} end'
    ).get_documents()
    
    girm_doc = JsonDocumentLoader(json_file_path='./data/girm.json').get_documents() # General Instruction of the Roman Missal
    
    documents = [nabre_bible_doc, catechism_doc, canon_doc, girm_doc]
    return list(itertools.chain(*documents)) # converts List[List[Documents]] to List[Documents]

  def _split_to_chunks(self, documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks
  
  def _get_vector_store(self, chunks, embeddings):
    pc = Pinecone(api_key= os.getenv('PINECONE_API_KEY'))
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    is_index_new = False
    if PINECONE_INDEX_NAME not in existing_indexes:
      is_index_new = True
      pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1024,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        metric="cosine")
    index = pc.Index(PINECONE_INDEX_NAME)
    
    if is_index_new:
      PineconeVectorStore.from_documents(chunks, embeddings, index_name=PINECONE_INDEX_NAME)

    return PineconeVectorStore(index=index, embedding=embeddings)

  def _build(self):
    # Load documents
    documents = self._load_documents()
    
    # Split documents into chunks
    chunks = self._split_to_chunks(documents=documents)
    
    # Initialize embeddings
    embeddings = HuggingFaceInferenceAPIEmbeddings(
      api_key=self._HUGGING_FACE_API_KEY, 
      model_name="thenlper/gte-large") # https://huggingface.co/thenlper/gte-large

    # Vectorstore for the chunks
    vector_store = self._get_vector_store(chunks=chunks, embeddings=embeddings)
    
    # Retriever
    
    # Memory
    
    # Conversational chain
    
if __name__ == '__main__':
  _ = RagChain()
