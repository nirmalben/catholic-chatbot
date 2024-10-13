from dotenv import load_dotenv
from JsonDocumentLoader import JsonDocumentLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

from typing import List

import itertools
import os
import pinecone

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
        jq_schema='.[] | . as $grandparent | .chapters[] | . as $parent | .verses[] | {book: $grandparent.book, chapter: $parent.chapter, verse: .verse, text: .text}',
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

    def _build(self):
      # Load documents
      documents = self._load_documents()
      
      # Split documents into chunks
      chunks = self._split_to_chunks(documents)
      
      # Initialize embeddings
      embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=self._HUGGING_FACE_API_KEY, 
        model_name="thenlper/gte-large"
      )

      # Vectorstore for the chunks
      pinecone.init(api_key= os.getenv('PINECONE_API_KEY'))

      if PINECONE_INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(name=PINECONE_INDEX_NAME, metric="cosine", dimension=768)
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=PINECONE_INDEX_NAME)
      else:
        vector_store = Pinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)

      # Retriever
      
      
      # Memory
      
      
      # Conversational chain
    
if __name__ == '__main__':
  _ = RagChain()
