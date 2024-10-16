from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.text_splitter import CharacterTextSplitter

class CompressionRetriever():
  def __init__(self, base_retriever, embeddings, chunk_size=384, k=4, similarity_threshold=None) -> None:
    # Initialize splitter to breakdown documents into smaller chunks
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, separator=". ")

    # Initialize filter to remove redundant documents
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

    # Initialize filter to filter based on relevance to the query
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, k=k, similarity_threshold=similarity_threshold)

    # Reorder the documents such that less relevant document will be at the middle of the list 
    # and more relevant elements at beginning / end
    reordering = LongContextReorder()

    # Initialze compressor pipeline with the above transformers
    pipeline_compressor = DocumentCompressorPipeline(
      transformers=[splitter, redundant_filter, relevant_filter, reordering]
    )
    
    self.retriever = ContextualCompressionRetriever(
      base_compressor=pipeline_compressor, base_retriever=base_retriever
    )

  def get_retriever(self):
    return self.retriever
