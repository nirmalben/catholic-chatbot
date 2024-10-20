from ChromaVectorStorage import ChromaVectorStorage
from CompressionRetriever import CompressionRetriever
from JsonDocumentLoader import JsonDocumentLoader
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_core.documents.base import Document
from langchain_huggingface import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from typing import List

import itertools

LLM_NAME = 'mistralai/Mixtral-8x7B-Instruct-v0.1' # https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1

class RagChain():
  def __init__(self, _HUGGING_FACE_API_KEY:str, session_state) -> None:
    self._HUGGING_FACE_API_KEY = _HUGGING_FACE_API_KEY
    self.chain = None
    self.session_state = session_state
    self.parent_path = Path(__file__).resolve().parent.parent.as_posix()
    self._build()

  def _load_bible(self) -> List[Document]:
    def metadata_func(record: dict, metadata: dict) -> dict:
      metadata["book"] = record.get("book")
      metadata["chapter"] = record.get("chapter")
      metadata["verse"] = record.get("verse")
      metadata["from"] = "New American Bible (Revised Edition) (NABRE)"
      return metadata
    
    return JsonDocumentLoader(
      json_file_path=self.parent_path + '/data/nabre.json',
      metadata_func=metadata_func,
      jq_schema='.[] | . as $grandparent | .chapters[] | . as $parent | .verses[] | {book: $grandparent.book, chapter: $parent.chapter, verse: .verse, text: .text}'
    ).get_documents()

  def _load_documents(self) -> List[Document]:
    nabre_bible_doc = self._load_bible()
    
    catechism_doc = JsonDocumentLoader(
      json_file_path=self.parent_path + '/data/catechism.json',
      jq_schema='.[] | {id: (.id | tonumber), text: .text, from: "Catechism of the Catholic Church" }'
    ).get_documents()

    canon_doc = JsonDocumentLoader(
      json_file_path=self.parent_path + '/data/canon.json',
      jq_schema='.[] | if has("sections") then . as $section | .sections[] | {id: [($section.id | tostring), (.id | tostring)] | join(".") | tonumber, text: .text, from: "Canon law of the Catholic Church"} else {id: (.id | tonumber), text: .text, from: "Canon law of the Catholic Church" } end'
    ).get_documents()
    
    # General Instruction of the Roman Missal
    girm_doc = JsonDocumentLoader(
      json_file_path=self.parent_path + '/data/girm.json',
      jq_schema='.[] | {id: (.id | tonumber), text: .text, from: "General Instruction of the Roman Missal"}'
    ).get_documents() 
    
    documents = [nabre_bible_doc, catechism_doc, canon_doc, girm_doc]
    return list(itertools.chain(*documents)) # converts List[List[Documents]] to List[Documents]

  def _split_to_chunks(self, documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=384, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks

  def _build(self):
    if "init_done" not in self.session_state:
      print("Loading Documents...")
      documents = self._load_documents()
      
      print("Splitting documents into chunks...")
      chunks = self._split_to_chunks(documents=documents)
      
      self.session_state["init_done"] = chunks
    else:
      chunks = self.session_state["init_done"]

    print("Initialize embeddings...")
    embeddings = HuggingFaceInferenceAPIEmbeddings(
      api_key=self._HUGGING_FACE_API_KEY, 
      model_name="sentence-transformers/all-MiniLM-L6-v2") # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    
    print("Initialize vectorstore...")
    vector_store = ChromaVectorStorage(chunks=chunks, embeddings=embeddings).get_vector_store()

    # print("Perform similarity search...")
    # print(vector_store.similarity_search("Mother Mary is our intercessor."))
    
    # Retriever
    retriever = CompressionRetriever(base_retriever=vector_store.as_retriever(), embeddings=embeddings).get_retriever()
    
    # Memory
    self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history", output_key="answer", input_key="question")
    
    # Conversational chain
    condense_question_prompt = PromptTemplate(
      input_variables=["chat_history", "question"],
      template="""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\n\n
      Chat History:\n{chat_history}\n
      Follow Up Input: {question}\n
      Standalone question:""")
    
    answer_template = """You are an expert in only offering tips about life using Catholicism and also answer questions on Catholicism. You have been asked the following question and if the question pertains to something that can answered with Catholicism only, using the following context, you can answer it. If the question is generic, does not pertain to Catholicism or does not relate to Catholicism, you can say that the question cannot be answered with Catholicism. If there is no question, you can respond accordingly\n\n
    Chat History:\n{chat_history}\n
    Context:\n{context}\n
    
    Question: {question}\n
    Answer:"""
    answer_prompt = ChatPromptTemplate.from_template(answer_template)

    standalone_query_llm = HuggingFaceEndpoint(
      repo_id=LLM_NAME,
      huggingfacehub_api_token=self._HUGGING_FACE_API_KEY,
      temperature=0.1,
      top_p=0.95,
      do_sample=True,
      max_new_tokens=1024
    )
    response_llm = HuggingFaceEndpoint(
      repo_id=LLM_NAME,
      huggingfacehub_api_token=self._HUGGING_FACE_API_KEY,
      temperature=0.1,
      top_p=0.95,
      do_sample=True,
      max_new_tokens=1024
    )

    self.chain = ConversationalRetrievalChain.from_llm(
      condense_question_prompt=condense_question_prompt,
      combine_docs_chain_kwargs={"prompt": answer_prompt},
      condense_question_llm=standalone_query_llm,
      llm=response_llm,
      memory=self.memory,
      retriever=retriever,
      chain_type="stuff",
      verbose=False,
      return_source_documents=True)
  
  def get_chain(self):
    return self.chain
  
  def clear_memory(self):
    self.memory.clear()
    
if __name__ == '__main__':
  _ = RagChain()
