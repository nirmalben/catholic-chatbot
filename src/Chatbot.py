from RagChain import RagChain
import streamlit as st

class Chatbot():
  def __init__(self, rag) -> None:
    self.rag = rag
  
  def _clear_chat_history(self):
    st.session_state.messages = [
      {
        "role": "assistant",
        "content": "I am an expert in Catholicism. How can I assist you today?",
      }
    ]
    self.rag.clear_memory()
    
  def _get_response(self, prompt):
    response = self.rag.get_chain().invoke({"question": prompt})
    
    answer = response["answer"]
    answer = answer[answer.find("Answer:") + len("Answer:") :]
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
      st.markdown(answer)
      with st.expander("**Source documents**"):
        documents_content = ""
        for document in response["source_documents"]:
          try:
            source = str(document.metadata["from"]) + " (#" + str(document.metadata["id"]) + ")"
          except:
            book = str(document.metadata["book"])
            chapter = str(document.metadata["chapter"])
            verse = str(document.metadata["verse"])
            source = str(document.metadata["from"]) + " (" + book + " " + chapter + ":" + verse + ")"
          documents_content += (
            "**Source: "
            + source
            + "**\n\n"
          )
          documents_content += document.page_content + "\n\n\n"
        st.markdown(documents_content)
  
  def run(self):
    st.set_page_config(page_title="Catholic Chatbot",  layout="wide")
    with st.sidebar:
      st.button("Clear Chat History", on_click=self._clear_chat_history)
    st.subheader("Catholic Chatbot")

    if "messages" not in st.session_state:
      st.session_state.messages = [
        {
          "role": "assistant",
          "content": "I am an expert in Catholicism. How can I assist you today?",
        }
      ]
    
    for msg in st.session_state.messages:
      st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
      with st.spinner("Running..."):
        self._get_response(prompt=prompt)

if __name__ == "__main__":
  Chatbot(rag=RagChain()).run()
