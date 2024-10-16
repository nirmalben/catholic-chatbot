from RagChain import RagChain
import streamlit as st

class Chatbot():
  def _clear_chat_history(self):
    st.session_state.messages = [
      {
        "role": "assistant",
        "content": "How can I assist you today? Let me know your questions and I can answer your queries based on Catholicism",
      }
    ]
    st.session_state.memory.clear()
    
  def _get_response(self, prompt):
    response = RagChain().get_chain().invoke({"question": prompt})
    answer = response["answer"]

    answer = answer[answer.find("\nAnswer: ") + len("\nAnswer: ") :]

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
      st.markdown(answer)
      with st.expander("**Source documents**"):
        documents_content = ""
        for document in response["source_documents"]:
          try:
            page = " (Page: " + str(document.metadata["page"]) + ")"
          except:
            page = ""
          documents_content += (
            "**Source: "
            + str(document.metadata["source"])
            + page
            + "**\n\n"
          )
          documents_content += document.page_content + "\n\n\n"
        st.markdown(documents_content)
  
  def run(self):
    st.set_page_config(page_title="Catholic Chatbot")
    with st.sidebar:
      st.title('Catholic Chatbot')
    
    col1, col2 = st.columns([7, 3])
    with col1:
      st.subheader("Chat with your data")
    with col2:
      st.button("Clear Chat History", on_click=self._clear_chat_history)

    if "messages" not in st.session_state:
      st.session_state.messages = [
        {
          "role": "assistant",
          "content": "How can I assist you today? Let me know your questions and I can answer your queries based on Catholicism",
        }
      ]
    
    for msg in st.session_state.messages:
      st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
      with st.spinner("Running..."):
        self._get_response(prompt=prompt)

if __name__ == "__main__":
  Chatbot().run()
