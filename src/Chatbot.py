from PIL import Image
from RagChain import RagChain
from pathlib import Path
import streamlit as st

SAMPLE_PROMPTS = [
  "How can you help me?",
  "What does the Book of Sirach say about hard work?",
  "How can I forgive?",
  "How can Mother Mary pray for me?"
]

class Chatbot():
  def __init__(self, rag) -> None:
    self.rag = rag

  def _clear_chat_history(self):
    st.session_state.messages = [
      {
        "role": "assistant",
        "content": "How can I assist you today?",
      }
    ]
    self.rag.clear_memory()
    
  def _get_response(self, prompt):
    with st.spinner("Running..."):
      try:
        response = self.rag.get_chain().invoke({"question": prompt})
      except:
        st.chat_message("user", avatar="👀").write(prompt)
        st.chat_message("assistant", avatar="💡").write(
          "Apologies! Something went wrong while processing your request. Please feel free to try again, rephrase your question or ask something else.")
        return

      answer = response["answer"]
      # print(answer)
      # answer = answer[answer.find("Answer:") + len("Answer:") :]
      
      st.session_state.messages.append({"role": "user", "content": prompt})
      st.session_state.messages.append({"role": "assistant", "content": answer})
      
      st.chat_message("user", avatar="👀").write(prompt)
      with st.chat_message("assistant", avatar="💡"):
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
    prompt = None
    parent_path = Path(__file__).resolve().parent.as_posix()
    img = Image.open(parent_path + "/ui/praying.png")
    img = img.resize((50, 50))
    
    st.set_page_config(page_title="CathWalk", layout="wide", page_icon=img)

    st.subheader("CathWalk")
    st.caption("Your privacy is important; none of your questions or prompts are stored or shared. I'm not perfect and may make mistakes. If unsure, please consult a human.")
    sample_prompt_buttons = st.columns(len(SAMPLE_PROMPTS) + 1)
    for i, sample_prompt in enumerate(sample_prompt_buttons[:-1]):
      if sample_prompt.button(SAMPLE_PROMPTS[i], use_container_width=True):
        prompt = SAMPLE_PROMPTS[i]
    sample_prompt_buttons[-1].button("Clear Chat History", use_container_width=True, on_click=self._clear_chat_history, type="primary")
    
    st.sidebar.image(img)
    with open(parent_path + "/ui/sidebar.md", "r") as sidebar_file:
      sidebar_content = sidebar_file.read()
    st.sidebar.markdown(sidebar_content)
    st.sidebar.info('Coming soon: Miracles!', icon='⚡️')
    st.sidebar.markdown('<a href="mailto:njbenann@gmail.com" style="text-decoration:none">Contact</a>', unsafe_allow_html=True)
    # st.sidebar.caption("Powered by Mixtral-8x7B-Instruct-v0.1 and all-MiniLM-L6-v2 using Hugging Face and Chroma")
    
    if "messages" not in st.session_state:
      st.session_state.messages = [
        {
          "role": "assistant",
          "content": "How can I assist you today?",
        }
      ]
    
    for msg in st.session_state.messages:
      if msg["role"] == "assistant":
        st.chat_message(msg["role"], avatar="💡").write(msg["content"])
      if msg["role"] == "user":
        st.chat_message(msg["role"], avatar="👀").write(msg["content"])

    if prompt := st.chat_input(placeholder="Ask me!") or prompt:
      self._get_response(prompt=prompt)

if __name__ == "__main__":
  Chatbot(rag=RagChain(st.secrets["HUGGING_FACE_API_KEY"], st.session_state)).run()
