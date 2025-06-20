import streamlit as st
import os
import tempfile
import asyncio
import re
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from llm_handler import LLMHandler

# --- The Definitive Fix for "different loop" Error ---
def run_async(coro):
    """A helper function to run asyncio code in Streamlit's environment."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="PDF Intelligence Agent", page_icon="🤖", layout="wide")

# --- UI & Agent Logic ---

class DocAgentApp:
    def __init__(self):
        st.markdown("""
        <style>
            .main-header { font-size: 2.5rem; font-weight: bold; text-align: center; margin-bottom: 2rem; }
            .stChatMessage { background-color: #f8f9fa; border-radius: 0.5rem; border: 1px solid #e0e0e0; }
        </style>
        """, unsafe_allow_html=True)
        if 'agent' not in st.session_state:
            st.session_state.agent = PDFAgent()
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "images" not in st.session_state:
            st.session_state.images = []
        if "processed" not in st.session_state:
            st.session_state.processed = False

    def render_sidebar(self):
        with st.sidebar:
            st.header("⚙️ Controls")
            uploaded_files = st.file_uploader("Upload PDF documents", type=['pdf'], accept_multiple_files=True)
            if uploaded_files and not st.session_state.processed:
                if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                    st.session_state.processed = True
                    st.session_state.messages = []
                    st.session_state.images = []
                    with st.spinner("Analyzing documents... This may take a moment."):
                        _, images = run_async(st.session_state.agent.process_documents(uploaded_files))
                        st.session_state.images = images
                    st.rerun()
            if st.session_state.processed:
                st.success("Documents are ready.")
            if st.button("🗑️ Clear Session", use_container_width=True):
                for key in list(st.session_state.keys()): del st.session_state[key]
                st.rerun()

    def render_chat(self):
        if not st.session_state.processed:
            st.info("👋 Welcome! Please upload your PDF documents in the sidebar and click 'Process' to get started.")
            return

        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])
                if message.get("images"):
                    st.write("---")
                    st.markdown(f"**Relevant Images ({len(message['images'])} found):**")
                    cols = st.columns(4)
                    for i, img in enumerate(message["images"]):
                        with cols[i % 4]:
                            st.image(img['image'], caption=f"Page {img['page']}", use_container_width=True)
        
        if query := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user", avatar="👤"):
                st.markdown(query)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    response_text = run_async(st.session_state.agent.query_documents(query))
                    no_answer_phrases = ["i can only answer", "could not find an answer", "you're welcome", "hello!"]
                    is_valid_answer = not any(phrase in response_text.lower() for phrase in no_answer_phrases)
                    images_to_display = []
                    if is_valid_answer:
                        cited_pages = self.parse_source_pages(response_text)
                        images_to_display = self.find_relevant_images(cited_pages, st.session_state.images)
                    
                    st.markdown(response_text)
                    if images_to_display:
                        st.write("---")
                        st.markdown(f"**Relevant Images ({len(images_to_display)} found):**")
                        cols = st.columns(4)
                        for i, img in enumerate(images_to_display):
                            with cols[i % 4]:
                                st.image(img['image'], caption=f"Page {img['page']}", use_container_width=True)

            st.session_state.messages.append({"role": "assistant", "content": response_text, "images": images_to_display})

    def parse_source_pages(self, response_text: str) -> list[int]:
        match = re.search(r'\(Source: (.*?)\)', response_text, re.IGNORECASE)
        return [int(p) for p in re.findall(r'\d+', match.group(1))] if match else []

    def find_relevant_images(self, cited_pages: list, all_images: list) -> list:
        if not cited_pages: return []
        found_images = [img for img in all_images if img['page'] in cited_pages]
        found_images.sort(key=lambda x: (x['page'], x['index']))
        return found_images

    def run(self):
        st.markdown("<div class='main-header'>PDF Intelligence Agent</div>", unsafe_allow_html=True)
        self.render_sidebar()
        self.render_chat()

if __name__ == "__main__":
    if not os.getenv('GOOGLE_API_KEY'):
        st.error("❌ GOOGLE_API_KEY not found. Please set it in your environment or Streamlit Cloud secrets.")
    else:
        app = DocAgentApp()
        app.run()