import streamlit as st
import os
import tempfile
import asyncio
import zipfile
import io
import re
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from llm_handler import LLMHandler
from image_processor import ImageProcessor
import nest_asyncio

# Apply the patch to allow nested asyncio event loops (Fixes "Event loop is closed" error)
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

# Page config
st.set_page_config(page_title="PDF Intelligence Agent", page_icon="🤖", layout="wide")

# --- Agent Class and Helper Functions ---

class PDFAgent:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.llm_handler = LLMHandler()
        self.image_processor = ImageProcessor()

    async def process_documents(self, uploaded_files):
        all_chunks, all_images = [], []
        progress_bar = st.progress(0, text="Starting document processing...")
        
        for i, file in enumerate(uploaded_files):
            progress_bar.progress((i + 0.5) / len(uploaded_files), text=f"Processing {file.name}...")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                text_chunks = await self.pdf_processor.extract_text_chunks(tmp_path)
                images = await self.pdf_processor.extract_images(tmp_path)
                
                # Perform OCR on extracted images
                for img_data in images:
                    ocr_text = await self.image_processor.extract_text_from_image(img_data['image'])
                    if ocr_text:
                        img_data['ocr_text'] = ocr_text.lower()
                
                all_chunks.extend(text_chunks)
                all_images.extend(images)
            finally:
                os.unlink(tmp_path)
        
        progress_bar.progress(0.9, text="Creating document embeddings...")
        if all_chunks:
            await self.vector_store.add_documents(all_chunks)
        
        progress_bar.progress(1.0, text="Processing complete!")
        st.success("Documents processed successfully!")
        return all_chunks, all_images

    async def query_documents(self, query, top_k=8):
        """The agent's core query logic."""
        relevant_chunks = await self.vector_store.similarity_search(query, k=top_k)
        response = await self.llm_handler.generate_response(query, relevant_chunks)
        return response

def parse_source_pages(response_text: str) -> list[int]:
    """Parses page numbers from the AI's source citation, e.g., '(Source: Page 10)'."""
    match = re.search(r'\(Source: (.*?)\)', response_text, re.IGNORECASE)
    return [int(p) for p in re.findall(r'\d+', match.group(1))] if match else []

def find_relevant_images(cited_pages: list, all_images: list) -> list:
    """
    Finds ALL images that appear on the specific page numbers cited by the LLM.
    This is the strictest and most accurate method for ensuring relevance.
    """
    if not cited_pages:
        return []
    
    found_images = [img for img in all_images if img['page'] in cited_pages]
    found_images.sort(key=lambda x: (x['page'], x['index']))
    return found_images

# --- Streamlit UI and Application Flow ---

def main():
    st.markdown("<h1>🤖 PDF Intelligence Agent</h1>", unsafe_allow_html=True)
    st.markdown("Upload your documents and ask me anything about their content.")

    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = PDFAgent()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "images" not in st.session_state:
        st.session_state.images = []
    if "processed" not in st.session_state:
        st.session_state.processed = False

    with st.sidebar:
        st.header("⚙️ Controls")
        uploaded_files = st.file_uploader("Upload PDF documents", type=['pdf'], accept_multiple_files=True)
        if uploaded_files and not st.session_state.processed:
            if st.button("🚀 Process Documents", type="primary"):
                st.session_state.processed = True
                st.session_state.messages = []
                st.session_state.images = []
                with st.spinner("Analyzing documents... This may take a moment."):
                    _, images = asyncio.run(st.session_state.agent.process_documents(uploaded_files))
                    st.session_state.images = images
                st.rerun()

        if st.session_state.processed:
             st.success("Documents are ready for questions.")
        if st.button("🗑️ Clear Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Chat interface
    # Display existing chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display images if they exist in the message history
            if message.get("images"):
                st.write("---")
                st.markdown(f"**Relevant Images ({len(message['images'])} found):**")
                cols = st.columns(3)
                for i, img in enumerate(message["images"]):
                    with cols[i % 3]:
                        st.image(img['image'], caption=f"Page {img['page']}", use_container_width=True)

    # Handle new user input
    if query := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.processed:
            st.warning("Please upload and process documents before asking questions.")
        else:
            # Add user message to history, preparing for the UI to update
            st.session_state.messages.append({"role": "user", "content": query})

            # Get the agent's response
            with st.spinner("Thinking..."):
                response_text = asyncio.run(st.session_state.agent.query_documents(query))
                
                # Check if the response is a valid answer or a refusal/chitchat
                no_answer_phrases = ["i can only answer", "could not find an answer", "you're welcome", "hello!"]
                is_valid_answer = not any(phrase in response_text.lower() for phrase in no_answer_phrases)
                
                images_to_display = []
                # ONLY search for images if the LLM provided a valid, document-related answer
                if is_valid_answer:
                    cited_pages = parse_source_pages(response_text)
                    images_to_display = find_relevant_images(cited_pages, st.session_state.images)

                # Add the complete assistant response (text + images) to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "images": images_to_display
                })

            # Rerun the script to display the new messages that were just added to the history
            st.rerun()

if __name__ == "__main__":
    if not os.getenv('GOOGLE_API_KEY'):
        st.error("❌ GOOGLE_API_KEY not found. Please set it in your environment or Streamlit Cloud secrets.")
    else:
        main()