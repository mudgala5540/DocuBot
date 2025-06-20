import streamlit as st
import os
import tempfile
from pathlib import Path
import asyncio
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from llm_handler import LLMHandler
from image_processor import ImageProcessor
import zipfile
import io
from dotenv import load_dotenv
import re
import nest_asyncio

# Apply the patch to allow nested asyncio event loops
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

# Page config
st.set_page_config(
    page_title="PDF Intelligence Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Add other CSS styles if you need them */
</style>
""", unsafe_allow_html=True)


class PDFAgent:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.llm_handler = LLMHandler()
        self.image_processor = ImageProcessor()

    async def process_documents(self, uploaded_files, progress_callback=None):
        all_chunks = []
        all_images = []

        for i, uploaded_file in enumerate(uploaded_files):
            if progress_callback:
                progress_callback(i / len(uploaded_files), f"Processing {uploaded_file.name}")

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                text_chunks = await self.pdf_processor.extract_text_chunks(tmp_path)
                images = await self.pdf_processor.extract_images(tmp_path)

                for img_data in images:
                    ocr_text = await self.image_processor.extract_text_from_image(img_data['image'])
                    if ocr_text.strip():
                        img_data['ocr_text'] = ocr_text.lower()

                all_chunks.extend(text_chunks)
                all_images.extend(images)

            finally:
                os.unlink(tmp_path)

        if progress_callback:
            progress_callback(0.8, "Creating vector embeddings...")

        if all_chunks:
            await self.vector_store.add_documents(all_chunks)

        if progress_callback:
            progress_callback(1.0, "Processing complete!")

        return all_chunks, all_images

    async def query_documents(self, query, top_k=8):
        relevant_chunks = await self.vector_store.similarity_search(query, k=top_k)
        response = await self.llm_handler.generate_response(query, relevant_chunks)
        return response ## <-- FIX: This method should only return one value (the response text).

def parse_source_pages(response_text: str) -> list[int]:
    """Parses page numbers from the AI's source citation, e.g., '(Source: Page 10, Page 201)'."""
    match = re.search(r'\(Source: (.*?)\)', response_text, re.IGNORECASE)
    return [int(p) for p in re.findall(r'\d+', match.group(1))] if match else []

def find_relevant_images(cited_pages: list, all_images: list) -> list:
    """Finds ALL images that appear on the specific page numbers cited by the LLM."""
    if not cited_pages:
        return []
    
    found_images = [img for img in all_images if img['page'] in cited_pages]
    found_images.sort(key=lambda x: (x['page'], x['index']))
    return found_images

def main():
    st.markdown('<div class="main-header"><h1>🤖 PDF Intelligence Agent</h1><p>Advanced AI-powered document analysis</p></div>', unsafe_allow_html=True)

    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = PDFAgent()
    if 'processed_docs' not in st.session_state:
        st.session_state.processed_docs = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'extracted_images' not in st.session_state:
        st.session_state.extracted_images = []
    if 'text_chunks' not in st.session_state:
        st.session_state.text_chunks = []


    # Sidebar
    with st.sidebar:
        st.header("📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files for processing"
        )

        if uploaded_files and not st.session_state.get('processing_started', False):
            if st.button("🚀 Process Documents", type="primary"):
                st.session_state.processing_started = True
                with st.spinner("Processing documents... This may take a moment."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(progress, status):
                        progress_bar.progress(progress)
                        status_text.text(status)

                    chunks, images = asyncio.run(
                        st.session_state.agent.process_documents(
                            uploaded_files,
                            update_progress
                        )
                    )

                    st.session_state.processed_docs = True
                    st.session_state.extracted_images = images
                    st.session_state.text_chunks = chunks
                    st.success(f"✅ Processed {len(chunks)} text chunks and {len(images)} images!")
                    st.rerun()

        if st.session_state.processed_docs:
            st.success("📚 Documents ready for querying!")
            st.subheader("📊 Document Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Text Chunks", len(st.session_state.text_chunks))
            with col2:
                st.metric("Images", len(st.session_state.extracted_images))

            if st.session_state.extracted_images:
                if st.button("📥 Download All Images"):
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        for i, img_data in enumerate(st.session_state.extracted_images):
                            img_bytes = io.BytesIO()
                            img_data['image'].save(img_bytes, format='PNG')
                            zip_file.writestr(f"image_{i+1}_page_{img_data['page']}.png", img_bytes.getvalue())

                    st.download_button(
                        label="Download Images.zip",
                        data=zip_buffer.getvalue(),
                        file_name="extracted_images.zip",
                        mime="application/zip"
                    )

        if st.button("🗑️ Clear Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Main content area
    if st.session_state.processed_docs:
        st.subheader("💬 Chat with Your Documents")

        # Display chat history from session state
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("images"):
                    st.write("---")
                    st.markdown(f"**Relevant Images ({len(message['images'])} found):**")
                    cols = st.columns(3)
                    for i, img_data in enumerate(message["images"]):
                        with cols[i % 3]:
                            st.image(img_data['image'], caption=f"Page {img_data['page']}", use_container_width=True)

        #One thats comprehensive
                # Handle user input
        if query := st.chat_input("Ask anything about your documents..."):
            # Add user message to history and display it immediately
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # Display assistant response in a container
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # 1. Get the raw response from the LLM
                    response_text, _ = asyncio.run(
                        st.session_state.agent.query_documents(query)
                    )
                    
                    # 2. Check if the response is a valid answer or a refusal/chitchat
                    no_answer_phrases = ["i can only answer", "could not find an answer", "you're welcome", "hello!"]
                    is_valid_answer = not any(phrase in response_text.lower() for phrase in no_answer_phrases)
                    
                    images_to_display = []
                    # 3. ONLY search for images if it was a valid, document-related answer
                    if is_valid_answer:
                        cited_pages = parse_source_pages(response_text)
                        images_to_display = find_relevant_images(cited_pages, st.session_state.extracted_images)

                    # 4. Display the response text and the images (if any)
                    st.markdown(response_text)
                    if images_to_display:
                        st.write("---")
                        st.markdown(f"**Relevant Images ({len(images_to_display)} found):**")
                        cols = st.columns(3)
                        for i, img_data in enumerate(images_to_display):
                            with cols[i % 3]:
                                st.image(img_data['image'], caption=f"Page {img_data['page']}", use_container_width=True)

            # 5. Add the complete message bundle to session state for future reruns
                      st.session_state.chat_history.append({
                        "role": "assistant", 
                "content": response_text,
                "images": images_to_display
            })
        

        # Handle user input
        if query := st.chat_input("Ask anything about your documents..."):
            st.session_state.chat_history.append({"role": "user", "content": query})

            with st.spinner("Thinking..."):
                # 1. Get the raw response from the LLM
                response_text = asyncio.run( ## <-- FIX: This call now correctly expects only one return value.
                    st.session_state.agent.query_documents(query)
                )
                
                # 2. Check if the response is a valid answer or a refusal/chitchat
                no_answer_phrases = ["i can only answer", "could not find an answer", "you're welcome", "hello!"]
                is_valid_answer = not any(phrase in response_text.lower() for phrase in no_answer_phrases)
                
                images_to_display = []
                # 3. ONLY search for images if it was a valid, document-related answer
                if is_valid_answer:
                    cited_pages = parse_source_pages(response_text)
                    images_to_display = find_relevant_images(cited_pages, st.session_state.extracted_images)

                # 4. Add the complete message bundle to session state for future reruns
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response_text,
                    "images": images_to_display
                })
            
            st.rerun()

    else:
        st.info("👋 Welcome! Please upload your PDF documents in the sidebar to get started.")


if __name__ == "__main__":
    if not os.getenv('GOOGLE_API_KEY'):
        st.error("❌ Please set GOOGLE_API_KEY in your .env file or Streamlit secrets")
        st.stop()
    main()