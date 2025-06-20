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
</style>
""", unsafe_allow_html=True)

class PDFAgent:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.llm_handler = LLMHandler()
        self.image_processor = ImageProcessor()
        
    async def process_documents(self, uploaded_files, progress_callback=None):
        """Process uploaded PDF documents"""
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
                        img_data['ocr_text'] = ocr_text
                
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
    
    async def query_documents(self, query, top_k=5):
        """Query processed documents"""
        relevant_chunks = await self.vector_store.similarity_search(query, k=top_k)
        response = await self.llm_handler.generate_response(query, relevant_chunks)
        # We return relevant_chunks here so the new logic can use them if needed,
        # but the primary method will be parsing the response text.
        return response, relevant_chunks

# --- NEW HELPER FUNCTIONS FOR SMARTER LOGIC ---

def is_small_talk(query):
    """Checks if a query is simple small talk."""
    greetings = ['hi', 'hello', 'hey', 'how are you', 'good morning', 'good afternoon', 'thank you']
    query_lower = query.lower().strip()
    if query_lower in greetings:
        return True
    return any(greet in query_lower for greet in greetings) and len(query_lower.split()) <= 3

def parse_source_pages_from_response(response_text: str) -> list[int]:
    """Parses page numbers from a string like '(Source: Page 10, Page 201)'."""
    import re
    match = re.search(r'\(Source: (.*?)\)', response_text, re.IGNORECASE)
    if not match:
        return []
    
    pages_str = match.group(1)
    page_numbers = re.findall(r'\d+', pages_str)
    
    return [int(p) for p in page_numbers]

def get_relevant_images_from_pages(page_numbers: list[int], all_images: list) -> list:
    """Finds images from a specific list of page numbers."""
    if not page_numbers:
        return []
        
    found_images = [
        img for img in all_images 
        if img['page'] in page_numbers
    ]
    
    found_images.sort(key=lambda x: x['page'])
    return found_images


def main():
    st.markdown('<div class="main-header"><h1>🤖 PDF Intelligence Agent</h1><p>Advanced AI-powered document analysis with RAG and multi-modal processing</p></div>', unsafe_allow_html=True)
    
    if 'agent' not in st.session_state:
        st.session_state.agent = PDFAgent()
    if 'processed_docs' not in st.session_state:
        st.session_state.processed_docs = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'extracted_images' not in st.session_state:
        st.session_state.extracted_images = []
    
    with st.sidebar:
        st.header("📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files for processing"
        )
        
        if uploaded_files and not st.session_state.processed_docs:
            if st.button("🚀 Process Documents", type="primary"):
                with st.spinner("Analyzing documents... This may take a moment."):
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
                    
                    st.success(f"✅ Processed {len(chunks)} text chunks and {len(images)} images!")
        
        if st.session_state.processed_docs:
            st.success("📚 Documents ready for querying!")
            
            st.subheader("📊 Document Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Text Chunks", len(st.session_state.agent.vector_store.documents))
            with col2:
                st.metric("Images", len(st.session_state.extracted_images))
            
            if st.session_state.extracted_images:
                if st.button("📥 Download Images"):
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        for i, img_data in enumerate(st.session_state.extracted_images):
                            img_bytes = io.BytesIO()
                            img_data['image'].save(img_bytes, format='PNG')
                            zip_file.writestr(f"image_{i+1}_page_{img_data['page']}.png", img_bytes.getvalue())
                    
                    st.download_button(
                        label="Download Images ZIP",
                        data=zip_buffer.getvalue(),
                        file_name="extracted_images.zip",
                        mime="application/zip"
                    )
        
        if st.button("🗑️ Clear Session"):
            keys_to_delete = list(st.session_state.keys())
            for key in keys_to_delete:
                del st.session_state[key]
            st.rerun()

    # --- NEW, SMARTER MAIN CONTENT AREA ---
    if st.session_state.processed_docs:
        st.subheader("💬 Chat with Your Documents")

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "images" in message and message["images"]:
                    st.write("**Relevant Images:**")
                    cols = st.columns(3)
                    for i, img_data in enumerate(message["images"][:6]): # Show max 6 relevant images
                        with cols[i % 3]:
                            st.image(img_data['image'], caption=f"Page {img_data['page']}", use_container_width=True)

        # Query input
        query = st.chat_input("Ask anything about your documents...")

        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})

            # Handle small talk first
            if is_small_talk(query):
                ai_response = "Hello! I'm ready to help. Please ask me anything about your documents."
                if "thank" in query.lower():
                    ai_response = "You're welcome! Is there anything else I can help with?"
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                st.rerun()
            else:
                with st.spinner("Thinking..."):
                    # 1. Get text response from LLM
                    response, _ = asyncio.run(st.session_state.agent.query_documents(query, top_k=7))

                    # 2. Parse the *actual* source pages from the LLM's text response
                    cited_pages = parse_source_pages_from_response(response)

                    # 3. Get images ONLY from those cited pages
                    relevant_images = get_relevant_images_from_pages(cited_pages, st.session_state.extracted_images)

                    # 4. Add to history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response,
                        "images": relevant_images
                    })
                
                st.rerun()

    else:
        st.markdown("""
        ## 🚀 Welcome to PDF Intelligence Agent
        
        This advanced AI system can:
        
        - 📄 **Extract and analyze text** from PDF documents
        - 🖼️ **Extract and process images** with OCR
        - 🧠 **Understand context** using RAG (Retrieval Augmented Generation)
        - 💬 **Answer questions** about your documents intelligently
        
        ### How to use:
        1. Upload your PDF documents using the sidebar
        2. Click "Process Documents" to analyze them
        3. Ask questions about your documents in natural language
        """)

if __name__ == "__main__":
    if not os.getenv('GOOGLE_API_KEY'):
        st.error("❌ GOOGLE_API_KEY not found. Please ensure it's set in your .env file or Streamlit secrets.")
    else:
        main()