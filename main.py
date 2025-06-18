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

load_dotenv()  # Loads variables from .env into the environment

print(f"API Key from .env: {os.getenv('GOOGLE_API_KEY')}")

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
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
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
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Extract text and images
                text_chunks = await self.pdf_processor.extract_text_chunks(tmp_path)
                images = await self.pdf_processor.extract_images(tmp_path)
                
                # Process images with OCR
                for img_data in images:
                    ocr_text = await self.image_processor.extract_text_from_image(img_data['image'])
                    if ocr_text.strip():
                        img_data['ocr_text'] = ocr_text
                
                all_chunks.extend(text_chunks)
                all_images.extend(images)
                
            finally:
                os.unlink(tmp_path)
        
        # Create vector embeddings
        if progress_callback:
            progress_callback(0.8, "Creating vector embeddings...")
        
        await self.vector_store.add_documents(all_chunks)
        
        if progress_callback:
            progress_callback(1.0, "Processing complete!")
        
        return all_chunks, all_images
    
    async def query_documents(self, query, top_k=5):
        """Query processed documents"""
        # Get relevant chunks
        relevant_chunks = await self.vector_store.similarity_search(query, k=top_k)
        
        # Generate response using LLM
        response = await self.llm_handler.generate_response(query, relevant_chunks)
        
        return response, relevant_chunks

def main():
    st.markdown('<div class="main-header"><h1>🤖 PDF Intelligence Agent</h1><p>Advanced AI-powered document analysis with RAG and multi-modal processing</p></div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = PDFAgent()
    if 'processed_docs' not in st.session_state:
        st.session_state.processed_docs = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'extracted_images' not in st.session_state:
        st.session_state.extracted_images = []
    
    # Sidebar
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
                with st.spinner("Processing documents..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(progress, status):
                        progress_bar.progress(progress)
                        status_text.text(status)
                    
                    # Process documents
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
            
            # Document stats
            st.subheader("📊 Document Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Text Chunks", len(st.session_state.agent.vector_store.documents))
            with col2:
                st.metric("Images", len(st.session_state.extracted_images))
            
            # Download extracted images
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
        
        # Clear session
        if st.button("🗑️ Clear Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()
    
    # Main content area
    if st.session_state.processed_docs:
        # Query interface
        st.subheader("💬 Chat with Your Documents")
        
        # Chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message"><strong>AI:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        
        # Query input
        query = st.chat_input("Ask anything about your documents...")
        
        if query:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            with st.spinner("Thinking..."):
                # Query documents
                response, relevant_chunks = asyncio.run(
                    st.session_state.agent.query_documents(query)
                )
                
                # Add AI response to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            st.experimental_rerun()
        
        # Show extracted images
        if st.session_state.extracted_images:
            st.subheader("🖼️ Extracted Images")
            
            # Filter images
            search_term = st.text_input("Search images by OCR text...")
            
            filtered_images = st.session_state.extracted_images
            if search_term:
                filtered_images = [
                    img for img in st.session_state.extracted_images 
                    if search_term.lower() in img.get('ocr_text', '').lower()
                ]
            
            # Display images in grid
            cols = st.columns(3)
            for i, img_data in enumerate(filtered_images[:12]):  # Limit to 12 images
                with cols[i % 3]:
                    st.image(img_data['image'], caption=f"Page {img_data['page']}", use_column_width=True)
                    if img_data.get('ocr_text'):
                        with st.expander("OCR Text"):
                            st.text(img_data['ocr_text'][:200] + "..." if len(img_data['ocr_text']) > 200 else img_data['ocr_text'])
            
            if len(filtered_images) > 12:
                st.info(f"Showing first 12 of {len(filtered_images)} images. Use search to filter.")
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 Welcome to PDF Intelligence Agent
        
        This advanced AI system can:
        
        - 📄 **Extract and analyze text** from PDF documents
        - 🖼️ **Extract and process images** with OCR
        - 🧠 **Understand context** using RAG (Retrieval Augmented Generation)
        - 💬 **Answer questions** about your documents intelligently
        - 🔍 **Search through content** efficiently
        
        ### How to use:
        1. Upload your PDF documents using the sidebar
        2. Click "Process Documents" to analyze them
        3. Ask questions about your documents in natural language
        4. Download extracted images and data
        
        ### Features:
        - **Cost-effective**: Uses efficient algorithms and minimal API calls
        - **CPU optimized**: Works without GPU requirements
        - **Multi-modal**: Processes both text and images
        - **Agentic**: Intelligent reasoning and context understanding
        """)

if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv('GOOGLE_API_KEY'):
        st.error("❌ Please set GOOGLE_API_KEY in your .env file")
        st.stop()
    
    main()