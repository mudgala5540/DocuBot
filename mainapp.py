import streamlit as st
import os
import tempfile
import asyncio
import re
import threading
import time
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from llm_handler import LLMHandler
from image_processor import ImageProcessor

# Load environment variables from .env file
load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="IntelliDoc Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- World-Class Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Reset and Base Styles */
    .stApp {
        font-family: 'Inter', sans-serif !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        min-height: 100vh;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 3rem 0 2rem 0;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 400;
        margin-bottom: 0;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(20px) !important;
    }
    
    .sidebar-content {
        padding: 1rem;
    }
    
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #374151;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    /* File Upload Styling */
    .uploadedFile {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 0.75rem !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Process Button */
    .process-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .process-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Status Pills */
    .status-pill {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.25rem 0.25rem 0.25rem 0;
        border: 1px solid;
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.1);
        color: #059669;
        border-color: rgba(16, 185, 129, 0.3);
    }
    
    .status-processing {
        background: rgba(59, 130, 246, 0.1);
        color: #2563eb;
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.1);
        color: #dc2626;
        border-color: rgba(239, 68, 68, 0.3);
    }
    
    /* Chat Container */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        min-height: 600px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        overflow: hidden;
    }
    
    /* Welcome Message */
    .welcome-card {
        text-align: center;
        padding: 4rem 2rem;
        color: #64748b;
    }
    
    .welcome-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.7;
    }
    
    .welcome-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #374151;
    }
    
    .welcome-text {
        font-size: 1rem;
        line-height: 1.6;
        max-width: 500px;
        margin: 0 auto;
    }
    
    /* Progress Bar Styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 10px !important;
    }
    
    /* Success/Error Messages */
    .success-message {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-weight: 500;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .error-message {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-weight: 500;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
    }
    
    /* Chat Input */
    .stChatInput > div > div > textarea {
        border-radius: 25px !important;
        border: 2px solid #e5e7eb !important;
        padding: 1rem 1.5rem !important;
        font-size: 1rem !important;
        background: white !important;
    }
    
    .stChatInput > div > div > textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: transparent !important;
        padding: 1rem !important;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.1) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Sidebar Buttons */
    .sidebar-btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    .sidebar-btn-secondary {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
        color: white !important;
    }
    
    .sidebar-btn-danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        color: white !important;
    }
    
    /* Loading Animation */
    .loading-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        background: rgba(102, 126, 234, 0.05);
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .loading-spinner {
        width: 32px;
        height: 32px;
        border: 3px solid #f3f4f6;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 1rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-text {
        font-weight: 500;
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# --- ROBUST EVENT LOOP MANAGEMENT ---
class AsyncExecutor:
    """Thread-safe async executor for Streamlit"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._loop = None
            self._thread = None
            self._initialized = True
            self._start_loop()
    
    def _start_loop(self):
        """Start the event loop in a separate thread"""
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()
        
        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        
        # Wait for loop to be ready
        while self._loop is None:
            time.sleep(0.01)
    
    def run_async(self, coro):
        """Execute async function safely"""
        if self._loop is None or self._loop.is_closed():
            self._start_loop()
        
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

# Global executor instance
if 'async_executor' not in st.session_state:
    st.session_state.async_executor = AsyncExecutor()

# --- AGENT CLASS ---
class PDFAgent:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.llm_handler = LLMHandler()
        self.image_processor = ImageProcessor()

    async def process_documents(self, uploaded_files):
        """Process uploaded PDF documents"""
        all_chunks, all_images = [], []
        total_files = len(uploaded_files)
        
        for i, file in enumerate(uploaded_files):
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Extract text and images
                text_chunks = await self.pdf_processor.extract_text_chunks(tmp_path)
                images = await self.pdf_processor.extract_images(tmp_path)
                
                # Process images with OCR
                for img_data in images:
                    try:
                        ocr_text = await self.image_processor.extract_text_from_image(img_data['image'])
                        if ocr_text:
                            img_data['ocr_text'] = ocr_text.lower()
                    except:
                        pass  # Continue if OCR fails
                
                all_chunks.extend(text_chunks)
                all_images.extend(images)
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        # Create embeddings
        if all_chunks:
            await self.vector_store.add_documents(all_chunks)
        
        return all_chunks, all_images

    async def query_documents(self, query, top_k=8):
        """Query processed documents"""
        relevant_chunks = await self.vector_store.similarity_search(query, k=top_k)
        response = await self.llm_handler.generate_response(query, relevant_chunks)
        return response, relevant_chunks

# --- HELPER FUNCTIONS ---
def parse_source_pages(response_text: str) -> list[int]:
    """Extract page numbers from response"""
    match = re.search(r'\(Source: (.*?)\)', response_text, re.IGNORECASE)
    return [int(p) for p in re.findall(r'\d+', match.group(1))] if match else []

def find_relevant_images(cited_pages: list, all_images: list) -> list:
    """Find images from cited pages"""
    if not cited_pages:
        return []
    found_images = [img for img in all_images if img['page'] in cited_pages]
    found_images.sort(key=lambda x: (x['page'], x['index']))
    return found_images

# --- MAIN APPLICATION ---
def main():
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🤖 IntelliDoc Agent</div>
        <div class="hero-subtitle">Advanced AI-powered document analysis and intelligent Q&A</div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = PDFAgent()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'images' not in st.session_state:
        st.session_state.images = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        # Document Upload Section
        st.markdown('<div class="section-header">📁 Document Upload</div>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF documents for analysis"
        )

        # Show selected files
        if uploaded_files:
            st.markdown('<div class="section-header">📋 Selected Files</div>', unsafe_allow_html=True)
            for file in uploaded_files:
                file_size = len(file.getvalue()) / (1024 * 1024)  # MB
                st.markdown(f"""
                <div class="status-pill status-processing">
                    📄 {file.name} ({file_size:.1f} MB)
                </div>
                """, unsafe_allow_html=True)

            # Process button
            col1, col2 = st.columns([3, 1])
            with col1:
                process_clicked = st.button(
                    "🚀 Process Documents", 
                    type="primary", 
                    use_container_width=True,
                    disabled=st.session_state.processing
                )
            with col2:
                if st.button("❌", help="Clear selection"):
                    st.rerun()

            # Process documents
            if process_clicked and not st.session_state.processing:
                st.session_state.processing = True
                st.session_state.processed_files = [f.name for f in uploaded_files]
                st.session_state.messages = []
                st.session_state.images = []
                
                try:
                    # Show processing status
                    progress_placeholder = st.empty()
                    status_placeholder = st.empty()
                    
                    with progress_placeholder:
                        st.markdown("""
                        <div class="loading-container">
                            <div class="loading-spinner"></div>
                            <div class="loading-text">Processing documents...</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Process documents using the async executor
                    chunks, images = st.session_state.async_executor.run_async(
                        st.session_state.agent.process_documents(uploaded_files)
                    )
                    
                    st.session_state.images = images
                    
                    # Clear progress and show success
                    progress_placeholder.empty()
                    status_placeholder.markdown("""
                    <div class="success-message">
                        ✅ Documents processed successfully! Ready for questions.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add welcome message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "🎉 Perfect! I've successfully analyzed your documents. You can now ask me any questions about their content. I'll provide detailed answers with source citations and relevant images."
                    })
                    
                    st.session_state.processing = False
                    time.sleep(2)  # Show success message briefly
                    status_placeholder.empty()
                    st.rerun()
                    
                except Exception as e:
                    st.session_state.processing = False
                    st.markdown(f"""
                    <div class="error-message">
                        ❌ Error processing documents: {str(e)}
                    </div>
                    """, unsafe_allow_html=True)

        # Show processed files
        if st.session_state.processed_files:
            st.markdown('<div class="section-header">✅ Active Documents</div>', unsafe_allow_html=True)
            for file_name in st.session_state.processed_files:
                st.markdown(f"""
                <div class="status-pill status-success">
                    📄 {file_name}
                </div>
                """, unsafe_allow_html=True)

        # Control buttons
        st.markdown('<div class="section-header">⚙️ Controls</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🔄 Reset All", use_container_width=True):
                keys_to_keep = ['async_executor']
                for key in list(st.session_state.keys()):
                    if key not in keys_to_keep:
                        del st.session_state[key]
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # Main Chat Interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Welcome message when no chat history
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">📚</div>
            <div class="welcome-title">Welcome to IntelliDoc Agent!</div>
            <div class="welcome-text">
                Upload your PDF documents using the sidebar to get started.<br><br>
                Once processed, I can answer questions about your documents with precise citations and show relevant images from the pages I reference.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Display chat messages
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

            # Show sources
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources", expanded=False):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**📖 Source {i+1} (Page {source['page']})**")
                        st.markdown(f"```\n{source['text']}\n```")
                        st.markdown("---")

            # Show relevant images
            if "images" in message and message["images"]:
                st.markdown("### 🖼️ Relevant Images")
                cols = st.columns(min(3, len(message["images"])))
                for i, img in enumerate(message["images"]):
                    with cols[i % 3]:
                        st.image(
                            img['image'],
                            caption=f"Page {img['page']}",
                            use_container_width=True
                        )

    # Chat input
    if query := st.chat_input("💬 Ask me anything about your documents..."):
        if not st.session_state.processed_files:
            st.warning("⚠️ Please upload and process documents first.")
            st.stop()

        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🧠 Analyzing your question..."):
                try:
                    # Use the async executor for querying
                    response_text, sources = st.session_state.async_executor.run_async(
                        st.session_state.agent.query_documents(query)
                    )
                    
                    # Check for valid response
                    no_answer_phrases = ["i can only answer", "could not find an answer", "you're welcome", "hello!"]
                    is_valid_answer = not any(phrase in response_text.lower() for phrase in no_answer_phrases)
                    
                    # Find relevant images
                    images_to_display = []
                    if is_valid_answer:
                        cited_pages = parse_source_pages(response_text)
                        images_to_display = find_relevant_images(cited_pages, st.session_state.images)

                    # Create assistant message
                    assistant_message = {
                        "role": "assistant",
                        "content": response_text,
                        "sources": sources if is_valid_answer else [],
                        "images": images_to_display
                    }
                    
                    st.session_state.messages.append(assistant_message)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
                    st.markdown("""
                    <div class="error-message">
                        🔧 Please try asking your question again. If the issue persists, try clearing the chat and re-processing your documents.
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    if not os.getenv('GOOGLE_API_KEY'):
        st.error("❌ **GOOGLE_API_KEY not found.** Please set it in your environment variables or Streamlit Cloud secrets.")
        st.stop()
    else:
        main()