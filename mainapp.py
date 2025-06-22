import streamlit as st
import os
import tempfile
import asyncio
import re
import threading
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from llm_handler import LLMHandler
from image_processor import ImageProcessor

# Load environment variables from .env file
load_dotenv()

# --- Page Config with Premium UI ---
st.set_page_config(
    page_title="IntelliDoc Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for World-Class UI ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main Container */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 25px 45px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Header Styles */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Sidebar Enhancements */
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar-header {
        color: white;
        font-size: 1.5rem;
        font-weight: 600;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Chat Container */
    .chat-container {
        background: white;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid #E5E7EB;
        min-height: 500px;
        margin: 1rem 0;
    }
    
    /* Progress Bar Enhancements */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Success/Error Messages */
    .success-message {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 500;
        margin: 1rem 0;
    }
    
    .error-message {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 500;
        margin: 1rem 0;
    }
    
    /* File Upload Area */
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: #764ba2;
    }
    
    /* Processing Animation */
    .processing-animation {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .spinner {
        width: 40px;
        height: 40px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 1rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Chat Messages */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .assistant-message {
        background: #F9FAFB;
        color: #374151;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Source Cards */
    .source-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* Image Gallery */
    .image-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .image-card {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .image-card:hover {
        transform: scale(1.05);
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 500;
        font-size: 0.9rem;
        margin: 0.25rem;
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.1);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .status-processing {
        background: rgba(102, 126, 234, 0.1);
        color: #667eea;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.1);
        color: #DC2626;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- DEFINITIVE EVENT LOOP FIX ---
class AsyncRunner:
    """Handles asyncio operations in Streamlit with proper event loop management"""
    
    def __init__(self):
        self._loop = None
        self._thread = None
    
    def _run_loop(self):
        """Run event loop in a separate thread"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
    
    def get_loop(self):
        """Get or create the event loop"""
        if self._loop is None or self._loop.is_closed():
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run_loop, daemon=True)
                self._thread.start()
                # Wait for loop to be ready
                while self._loop is None:
                    pass
        return self._loop
    
    def run_async(self, coro):
        """Run async function safely"""
        loop = self.get_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

# Global async runner instance
if 'async_runner' not in st.session_state:
    st.session_state.async_runner = AsyncRunner()

# --- Agent Class ---
class PDFAgent:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.llm_handler = LLMHandler()
        self.image_processor = ImageProcessor()

    async def process_documents(self, uploaded_files, progress_callback=None):
        all_chunks, all_images = [], []
        total_files = len(uploaded_files)
        
        for i, file in enumerate(uploaded_files):
            if progress_callback:
                progress_callback(i, total_files, f"Processing {file.name}")

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                text_chunks = await self.pdf_processor.extract_text_chunks(tmp_path)
                images = await self.pdf_processor.extract_images(tmp_path)
                for img_data in images:
                    ocr_text = await self.image_processor.extract_text_from_image(img_data['image'])
                    if ocr_text: 
                        img_data['ocr_text'] = ocr_text.lower()
                all_chunks.extend(text_chunks)
                all_images.extend(images)
            finally:
                os.unlink(tmp_path)
        
        if progress_callback:
            progress_callback(total_files, total_files, "Creating embeddings...")
        
        if all_chunks:
            await self.vector_store.add_documents(all_chunks)
        
        return all_chunks, all_images

    async def query_documents(self, query, top_k=8):
        relevant_chunks = await self.vector_store.similarity_search(query, k=top_k)
        response = await self.llm_handler.generate_response(query, relevant_chunks)
        return response, relevant_chunks

# --- Helper Functions ---
def parse_source_pages(response_text: str) -> list[int]:
    match = re.search(r'\(Source: (.*?)\)', response_text, re.IGNORECASE)
    return [int(p) for p in re.findall(r'\d+', match.group(1))] if match else []

def find_relevant_images(cited_pages: list, all_images: list) -> list:
    if not cited_pages: 
        return []
    found_images = [img for img in all_images if img['page'] in cited_pages]
    found_images.sort(key=lambda x: (x['page'], x['index']))
    return found_images

def display_processing_animation(message="Processing..."):
    """Display a beautiful processing animation"""
    return st.markdown(f"""
    <div class="processing-animation">
        <div class="spinner"></div>
        <span style="font-size: 1.1rem; font-weight: 500; color: #667eea;">{message}</span>
    </div>
    """, unsafe_allow_html=True)

# --- Main Application ---
def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <div class="hero-title">🤖 IntelliDoc Agent</div>
        <div class="hero-subtitle">Your AI-powered assistant for intelligent document analysis</div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = PDFAgent()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "images" not in st.session_state:
        st.session_state.images = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">⚙️ Control Panel</div>', unsafe_allow_html=True)
        
        # File Upload Section
        st.markdown("### 📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF documents to analyze"
        )

        if uploaded_files:
            st.markdown("### 📋 Selected Files")
            for file in uploaded_files:
                file_size = len(file.getvalue()) / (1024 * 1024)  # MB
                st.markdown(f"""
                <div class="status-indicator status-processing">
                    📄 {file.name} ({file_size:.1f} MB)
                </div>
                """, unsafe_allow_html=True)

            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                # Clear previous session
                st.session_state.processed_files = [f.name for f in uploaded_files]
                st.session_state.messages = []
                st.session_state.images = []
                
                # Processing with beautiful UI
                progress_container = st.container()
                
                def progress_callback(current, total, message):
                    with progress_container:
                        progress = current / total if total > 0 else 0
                        st.progress(progress, text=f"{message} ({current}/{total})")
                
                with st.spinner("🔄 Processing documents..."):
                    try:
                        # Use the safe async runner
                        _, images = st.session_state.async_runner.run_async(
                            st.session_state.agent.process_documents(uploaded_files, progress_callback)
                        )
                        st.session_state.images = images
                        
                        st.markdown("""
                        <div class="success-message">
                            ✅ Documents processed successfully! You can now chat with your documents.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": "🎉 Perfect! I've successfully processed your documents. I'm ready to answer any questions you have about their content. What would you like to know?"
                        })
                        st.rerun()
                        
                    except Exception as e:
                        st.markdown(f"""
                        <div class="error-message">
                            ❌ Error processing documents: {str(e)}
                        </div>
                        """, unsafe_allow_html=True)

        # Processed Files Display
        if st.session_state.processed_files:
            st.markdown("---")
            st.markdown("### ✅ Active Documents")
            for file_name in st.session_state.processed_files:
                st.markdown(f"""
                <div class="status-indicator status-success">
                    📄 {file_name}
                </div>
                """, unsafe_allow_html=True)

        # Session Controls
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🔄 Reset All", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key != 'async_runner':  # Preserve the async runner
                        del st.session_state[key]
                st.rerun()

    # Main Chat Interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Welcome Message
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #6B7280;">
            <h3>👋 Welcome to IntelliDoc Agent!</h3>
            <p>Upload your PDF documents using the sidebar to get started.</p>
            <p>Once processed, you can ask me anything about your documents and I'll provide detailed answers with source citations.</p>
        </div>
        """, unsafe_allow_html=True)

    # Chat Messages
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

            # Sources
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources", expanded=False):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"""
                        <div class="source-card">
                            <h4>📖 Source {i+1} (Page {source['page']})</h4>
                            <p>{source['text']}</p>
                        </div>
                        """, unsafe_allow_html=True)

            # Images
            if "images" in message and message["images"]:
                st.markdown("### 🖼️ Relevant Images")
                cols = st.columns(min(3, len(message["images"])))
                for i, img in enumerate(message["images"]):
                    with cols[i % 3]:
                        st.image(
                            img['image'], 
                            caption=f"📄 Page {img['page']}", 
                            use_container_width=True
                        )

    # Chat Input
    if query := st.chat_input("💬 Ask me anything about your documents...", key="chat_input"):
        if not st.session_state.processed_files:
            st.warning("⚠️ Please upload and process documents first before asking questions.")
            st.stop()

        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🧠 Analyzing your question..."):
                try:
                    # Use the safe async runner
                    response_text, sources = st.session_state.async_runner.run_async(
                        st.session_state.agent.query_documents(query)
                    )
                    
                    # Check for valid answer
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
                        🔧 If this error persists, try clearing the chat and asking your question again.
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    if not os.getenv('GOOGLE_API_KEY'):
        st.error("❌ **GOOGLE_API_KEY not found.** Please set it in your environment variables or Streamlit Cloud secrets.")
        st.stop()
    else:
        main()