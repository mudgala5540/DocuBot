import streamlit as st
import os
import tempfile
import asyncio
import re
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from llm_handler import LLMHandler
from image_processor import ImageProcessor
import nest_asyncio

# Apply the patch to allow nested asyncio event loops. This MUST be at the top.
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

# --- Page Config (UI Enhancements) ---
st.set_page_config(
    page_title="Document Intelligence Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THE DEFINITIVE FIX for the "Task attached to a different loop" Error ---
# This helper function ensures we are always using the same, valid event loop
# across all Streamlit reruns. This is the key to stability.
def run_async(coro):
    """
    Runs an asyncio coroutine in a way that is compatible with Streamlit's
    rerun-based execution model.
    """
    # Get the current running event loop. If there is none, it creates one.
    # nest_asyncio ensures that we can "run" a loop that is already running.
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


# --- Agent Class (Modified to return sources for the UI) ---
class PDFAgent:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.llm_handler = LLMHandler()
        self.image_processor = ImageProcessor()

    async def process_documents(self, uploaded_files):
        all_chunks, all_images = [], []
        
        # UI ENHANCEMENT: Use st.status for a cleaner processing log
        with st.status("Processing documents...", expanded=True) as status:
            for i, file in enumerate(uploaded_files):
                st.write(f"➡️ Processing: {file.name}")
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    st.write("📄 Extracting text chunks...")
                    text_chunks = await self.pdf_processor.extract_text_chunks(tmp_path)
                    st.write("🖼️ Extracting images...")
                    images = await self.pdf_processor.extract_images(tmp_path)
                    st.write("🔍 Performing OCR on images...")
                    for img_data in images:
                        ocr_text = await self.image_processor.extract_text_from_image(img_data['image'])
                        if ocr_text: img_data['ocr_text'] = ocr_text.lower()
                    
                    all_chunks.extend(text_chunks)
                    all_images.extend(images)
                finally:
                    os.unlink(tmp_path)
            
            st.write("🧠 Creating vector embeddings...")
            if all_chunks:
                await self.vector_store.add_documents(all_chunks)
            
            status.update(label="✅ Processing complete!", state="complete", expanded=False)
        
        return all_chunks, all_images

    async def query_documents(self, query, top_k=8):
        """
        UI ENHANCEMENT: Modified to return source chunks for transparency.
        """
        relevant_chunks = await self.vector_store.similarity_search(query, k=top_k)
        response = await self.llm_handler.generate_response(query, relevant_chunks)
        return response, relevant_chunks

# --- Helper Functions (No changes needed) ---
def parse_source_pages(response_text: str) -> list[int]:
    match = re.search(r'\(Source: (.*?)\)', response_text, re.IGNORECASE)
    return [int(p) for p in re.findall(r'\d+', match.group(1))] if match else []

def find_relevant_images(cited_pages: list, all_images: list) -> list:
    if not cited_pages: return []
    found_images = [img for img in all_images if img['page'] in cited_pages]
    found_images.sort(key=lambda x: (x['page'], x['index']))
    return found_images

# --- Streamlit UI and Application Flow ---
def main():
    st.markdown("<h1><center>🤖 Document Intelligence Agent</center></h1>", unsafe_allow_html=True)

    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = PDFAgent()
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Welcome! Please upload PDF documents to begin."}]
    if "images" not in st.session_state:
        st.session_state.images = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    # UI ENHANCEMENT: Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Controls")
        uploaded_files = st.file_uploader(
            "Upload your PDF documents", 
            type=['pdf'], 
            accept_multiple_files=True,
            help="Upload one or more PDFs for analysis."
        )

        if uploaded_files and st.button("🚀 Process Documents", type="primary", use_container_width=True):
            st.session_state.processed_files = [f.name for f in uploaded_files]
            st.session_state.messages = [{"role": "assistant", "content": f"Processing {len(uploaded_files)} documents..."}]
            st.session_state.images = []
            
            # **CRITICAL FIX IN ACTION**: Use the robust async runner
            _, images = run_async(st.session_state.agent.process_documents(uploaded_files))
            st.session_state.images = images
            st.session_state.messages.append({"role": "assistant", "content": "✅ Documents processed. Ask me anything!"})
            st.rerun()

        if st.session_state.processed_files:
            st.markdown("---")
            st.subheader("✅ Processed Files")
            for file_name in st.session_state.processed_files:
                st.info(f"📄 {file_name}")
        
        st.markdown("---")
        if st.button("🗑️ Clear Session", use_container_width=True):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

    # UI ENHANCEMENT: Main chat interface
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

            # UI ENHANCEMENT: Display source chunks for transparency
            if "sources" in message and message["sources"]:
                with st.expander("Show Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.info(f"**Source {i+1} (Page {source['page']})**\n\n---\n\n" + source['text'])

            # UI ENHANCEMENT: Display relevant images
            if "images" in message and message["images"]:
                st.markdown("**Relevant Images:**")
                cols = st.columns(min(3, len(message["images"])))
                for i, img in enumerate(message["images"]):
                    with cols[i % 3]:
                        st.image(img['image'], caption=f"Page {img['page']}", use_container_width=True)

    # Handle new user input
    if query := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.processed_files:
            st.warning("Please upload and process at least one document before asking questions.")
        else:
            st.session_state.messages.append({"role": "user", "content": query})
            
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    # **CRITICAL FIX IN ACTION**: Use the robust async runner
                    response_text, sources = run_async(st.session_state.agent.query_documents(query))
                    
                    no_answer_phrases = ["i can only answer", "could not find an answer", "you're welcome", "hello!"]
                    is_valid_answer = not any(phrase in response_text.lower() for phrase in no_answer_phrases)
                    
                    images_to_display = []
                    if is_valid_answer:
                        cited_pages = parse_source_pages(response_text)
                        images_to_display = find_relevant_images(cited_pages, st.session_state.images)

                    assistant_message = {
                        "role": "assistant", 
                        "content": response_text,
                        "sources": sources if is_valid_answer else [],
                        "images": images_to_display
                    }
                    st.session_state.messages.append(assistant_message)
                    st.rerun()

if __name__ == "__main__":
    if not os.getenv('GOOGLE_API_KEY'):
        st.error("❌ **GOOGLE_API_KEY not found.** Please set it in your environment variables or Streamlit Cloud secrets.")
    else:
        main()