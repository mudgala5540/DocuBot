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

# --- Page Config (UI OVERHAUL) ---
st.set_page_config(
    page_title="IntelliDoc Agent",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THE DEFINITIVE FIX for the "Task attached to a different loop" Error ---
def get_or_create_eventloop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

def run_async(coro):
    """Run async coroutine in the current event loop safely"""
    loop = get_or_create_eventloop()
    if loop.is_running():
        # If loop is already running, we need to create a new task
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return loop.run_until_complete(coro)

# --- Agent Class (CRITICAL FIX APPLIED) ---
class PDFAgent:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.llm_handler = LLMHandler()
        self.image_processor = ImageProcessor()

    async def process_documents(self, uploaded_files, progress_bar):
        all_chunks, all_images = [], []
        total_files = len(uploaded_files)
        
        for i, file in enumerate(uploaded_files):
            progress_value = (i + 1) / total_files
            progress_bar.progress(progress_value, text=f"Processing file {i+1}/{total_files}: {file.name}")

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                text_chunks = await self.pdf_processor.extract_text_chunks(tmp_path)
                images = await self.pdf_processor.extract_images(tmp_path)
                
                # Enhanced OCR processing for better image relevance matching
                for img_data in images:
                    try:
                        ocr_text = await self.image_processor.extract_text_from_image(img_data['image'])
                        img_analysis = await self.image_processor.analyze_image_content(img_data['image'])
                        
                        # Store both OCR text and analysis
                        img_data['ocr_text'] = ocr_text.lower() if ocr_text else ""
                        img_data['analysis'] = img_analysis
                        img_data['has_meaningful_content'] = (
                            len(ocr_text.strip()) > 10 or  # Has substantial text
                            img_analysis.get('likely_chart_or_diagram', False) or  # Is a chart/diagram
                            img_analysis.get('likely_contains_text', False)  # Contains text elements
                        )
                    except Exception as e:
                        print(f"Error processing image on page {img_data['page']}: {e}")
                        img_data['ocr_text'] = ""
                        img_data['analysis'] = {}
                        img_data['has_meaningful_content'] = False
                
                all_chunks.extend(text_chunks)
                all_images.extend(images)
            finally:
                os.unlink(tmp_path)
        
        progress_bar.progress(1.0, text="Creating document embeddings...")
        if all_chunks:
            await self.vector_store.add_documents(all_chunks)
        
        return all_chunks, all_images

    async def query_documents(self, query, top_k=8):
        relevant_chunks = await self.vector_store.similarity_search(query, k=top_k)
        response = await self.llm_handler.generate_response(query, relevant_chunks)
        return response, relevant_chunks

# --- Enhanced Helper Functions ---
def parse_source_pages(response_text: str) -> list[int]:
    """Extract page numbers from source citations"""
    match = re.search(r'\(Source: (.*?)\)', response_text, re.IGNORECASE)
    if match:
        page_numbers = re.findall(r'\d+', match.group(1))
        return [int(p) for p in page_numbers]
    return []

def is_query_document_related(query: str, response_text: str) -> bool:
    """Determine if the query is document-related based on query and response"""
    non_document_phrases = [
        "hello", "hi", "thank", "how are you", "good morning", "good evening",
        "you're welcome", "goodbye", "bye"
    ]
    nonsense_indicators = [
        "random", "joke", "color of the sky", "weather", "asdf", "lorem ipsum"
    ]
    
    query_lower = query.lower().strip()
    response_lower = response_text.lower()
    
    # Check for greetings or casual conversation
    if any(phrase in query_lower for phrase in non_document_phrases):
        return False
    
    # Check for nonsense or irrelevant queries
    if any(indicator in query_lower for indicator in nonsense_indicators) or len(query_lower) < 5:
        return False
    
    # Check if response indicates document usage
    if "(source:" in response_lower or "based on the provided documents" in response_lower:
        return True
    
    # Check if response indicates no relevant information found but is still document-related
    if "could not find" in response_lower and "document" in response_lower:
        return True
    
    # Default to assuming document-related if query is substantial
    return len(query_lower.split()) > 2

def find_relevant_images_enhanced(query: str, cited_pages: list, all_images: list, response_text: str) -> list:
    """Enhanced image relevance matching with stricter criteria"""
    if not all_images or not cited_pages:
        return []
    
    query_words = set(query.lower().split())
    response_words = set(response_text.lower().split())
    combined_search_terms = query_words.union(response_words)
    
    # Remove stop words
    stop_words = {
        "the", "is", "at", "which", "on", "a", "an", "and", "or", "but", "in", 
        "with", "to", "for", "of", "as", "by", "from", "about", "what", "how", 
        "when", "where", "why", "who"
    }
    search_terms = combined_search_terms - stop_words
    
    relevant_images = []
    min_relevance_score = 2.0  # Stricter threshold for relevance
    
    for img in all_images:
        if img['page'] not in cited_pages or not img.get('has_meaningful_content', False):
            continue
        
        relevance_score = 0.0
        
        # OCR text relevance (weighted heavily)
        if img.get('ocr_text'):
            img_words = set(img['ocr_text'].lower().split())
            text_overlap = len(search_terms.intersection(img_words))
            relevance_score += text_overlap * 3.0  # Increased weight for text matches
        
        # Chart/diagram relevance
        if img.get('analysis', {}).get('likely_chart_or_diagram', False):
            data_keywords = {
                "chart", "graph", "data", "statistics", "figure", "diagram", 
                "plot", "table", "number", "percentage", "rate", "analysis", 
                "trend", "comparison"
            }
            if search_terms.intersection(data_keywords):
                relevance_score += 4.0  # Higher weight for charts/diagrams
        
        # Text-containing image relevance
        if img.get('analysis', {}).get('likely_contains_text', False):
            relevance_score += 1.5
        
        # Image quality boost (high-quality images are more likely relevant)
        if img.get('analysis', {}).get('image_quality') == 'high':
            relevance_score += 1.0
        
        # Only include images above the minimum relevance score
        if relevance_score >= min_relevance_score:
            img_copy = img.copy()
            img_copy['relevance_score'] = relevance_score
            relevant_images.append(img_copy)
    
    # Sort by relevance score (descending) and then by page number
    relevant_images.sort(key=lambda x: (-x['relevance_score'], x['page'], x['index']))
    
    return relevant_images

# --- Streamlit UI and Application Flow ---
def main():
    st.markdown("""
        <style>
            .stApp {
                background-color: #F0F2F6;
            }
            .st-emotion-cache-16txtl3 {
                padding: 1rem 1rem 1rem;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: #1E1E1E;'>✨ IntelliDoc Agent</h1>", unsafe_allow_html=True)

    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = PDFAgent()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "images" not in st.session_state:
        st.session_state.images = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Controls")
        uploaded_files = st.file_uploader(
            "Upload PDF documents", 
            type=['pdf'], 
            accept_multiple_files=True
        )

        if uploaded_files and st.button("🚀 Process Documents", type="primary", use_container_width=True):
            st.session_state.processed_files = [f.name for f in uploaded_files]
            st.session_state.messages = []
            st.session_state.images = []
            
            progress_bar_placeholder = st.empty()
            progress_bar = progress_bar_placeholder.progress(0)
            
            try:
                _, images = run_async(st.session_state.agent.process_documents(uploaded_files, progress_bar))
                st.session_state.images = images
                progress_bar_placeholder.empty()
                st.success("Documents processed successfully!")
                st.session_state.messages.append({"role": "assistant", "content": "✅ Documents are ready. Feel free to ask any questions."})
            except Exception as e:
                progress_bar_placeholder.empty()
                st.error(f"Error processing documents: {e}")
            st.rerun()

        if st.session_state.processed_files:
            st.markdown("---")
            st.subheader("✅ Processed Files")
            for file_name in st.session_state.processed_files:
                st.info(f"📄 {file_name}")
        
        st.markdown("---")
        if st.button("🗑️ Clear Session", use_container_width=True):
            for key in list(st.session_state.keys()): 
                del st.session_state[key]
            st.rerun()

    # Main chat area
    with st.container(border=True):
        if not st.session_state.messages and not st.session_state.processed_files:
             st.info("Welcome! Please upload your documents using the sidebar to get started.")
        elif not st.session_state.messages and st.session_state.processed_files:
             st.info("Documents processed. Ask a question to begin the conversation.")

        for message in st.session_state.messages:
            avatar = "👤" if message["role"] == "user" else "✨"
            with st.chat_message(message["role"], avatar=avatar):
                if "error" in message:
                    st.error(message["content"])
                else:
                    st.markdown(message["content"])

                # Show sources only for document-related queries
                if "sources" in message and message["sources"] and message.get("is_document_related", True):
                    with st.expander("Show Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.info(f"**Source {i+1} (Page {source['page']})**\n\n---\n\n" + source['text'])

                # Show images only for document-related queries
                if "images" in message and message["images"]:
                    st.markdown("**Relevant Images:**")
                    num_images = len(message["images"])
                    if num_images <= 3:
                        cols = st.columns(num_images)
                    else:
                        for i in range(0, num_images, 3):
                            cols = st.columns(min(3, num_images - i))
                            for j, img in enumerate(message["images"][i:i+3]):
                                with cols[j]:
                                    st.image(
                                        img['image'], 
                                        caption=f"Page {img['page']} (Relevance: {img.get('relevance_score', 0):.1f})", 
                                        use_container_width=True
                                    )

    # Chat input
    if query := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": query})
        
        try:
            if st.session_state.processed_files:
                response_text, sources = run_async(st.session_state.agent.query_documents(query))
            else:
                simple_greetings = ["hello", "hi", "hey", "good morning", "good evening", "thank you", "thanks"]
                query_lower = query.lower().strip()
                if any(greeting in query_lower for greeting in simple_greetings):
                    response_text = "Hello! Please upload your documents using the sidebar so I can help answer questions about them."
                    sources = []
                else:
                    response_text = "I need documents to be uploaded first before I can answer questions about them. Please use the sidebar to upload your PDF documents."
                    sources = []
            
            # Determine if this is a document-related query
            is_doc_related = is_query_document_related(query, response_text)
            
            # Find relevant images only for document-related queries
            images_to_display = []
            if is_doc_related and st.session_state.images:
                cited_pages = parse_source_pages(response_text)
                images_to_display = find_relevant_images_enhanced(
                    query, cited_pages, st.session_state.images, response_text
                )

            assistant_message = {
                "role": "assistant", 
                "content": response_text,
                "sources": sources if is_doc_related else [],
                "images": images_to_display,
                "is_document_related": is_doc_related
            }
            
        except Exception as e:
            st.error(f"A critical error occurred: {e}")
            assistant_message = {
                "role": "assistant",
                "content": f"I'm sorry, I encountered an error: {e}",
                "error": True
            }
            
        st.session_state.messages.append(assistant_message)
        st.rerun()

if __name__ == "__main__":
    if not os.getenv('GOOGLE_API_KEY'):
        st.error("❌ **GOOGLE_API_KEY not found.** Please set it in your environment variables or Streamlit Cloud secrets.")
    else:
        main()