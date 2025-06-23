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
                
                # Enhanced OCR processing for images
                for img_data in images:
                    ocr_text = await self.image_processor.extract_text_from_image(img_data['image'])
                    if ocr_text.strip():
                        img_data['ocr_text'] = ocr_text.lower()
                        img_data['has_text'] = True
                    else:
                        img_data['has_text'] = False
                    
                    # Analyze image content for better relevance detection
                    analysis = await self.image_processor.analyze_image_content(img_data['image'])
                    img_data['analysis'] = analysis
                
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
    """Enhanced source page parsing with better regex patterns"""
    # Multiple patterns to catch different citation formats
    patterns = [
        r'\(Source:\s*Page\s*([\d,\s]+)\)',  # (Source: Page 1, 2, 3)
        r'\(Source:\s*([\d,\s]+)\)',        # (Source: 1, 2, 3)
        r'Source:\s*Page\s*([\d,\s]+)',     # Source: Page 1, 2, 3
        r'Source:\s*([\d,\s]+)',            # Source: 1, 2, 3
        r'\(Pages?\s*([\d,\s]+)\)',         # (Page 1, 2, 3) or (Pages 1, 2, 3)
        r'Pages?\s*([\d,\s]+)',             # Page 1, 2, 3 or Pages 1, 2, 3
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            # Extract all numbers from the matched string
            page_numbers = re.findall(r'\d+', match.group(1))
            try:
                return [int(p) for p in page_numbers]
            except ValueError:
                continue
    
    return []

def is_query_document_related(query: str, response: str) -> bool:
    """Determine if the query is document-related based on response patterns"""
    casual_phrases = [
        "you're welcome", "hello!", "hi there", "how can i help",
        "thank you", "thanks", "good morning", "good afternoon", 
        "good evening", "bye", "goodbye", "see you"
    ]
    
    irrelevant_phrases = [
        "i can only answer", "designed to answer questions about",
        "please ask something related", "unrelated to documents",
        "could not find an answer", "not related to the documents"
    ]
    
    response_lower = response.lower()
    
    # Check if it's a casual/greeting response
    if any(phrase in response_lower for phrase in casual_phrases):
        return False
    
    # Check if it's an irrelevant query response
    if any(phrase in response_lower for phrase in irrelevant_phrases):
        return False
    
    # Check for error responses
    if "error generating response" in response_lower:
        return False
    
    return True

def find_relevant_images_enhanced(cited_pages: list, all_images: list, query: str = "") -> list:
    """Enhanced image relevance detection"""
    if not cited_pages:
        return []
    
    # First, get all images from cited pages
    page_images = [img for img in all_images if img['page'] in cited_pages]
    
    if not page_images:
        return []
    
    # If query contains visual keywords, prioritize images with text or high edge density
    visual_keywords = [
        'chart', 'graph', 'table', 'diagram', 'image', 'figure', 'plot',
        'visual', 'picture', 'illustration', 'screenshot', 'map', 'drawing'
    ]
    
    query_lower = query.lower()
    has_visual_keywords = any(keyword in query_lower for keyword in visual_keywords)
    
    relevant_images = []
    
    for img in page_images:
        # Basic relevance: image is on a cited page
        relevance_score = 1.0
        
        # Boost score if image has text content and query might be asking about text
        if img.get('has_text', False):
            relevance_score += 0.5
            
            # Check if query words match OCR text
            if 'ocr_text' in img and query_lower:
                query_words = set(query_lower.split())
                ocr_words = set(img['ocr_text'].split())
                word_overlap = len(query_words.intersection(ocr_words))
                if word_overlap > 0:
                    relevance_score += word_overlap * 0.3
        
        # Boost score for images that look like charts/diagrams if query has visual keywords
        if has_visual_keywords and img.get('analysis', {}).get('likely_chart_or_diagram', False):
            relevance_score += 0.7
        
        # Boost score for high-quality images
        image_quality = img.get('analysis', {}).get('image_quality', 'low')
        if image_quality == 'high':
            relevance_score += 0.3
        elif image_quality == 'medium':
            relevance_score += 0.1
        
        # Only include images with sufficient relevance
        if relevance_score >= 1.0:  # At least on cited page
            img['relevance_score'] = relevance_score
            relevant_images.append(img)
    
    # Sort by relevance score and page number
    relevant_images.sort(key=lambda x: (-x['relevance_score'], x['page'], x.get('index', 0)))
    
    return relevant_images

# --- Streamlit UI and Application Flow ---
def main():
    # Custom CSS for a more polished look
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
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "✅ Documents are ready. Feel free to ask any questions about their content."
                })
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

                if "sources" in message and message["sources"]:
                    with st.expander("📋 Show Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.info(f"**Source {i+1} (Page {source['page']}, Similarity: {source.get('similarity_score', 0):.3f})**\n\n---\n\n" + source['text'][:500] + "..." if len(source['text']) > 500 else source['text'])

                if "images" in message and message["images"]:
                    st.markdown("**📸 Relevant Images:**")
                    
                    # Display images in a grid layout
                    num_images = len(message["images"])
                    cols_per_row = min(3, num_images)
                    
                    for i in range(0, num_images, cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, img in enumerate(message["images"][i:i+cols_per_row]):
                            with cols[j]:
                                st.image(
                                    img['image'], 
                                    caption=f"Page {img['page']} (Relevance: {img.get('relevance_score', 1.0):.2f})",
                                    use_container_width=True
                                )
                                
                                # Show OCR text if available
                                if img.get('has_text', False) and img.get('ocr_text'):
                                    with st.expander(f"📝 Text in Image (Page {img['page']})"):
                                        st.text(img['ocr_text'][:200] + "..." if len(img['ocr_text']) > 200 else img['ocr_text'])

    # Chat input
    if query := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.processed_files:
            st.warning("Please upload and process at least one document first.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": query})
        
        try:
            # Get response from the agent
            response_text, sources = run_async(st.session_state.agent.query_documents(query))
            
            # Enhanced document-related query detection
            is_document_query = is_query_document_related(query, response_text)
            
            images_to_display = []
            sources_to_display = []
            
            if is_document_query:
                # Parse cited pages from response
                cited_pages = parse_source_pages(response_text)
                
                if cited_pages:
                    # Find relevant images using enhanced algorithm
                    images_to_display = find_relevant_images_enhanced(cited_pages, st.session_state.images, query)
                    sources_to_display = sources
            
            assistant_message = {
                "role": "assistant", 
                "content": response_text,
                "sources": sources_to_display,
                "images": images_to_display
            }
            
        except Exception as e:
            st.error(f"A critical error occurred: {e}")
            assistant_message = {
                "role": "assistant",
                "content": f"I'm sorry, I encountered a critical error trying to answer your question. The error was: {e}",
                "error": True
            }
            
        st.session_state.messages.append(assistant_message)
        st.rerun()

if __name__ == "__main__":
    if not os.getenv('GOOGLE_API_KEY'):
        st.error("❌ **GOOGLE_API_KEY not found.** Please set it in your environment variables or Streamlit Cloud secrets.")
    else:
        main()