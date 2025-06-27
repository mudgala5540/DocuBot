import streamlit as st
import os
import tempfile
import asyncio
import re
import pickle
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from llm_handler import LLMHandler
from image_processor import ImageProcessor
import nest_asyncio
import logging
import hashlib

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nest_asyncio.apply()
load_dotenv()

st.set_page_config(
    page_title="IntelliDoc Agent",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    """Run an async coroutine in a synchronous context."""
    loop = get_or_create_eventloop()
    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return loop.run_until_complete(coro)

def sanitize_response(response: str) -> str:
    """Remove internal processing messages from response."""
    unwanted_phrases = [
        r"STEP \d+:", r"thinking", r"processing", r"query classification",
        r"internal error", r"debug:", r"agentic prompt", r"Thought:"
    ]
    for phrase in unwanted_phrases:
        response = re.sub(phrase, "", response, flags=re.IGNORECASE)
    return response.strip()

class PDFAgent:
    def __init__(self, cache_dir: str = ".cache"):
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.llm_handler = LLMHandler()
        self.image_processor = ImageProcessor()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.vector_cache_file = os.path.join(cache_dir, "vector_store.pkl")
        self.summary_cache_file = os.path.join(cache_dir, "document_summary.pkl")

    async def process_documents(self, uploaded_files, progress_bar):
        """Process uploaded PDF files asynchronously."""
        if not uploaded_files:
            logger.warning("No files uploaded for processing")
            st.error("No files uploaded. Please select at least one PDF file.")
            return [], []
        
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
                
                for img_data in images:
                    try:
                        ocr_text = await self.image_processor.extract_text_from_image(img_data['image'])
                        img_analysis = await self.image_processor.analyze_image_content(img_data['image'])
                        tables = await self.image_processor.extract_tables_from_image(img_data['image'])
                        
                        img_data['ocr_text'] = ocr_text.lower() if ocr_text else ""
                        img_data['analysis'] = img_analysis
                        img_data['tables'] = tables
                        img_data['has_meaningful_content'] = (
                            len(ocr_text.strip()) > 5 or
                            img_analysis.get('likely_chart_or_diagram', False) or
                            img_analysis.get('likely_contains_text', False) or
                            len(tables) > 0 or
                            img_analysis.get('image_quality') in ['high', 'medium']
                        )
                    except Exception as e:
                        logger.error(f"Error processing image on page {img_data['page']}: {e}")
                        img_data['ocr_text'] = ""
                        img_data['analysis'] = {}
                        img_data['tables'] = []
                        img_data['has_meaningful_content'] = False
                
                all_chunks.extend(text_chunks)
                all_images.extend(images)
            except Exception as e:
                logger.error(f"Error processing file {file.name}: {e}")
                st.warning(f"Failed to process {file.name}: {str(e)}")
            finally:
                os.unlink(tmp_path)
        
        if not all_chunks:
            logger.warning("No text chunks extracted from uploaded files")
            st.error("No text content extracted from the uploaded files. Please check the documents.")
            return [], all_images
        
        progress_bar.progress(1.0, text="Creating document embeddings and summary...")
        try:
            await self.vector_store.add_documents(all_chunks)
            self.vector_store.save_index(self.vector_cache_file)
            summary = await self.llm_handler.summarize_document(all_chunks, all_images)
            with open(self.summary_cache_file, 'wb') as f:
                pickle.dump(summary, f)
        except Exception as e:
            logger.error(f"Error creating embeddings or summary: {e}")
            st.error(f"Error generating embeddings or summary: {str(e)}")
        
        return all_chunks, all_images

    async def query_documents(self, query, top_k=15):
        """Query documents asynchronously."""
        try:
            relevant_chunks = await self.vector_store.hybrid_search(query, k=top_k)
            response = await self.llm_handler.generate_response(query, relevant_chunks)
            return response, relevant_chunks
        except Exception as e:
            logger.error(f"Error querying documents: {e}")
            return self.llm_handler.sanitize_response(f"Error querying documents: {str(e)}"), []

def parse_source_pages(response_text: str) -> list[int]:
    """Parse page numbers from response text."""
    pages = []
    match = re.search(r'\(Source: (.*?)\)', response_text, re.IGNORECASE)
    if match:
        page_numbers = re.findall(r'(?:Page\s+)?(\d+)', match.group(1), re.IGNORECASE)
        pages.extend([int(p) for p in page_numbers])
    page_refs = re.findall(r'(?:page|pg)\s+(\d+)', response_text, re.IGNORECASE)
    pages.extend([int(p) for p in page_refs])
    return list(set(pages))

def is_query_document_related(query: str, response_text: str) -> bool:
    """Determine if the query is document-related."""
    casual_phrases = [
        "hello", "hi", "hey", "thank", "thanks", "thank you", "you're welcome",
        "how are you", "good morning", "good afternoon", "good evening", 
        "goodbye", "bye", "see you", "nice to meet", "pleasure"
    ]
    irrelevant_phrases = [
        "asdf", "qwerty", "lorem ipsum", "test test", "xyz", "abc",
        "color of the sky", "weather today", "tell me a joke", "what's funny",
        "random question", "how's the weather"
    ]
    
    query_lower = query.lower().strip()
    response_lower = response_text.lower()
    
    if any(phrase in query_lower for phrase in casual_phrases):
        return False
    if (any(phrase in query_lower for phrase in irrelevant_phrases) or 
        len(query_lower) < 3 or 
        query_lower.count(query_lower[0]) > len(query_lower) * 0.7):
        return False
    
    document_indicators = [
        "(source:", "based on the provided documents", "from the document",
        "according to the document", "the document shows", "as mentioned in",
        "could not find", "page"
    ]
    
    return any(indicator in response_lower for indicator in document_indicators) or len(query_lower.split()) > 2

def find_relevant_images_enhanced(query: str, cited_pages: list, all_images: list, response_text: str, source_chunks: list = None) -> list:
    """Find relevant images based on query and response."""
    if not all_images:
        return []
    
    if not cited_pages and source_chunks:
        cited_pages = list(set([chunk.get('page', 0) for chunk in source_chunks if chunk.get('page')]))
    
    include_all_pages = not cited_pages
    
    query_words = set(query.lower().split())
    response_words = set(response_text.lower().split())
    combined_search_terms = query_words.union(response_words)
    
    stop_words = {
        "the", "is", "at", "which", "on", "a", "an", "and", "or", "but", "in", 
        "with", "to", "for", "of", "as", "by", "from", "about", "this", "that",
        "they", "them", "their", "there", "then", "than", "these", "those"
    }
    search_terms = combined_search_terms - stop_words
    
    relevant_images = []
    
    for img in all_images:
        if not include_all_pages and img['page'] not in cited_pages:
            continue
        
        relevance_score = 0.0
        
        if not include_all_pages and img['page'] in cited_pages:
            relevance_score += 5.0
        elif include_all_pages:
            relevance_score += 0.7
        
        if img.get('ocr_text'):
            img_words = set(img['ocr_text'].lower().split())
            text_overlap = len(search_terms.intersection(img_words))
            relevance_score += text_overlap * 4.0
            for term in search_terms:
                if len(term) > 3 and term in img['ocr_text']:
                    relevance_score += 3.0
        
        if img.get('tables'):
            relevance_score += len(img['tables']) * 6.0
        
        if img.get('analysis', {}).get('likely_chart_or_diagram', False):
            data_keywords = {
                "chart", "graph", "data", "statistics", "figure", "diagram", 
                "plot", "table", "number", "percentage", "rate", "analysis", 
                "trend", "comparison", "result", "finding", "metric", "value"
            }
            keyword_matches = search_terms.intersection(data_keywords)
            relevance_score += len(keyword_matches) * 5.0
            if not keyword_matches:
                relevance_score += 2.0
        
        if img.get('analysis', {}).get('likely_contains_text', False):
            relevance_score += 2.5
        
        quality = img.get('analysis', {}).get('image_quality', 'low')
        if quality == 'high':
            relevance_score += 2.5
        elif quality == 'medium':
            relevance_score += 1.2
        
        if img.get('has_meaningful_content', False):
            relevance_score += 1.5
        
        if img.get('width', 0) > 500 and img.get('height', 0) > 300:
            relevance_score += 1.5
        
        min_relevance_threshold = 0.7 if include_all_pages else 2.5
        if relevance_score >= min_relevance_threshold:
            img_copy = img.copy()
            img_copy['relevance_score'] = relevance_score
            relevant_images.append(img_copy)
    
    relevant_images.sort(key=lambda x: (-x['relevance_score'], x['page'], x.get('index', 0)))
    return relevant_images[:20]

def main():
    """Main function to run the Streamlit app."""
    st.markdown("""
        <style>
            .stApp { background-color: #F0F2F6; }
            .st-emotion-cache-16txtl3 { padding: 1rem; }
            .stSpinner { margin: 1rem auto; }
            .error-message { color: #D32F2F; font-weight: bold; }
            .retry-button { margin-top: 1rem; }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: #1E1E1E;'>✨ IntelliDoc Agent</h1>", unsafe_allow_html=True)

    if 'agent' not in st.session_state:
        st.session_state.agent = PDFAgent()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "images" not in st.session_state:
        st.session_state.images = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "summary_failed" not in st.session_state:
        st.session_state.summary_failed = False

    with st.sidebar:
        st.header("⚙️ Controls")
        uploaded_files = st.file_uploader(
            "Upload PDF documents", 
            type=['pdf'], 
            accept_multiple_files=True
        )

        if uploaded_files and st.button("🚀 Process Documents", type="primary", use_container_width=True):
            with st.spinner("Processing documents..."):
                st.session_state.processed_files = [f.name for f in uploaded_files]
                st.session_state.messages = []
                st.session_state.images = []
                st.session_state.summary = None
                st.session_state.summary_failed = False
                
                try:
                    chunks, images = run_async(st.session_state.agent.process_documents(uploaded_files, st.progress(0)))
                    st.session_state.images = images
                    if os.path.exists(st.session_state.agent.summary_cache_file):
                        with open(st.session_state.agent.summary_cache_file, 'rb') as f:
                            st.session_state.summary = pickle.load(f)
                        if st.session_state.summary.startswith("Error") or "insufficient" in st.session_state.summary.lower():
                            st.session_state.summary_failed = True
                    st.success("Documents processed successfully!")
                    st.session_state.messages.append({"role": "assistant", "content": "✅ Documents are ready. Feel free to ask any questions."})
                except Exception as e:
                    logger.error(f"Error processing documents: {e}")
                    st.error(f"Error processing documents: {e}")
                    st.session_state.summary_failed = True
                st.rerun()

        if st.session_state.processed_files:
            st.markdown("---")
            st.subheader("✅ Processed Files")
            for file_name in st.session_state.processed_files:
                st.info(f"📄 {file_name}")
        
        if st.session_state.images:
            st.markdown("---")
            st.subheader("📊 Debug Info")
            st.info(f"Total images: {len(st.session_state.images)}")
            st.info(f"Images with OCR: {sum(1 for img in st.session_state.images if img.get('ocr_text'))}")
            st.info(f"Images with tables: {sum(1 for img in st.session_state.images if img.get('tables'))}")
            st.info(f"Meaningful images: {sum(1 for img in st.session_state.images if img.get('has_meaningful_content'))}")
        
        if st.session_state.summary_failed:
            st.markdown("---")
            if st.button("🔄 Retry Summary", key="retry_summary", help="Retry generating the document summary"):
                with st.spinner("Retrying summary generation..."):
                    try:
                        chunks, _ = run_async(st.session_state.agent.process_documents(
                            [open(f, 'rb') for f in uploaded_files], st.progress(0)
                        ))
                        summary = run_async(st.session_state.agent.llm_handler.summarize_document(chunks, st.session_state.images))
                        with open(st.session_state.agent.summary_cache_file, 'wb') as f:
                            pickle.dump(summary, f)
                        st.session_state.summary = summary
                        st.session_state.summary_failed = False if not summary.startswith("Error") else True
                        st.success("Summary retry completed!")
                    except Exception as e:
                        logger.error(f"Error retrying summary: {e}")
                        st.error(f"Error retrying summary: {e}")
                    st.rerun()

        st.markdown("---")
        if st.button("🗑️ Clear Session", use_container_width=True):
            for key in list(st.session_state.keys()): 
                del st.session_state[key]
            if os.path.exists(st.session_state.agent.vector_cache_file):
                os.unlink(st.session_state.agent.vector_cache_file)
            if os.path.exists(st.session_state.agent.summary_cache_file):
                os.unlink(st.session_state.agent.summary_cache_file)
            st.rerun()

    if st.session_state.summary:
        with st.expander("📝 Document Summary", expanded=True):
            st.markdown(st.session_state.summary)
            if st.session_state.summary_failed:
                st.warning("The summary may be incomplete. Click 'Retry Summary' in the sidebar to try again.")

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

                if "sources" in message and message["sources"] and message.get("is_document_related", True):
                    with st.expander("📚 Show Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.info(f"**Source {i+1} (Page {source['page']}, Score: {source.get('similarity_score', 0):.3f})**\n\n{source['text']}")

                if "images" in message and message["images"]:
                    st.markdown("**🖼️ Relevant Images:**")
                    num_images = len(message["images"])
                    cols_per_row = 3
                    for i in range(0, num_images, cols_per_row):
                        cols = st.columns(min(cols_per_row, num_images - i))
                        for j, img in enumerate(message["images"][i:i+cols_per_row]):
                            with cols[j]:
                                st.image(
                                    img['image'], 
                                    caption=f"📄 Page {img['page']} | Score: {img.get('relevance_score', 0):.1f}", 
                                    use_container_width=True
                                )
                                if img.get('ocr_text') and len(img['ocr_text'].strip()) > 10:
                                    with st.expander(f"Text from Page {img['page']}"):
                                        st.text(img['ocr_text'][:400] + "..." if len(img['ocr_text']) > 400 else img['ocr_text'])
                                if img.get('tables'):
                                    with st.expander(f"📊 Tables from Page {img['page']}"):
                                        for table in img['tables']:
                                            st.table(table)

    if query := st.chat_input("Ask a question about your documents..."):
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role": "user", "content": query})
            
            try:
                if st.session_state.processed_files:
                    if query.lower().strip() in ["what is this document about", "summarize document", "document summary", "tell me more about the document"]:
                        if st.session_state.summary:
                            response_text = st.session_state.summary
                            sources = []
                        else:
                            response_text = "No summary available. Please process documents again or retry summary generation."
                            sources = []
                    else:
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
                
                response_text = sanitize_response(response_text)
                is_doc_related = is_query_document_related(query, response_text)
                images_to_display = []
                if is_doc_related and st.session_state.images:
                    cited_pages = parse_source_pages(response_text)
                    images_to_display = find_relevant_images_enhanced(
                        query, cited_pages, st.session_state.images, response_text, sources
                    )

                assistant_message = {
                    "role": "assistant", 
                    "content": response_text,
                    "sources": sources if is_doc_related else [],
                    "images": images_to_display,
                    "is_document_related": is_doc_related
                }
            except Exception as e:
                logger.error(f"Critical error processing query: {e}")
                st.error(f"A critical error occurred: {e}")
                assistant_message = {
                    "role": "assistant",
                    "content": sanitize_response(f"I'm sorry, I encountered an error: {e}"),
                    "error": True
                }
            
            st.session_state.messages.append(assistant_message)
            st.rerun()

if __name__ == "__main__":
    if not os.getenv('GOOGLE_API_KEY'):
        st.error("❌ **GOOGLE_API_KEY not found.** Please set it in your environment variables or Streamlit Cloud secrets.")
    else:
        main()