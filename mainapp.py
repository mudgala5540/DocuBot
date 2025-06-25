# mainapp.py - TEMPORARY VERSION TO BYPASS CACHE ERROR

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
from typing import List, Dict, Any, Tuple

# Apply the patch for nested asyncio event loops. CRITICAL for Streamlit.
nest_asyncio.apply()

load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="IntelliDoc Agent",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Robust Asyncio Runner for Streamlit ---
def run_async(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)

# --- Agent Class ---
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
            progress_text = f"Processing file {i+1}/{total_files}: {file.name}"
            progress_bar.progress((i + 1) / total_files, text=progress_text)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                text_chunks_task = self.pdf_processor.extract_text_chunks(tmp_path)
                images_task = self.pdf_processor.extract_images(tmp_path)
                text_chunks, images = await asyncio.gather(text_chunks_task, images_task)

                image_processing_tasks = []
                for img_data in images:
                    image_processing_tasks.append(self.process_single_image(img_data))
                
                processed_images = await asyncio.gather(*image_processing_tasks)

                all_chunks.extend(text_chunks)
                all_images.extend(processed_images)

            finally:
                os.unlink(tmp_path)
        
        progress_bar.progress(1.0, text="Creating document embeddings...")
        if all_chunks:
            await self.vector_store.add_documents(all_chunks)
        
        return all_chunks, all_images

    async def process_single_image(self, img_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            ocr_text, analysis = await asyncio.gather(
                self.image_processor.extract_text_from_image(img_data['image']),
                self.image_processor.analyze_image_content(img_data['image'])
            )
            img_data['ocr_text'] = ocr_text.lower() if ocr_text else ""
            img_data['analysis'] = analysis
        except Exception as e:
            print(f"Error processing image on page {img_data.get('page', 'N/A')}: {e}")
            img_data['ocr_text'] = ""
            img_data['analysis'] = {}
        return img_data

    async def query_agent(self, query: str, top_k=10) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        relevant_chunks = await self.vector_store.similarity_search(query, k=top_k)
        llm_response = await self.llm_handler.get_agentic_response(query, relevant_chunks)
        return llm_response, relevant_chunks

# --- Enhanced Helper Functions ---
def find_relevant_images(
    llm_response: Dict[str, Any], 
    all_images: List[Dict[str, Any]], 
    source_chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not all_images or llm_response["query_type"] != "DOCUMENT_QUERY":
        return []

    cited_pages = llm_response.get("cited_pages", [])
    if not cited_pages and source_chunks:
        cited_pages = sorted(list(set(chunk.get('page', 0) for chunk in source_chunks)))
    
    if not cited_pages:
        return []

    query_and_response = llm_response["response_text"]
    search_words = set(re.findall(r'\b\w{4,}\b', query_and_response.lower()))

    scored_images = []
    for img in all_images:
        if img.get('page') not in cited_pages:
            continue

        score = 5.0
        if img.get("ocr_text"):
            ocr_words = set(img["ocr_text"].split())
            matches = search_words.intersection(ocr_words)
            if matches:
                score += len(matches) * 2.0
        
        analysis = img.get("analysis", {})
        if analysis.get('likely_chart_or_diagram'):
            data_keywords = {"chart", "graph", "data", "figure", "table", "percent", "trend"}
            if search_words.intersection(data_keywords):
                score += 10.0
            else:
                score += 3.0
        
        if analysis.get('image_quality') == 'high':
            score += 2.0

        img_copy = img.copy()
        img_copy['relevance_score'] = round(score, 1)
        scored_images.append(img_copy)

    scored_images.sort(key=lambda x: x['relevance_score'], reverse=True)
    return scored_images

# --- Streamlit UI and Application Flow ---
def main():
    st.markdown("<h1 style='text-align: center; color: #1E1E1E;'>✨ IntelliDoc Agent</h1>", unsafe_allow_html=True)

    if 'agent' not in st.session_state:
        st.session_state.agent = PDFAgent()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "all_images" not in st.session_state:
        st.session_state.all_images = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    with st.sidebar:
        st.header("⚙️ Controls")
        uploaded_files = st.file_uploader(
            "Upload PDF documents", 
            type=['pdf'], 
            accept_multiple_files=True
        )

        if uploaded_files and st.button("🚀 Process Documents", type="primary", use_container_width=True):
            with st.spinner("Processing documents... This may take a moment."):
                st.session_state.processed_files = {f.name for f in uploaded_files}
                st.session_state.messages = []
                st.session_state.all_images = []
                
                progress_bar = st.progress(0, text="Initializing...")
                
                try:
                    _, images = run_async(st.session_state.agent.process_documents(uploaded_files, progress_bar))
                    st.session_state.all_images = images
                    progress_bar.empty()
                    st.success("Documents processed successfully!")
                    st.session_state.messages.append({"role": "assistant", "content": "✅ Documents are ready. What would you like to know?"})
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Error processing documents: {e}")
            st.rerun()

        if st.session_state.processed_files:
            st.markdown("---")
            st.subheader("✅ Processed Files")
            for file_name in st.session_state.processed_files:
                st.info(f"📄 {file_name}")
            
            with st.expander("📊 Usage Stats"):
                stats = st.session_state.agent.llm_handler.get_usage_stats()
                st.json(stats)
        st.markdown("---")
        if st.button("🗑️ Clear Session", use_container_width=True):
            for key in list(st.session_state.keys()): 
                del st.session_state[key]
            st.rerun()

    if not st.session_state.processed_files:
        st.info("Welcome! Please upload your PDF documents in the sidebar to get started.")
    
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "✨"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Show Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.info(f"**Source {i+1} (Page {source['page']}, Score: {source.get('similarity_score', 0):.3f})**\n\n> {source['text']}")
            
            if "images" in message and message["images"]:
                st.markdown("**🖼️ Relevant Images**")
                cols = st.columns(3)
                for i, img_data in enumerate(message["images"]):
                    with cols[i % 3]:
                        st.image(
                            img_data['image'], 
                            caption=f"Page {img_data['page']} | Score: {img_data['relevance_score']}", 
                            use_container_width=True
                        )

    if query := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.processed_files:
            st.warning("Please upload and process documents before asking questions.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("Thinking..."):
                try:
                    llm_response, sources = run_async(st.session_state.agent.query_agent(query))
                    
                    response_text = llm_response.get("response_text", "I'm sorry, I couldn't generate a valid response.")
                    
                    images_to_display = []
                    if llm_response.get("query_type") == "DOCUMENT_QUERY":
                        images_to_display = find_relevant_images(llm_response, st.session_state.all_images, sources)

                    assistant_message = {
                        "role": "assistant", 
                        "content": response_text,
                        "sources": sources if llm_response.get("query_type") == "DOCUMENT_QUERY" else [],
                        "images": images_to_display
                    }

                except Exception as e:
                    st.error(f"A critical error occurred: {e}")
                    assistant_message = {
                        "role": "assistant",
                        "content": f"I'm sorry, I encountered an error: {e}",
                    }
            
            st.session_state.messages.append(assistant_message)
            st.rerun()

if __name__ == "__main__":
    if not os.getenv('GOOGLE_API_KEY'):
        st.error("❌ **GOOGLE_API_KEY not found.** Please set it in your .env file or Streamlit secrets.")
    else:
        main()