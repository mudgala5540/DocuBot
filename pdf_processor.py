import fitz  # PyMuPDF
import io
from PIL import Image
import hashlib
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
import os # <--- THE FIX IS HERE

class PDFProcessor:
    def __init__(self):
        # Increased workers as PDF processing can be parallelized well
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
    
    async def extract_text_chunks(self, pdf_path: str, chunk_size: int = 1000, overlap: int = 150) -> List[Dict[str, Any]]:
        """Extract text from PDF and create intelligent chunks asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._extract_text_chunks_sync, pdf_path, chunk_size, overlap)
    
    def _extract_text_chunks_sync(self, pdf_path: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Synchronous text extraction with intelligent chunking."""
        all_text_by_page = []
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    cleaned_text = self._clean_text(text)
                    all_text_by_page.append({'text': cleaned_text, 'page': page_num + 1})
        
        return self._create_semantic_chunks(all_text_by_page, chunk_size, overlap)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = text.replace('\n', ' ').replace('\r', '') # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text) # Collapse multiple whitespace
        text = re.sub(r'(\w)-\s(\w)', r'\1\2', text) # Fix hyphenated words
        return text.strip()
    
    def _create_semantic_chunks(self, pages: List[Dict[str, Any]], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Creates chunks from text, respecting page boundaries."""
        chunks = []
        # Use sentence splitting for more semantic chunks
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""] # Prioritize semantic breaks
        )

        for page_data in pages:
            page_text = page_data['text']
            page_num = page_data['page']
            page_chunks = text_splitter.split_text(page_text)
            
            for chunk_text in page_chunks:
                chunk_id = hashlib.md5((f"{page_num}-{chunk_text}").encode()).hexdigest()[:12]
                chunks.append({
                    'text': chunk_text,
                    'page': page_num,
                    'chunk_id': chunk_id,
                    'type': 'text'
                })
        return chunks

    async def extract_images(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract images from PDF asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._extract_images_sync, pdf_path)
    
    def _extract_images_sync(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Synchronous image extraction with robust filtering for real images."""
        images = []
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                for img_info in doc.get_page_images(page_num, full=True):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        
                        # Definitive check for real, content-ful images
                        if not base_image or not base_image.get("image"):
                            continue
                        if base_image["width"] < 100 or base_image["height"] < 100:
                            continue # Filter out small icons, logos etc.
                        if "cs-name" in base_image and "DeviceGray" in base_image["cs-name"] and base_image["bpc"] == 1:
                             continue # Filter out simple black/white masks often used in vector graphics

                        img_pil = Image.open(io.BytesIO(base_image["image"]))
                        
                        images.append({
                            'image': img_pil,
                            'page': page_num + 1,
                            'index': xref,
                            'width': base_image["width"],
                            'height': base_image["height"],
                        })
                    except Exception as e:
                        print(f"Skipping problematic image object on page {page_num + 1}: {e}")
                        continue
        return images