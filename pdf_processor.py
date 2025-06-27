import fitz
import io
from PIL import Image
import hashlib
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
import logging
import os

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def extract_text_chunks(self, pdf_path: str, chunk_size: int = 1200, overlap: int = 300) -> List[Dict[str, Any]]:
        """Extract text from PDF and create intelligent chunks"""
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return []
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self.executor, self._extract_text_chunks_sync, pdf_path, chunk_size, overlap)
        except Exception as e:
            logger.error(f"Error extracting text chunks from {pdf_path}: {e}")
            return []
    
    def _extract_text_chunks_sync(self, pdf_path: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Synchronous text extraction with intelligent chunking"""
        try:
            doc = fitz.open(pdf_path)
            chunks = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if not text.strip():
                    logger.debug(f"No text found on page {page_num + 1}")
                    continue
                
                text = self._clean_text(text)
                page_chunks = self._create_semantic_chunks(text, chunk_size, overlap, page_num)
                chunks.extend(page_chunks)
            
            doc.close()
            logger.info(f"Extracted {len(chunks)} text chunks from {pdf_path}")
            return chunks
        except Exception as e:
            logger.error(f"Error in synchronous text extraction: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    def _create_semantic_chunks(self, text: str, chunk_size: int, overlap: int, page_num: int) -> List[Dict[str, Any]]:
        """Create semantically meaningful chunks"""
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunk_data = {
                    'text': current_chunk.strip(),
                    'page': page_num + 1,
                    'chunk_id': hashlib.md5(current_chunk.encode()).hexdigest()[:8],
                    'type': 'text'
                }
                chunks.append(chunk_data)
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + paragraph
            else:
                current_chunk += " " + paragraph if current_chunk else paragraph
        
        if current_chunk.strip():
            chunk_data = {
                'text': current_chunk.strip(),
                'page': page_num + 1,
                'chunk_id': hashlib.md5(current_chunk.encode()).hexdigest()[:8],
                'type': 'text'
            }
            chunks.append(chunk_data)
        
        return chunks
    
    async def extract_images(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract images from PDF"""
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return []
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self.executor, self._extract_images_sync, pdf_path)
        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path}: {e}")
            return []
    
    def _extract_images_sync(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Synchronous image extraction with improved filtering"""
        try:
            doc = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(doc)):
                for img_info in doc.get_page_images(page_num):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        if not base_image.get("colorspace") or base_image["width"] < 100 or base_image["height"] < 100:
                            continue
                        
                        img_pil = Image.open(io.BytesIO(image_bytes))
                        
                        image_data = {
                            'image': img_pil,
                            'page': page_num + 1,
                            'index': xref,
                            'width': base_image["width"],
                            'height': base_image["height"],
                        }
                        images.append(image_data)
                    except Exception as e:
                        logger.warning(f"Skipping problematic image on page {page_num + 1}: {e}")
                        continue
            
            doc.close()
            logger.info(f"Extracted {len(images)} images from {pdf_path}")
            return images
        except Exception as e:
            logger.error(f"Error in synchronous image extraction: {e}")
            return []
    
    def get_document_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract document metadata"""
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return {}
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            info = {
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'subject': metadata.get('subject', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
                'modification_date': metadata.get('modDate', ''),
                'pages': len(doc),
                'encrypted': doc.needs_pass
            }
            doc.close()
            return info
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}