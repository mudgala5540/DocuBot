import fitz  # PyMuPDF
import io
from PIL import Image
import hashlib
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re

class PDFProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def extract_text_chunks(self, pdf_path: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Extract text from PDF and create intelligent chunks"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._extract_text_chunks_sync, pdf_path, chunk_size, overlap)
    
    def _extract_text_chunks_sync(self, pdf_path: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Synchronous text extraction with intelligent chunking"""
        doc = fitz.open(pdf_path)
        chunks = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            if not text.strip():
                continue
            
            # Clean and preprocess text
            text = self._clean_text(text)
            
            # Create semantic chunks
            page_chunks = self._create_semantic_chunks(text, chunk_size, overlap, page_num)
            chunks.extend(page_chunks)
        
        doc.close()
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between joined words
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words across lines
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines to single
        
        return text.strip()
    
    def _create_semantic_chunks(self, text: str, chunk_size: int, overlap: int, page_num: int) -> List[Dict[str, Any]]:
        """Create semantically meaningful chunks"""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                # Create chunk
                chunk_data = {
                    'text': current_chunk.strip(),
                    'page': page_num + 1,
                    'chunk_id': hashlib.md5(current_chunk.encode()).hexdigest()[:8],
                    'type': 'text'
                }
                chunks.append(chunk_data)
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + paragraph
            else:
                current_chunk += " " + paragraph if current_chunk else paragraph
        
        # Add final chunk
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
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._extract_images_sync, pdf_path)
    
    def _extract_images_sync(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Synchronous image extraction"""
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Skip if image is too small (likely decorative)
                    if pix.width < 100 or pix.height < 100:
                        pix = None
                        continue
                    
                    # Convert to PIL Image
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        img_pil = Image.open(io.BytesIO(img_data))
                        
                        image_info = {
                            'image': img_pil,
                            'page': page_num + 1,
                            'index': img_index,
                            'width': pix.width,
                            'height': pix.height,
                            'size': len(img_data),
                            'format': 'PNG'
                        }
                        images.append(image_info)
                    
                    pix = None
                    
                except Exception as e:
                    print(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
                    continue
        
        doc.close()
        return images
    
    def get_document_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract document metadata"""
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