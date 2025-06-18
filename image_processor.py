import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from typing import Dict, Any, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io

class ImageProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Configure Tesseract (you may need to set the path based on your installation)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
        # For Linux/Mac, tesseract should be in PATH
    
    async def extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using OCR"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._extract_text_sync, image)
    
    def _extract_text_sync(self, image: Image.Image) -> str:
        """Synchronous OCR text extraction"""
        try:
            # Preprocess image for better OCR
            processed_image = self._preprocess_image(image)
            
            # Configure OCR
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?@#$%^&*()_+-=[]{}|;:,.<>?/~` '
            
            # Extract text
            text = pytesseract.image_to_string(processed_image, config=custom_config)
            
            # Clean extracted text
            cleaned_text = self._clean_ocr_text(text)
            
            return cleaned_text
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR accuracy"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Resize if image is too small
        height, width = opencv_image.shape[:2]
        if height < 100 or width < 100:
            scale_factor = max(200/height, 200/width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            opencv_image = cv2.resize(opencv_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((1,1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to PIL
        result_image = Image.fromarray(cleaned)
        
        return result_image
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean and normalize OCR extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s.,!?@#$%^&*()_+\-=\[\]{}|;:,.<>?/~`]', '', text)
        
        # Fix common OCR mistakes
        replacements = {
            '0': 'O',  # Only in specific contexts
            '1': 'l',  # Only in specific contexts
            '5': 'S',  # Only in specific contexts
        }
        
        # Apply context-aware replacements (basic implementation)
        words = text.split()
        cleaned_words = []
        
        for word in words:
            # Skip very short words that might be artifacts
            if len(word.strip()) > 1:
                cleaned_words.append(word.strip())
        
        return ' '.join(cleaned_words).strip()
    
    async def analyze_image_content(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image content and properties"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._analyze_image_sync, image)
    
    def _analyze_image_sync(self, image: Image.Image) -> Dict[str, Any]:
        """Synchronous image analysis"""
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Basic image properties
        height, width = gray.shape
        
        # Calculate image quality metrics
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # Sharpness
        
        # Detect if image contains text (using edge detection)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Detect if image might be a chart/graph
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        # Basic color analysis
        avg_color = np.mean(opencv_image, axis=(0, 1))
        
        return {
            'width': width,
            'height': height,
            'aspect_ratio': width / height,
            'sharpness': float(laplacian_var),
            'edge_density': float(edge_density),
            'num_contours': len(large_contours),
            'avg_color_bgr': avg_color.tolist(),
            'likely_contains_text': edge_density > 0.1,
            'likely_chart_or_diagram': len(large_contours) > 5 and edge_density > 0.15,
            'image_quality': 'high' if laplacian_var > 100 else 'medium' if laplacian_var > 50 else 'low'
        }
    
    async def extract_tables_from_image(self, image: Image.Image) -> List[List[str]]:
        """Extract table data from image"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._extract_tables_sync, image)
    
    def _extract_tables_sync(self, image: Image.Image) -> List[List[str]]:
        """Synchronous table extraction"""
        try:
            # Preprocess for table detection
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            
            # Combine lines to find table structure
            table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Find contours (potential cells)
            contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # If we detect table-like structure, use OCR with table configuration
            if len(contours) > 4:  # Likely a table
                table_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
                text = pytesseract.image_to_string(image, config=table_config)
                
                # Simple table parsing (basic implementation)
                lines = text.strip().split('\n')
                table_data = []
                
                for line in lines:
                    if line.strip():
                        # Split by multiple spaces (table columns)
                        cells = [cell.strip() for cell in line.split('  ') if cell.strip()]
                        if cells:
                            table_data.append(cells)
                
                return table_data
            
            return []
            
        except Exception as e:
            print(f"Table extraction error: {e}")
            return []
    
    async def batch_process_images(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Process multiple images in batch"""
        tasks = []
        
        for i, image in enumerate(images):
            task = asyncio.create_task(self._process_single_image(image, i))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'image_index': i,
                    'error': str(result),
                    'ocr_text': '',
                    'analysis': {}
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_single_image(self, image: Image.Image, index: int) -> Dict[str, Any]:
        """Process a single image completely"""
        # Extract text
        ocr_text = await self.extract_text_from_image(image)
        
        # Analyze image
        analysis = await self.analyze_image_content(image)
        
        # Extract tables if likely to contain them
        tables = []
        if analysis.get('likely_chart_or_diagram', False):
            tables = await self.extract_tables_from_image(image)
        
        return {
            'image_index': index,
            'ocr_text': ocr_text,
            'analysis': analysis,
            'tables': tables,
            'has_text': len(ocr_text.strip()) > 0,
            'has_tables': len(tables) > 0
        }