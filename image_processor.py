import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from typing import Dict, Any, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import logging
import re

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using OCR"""
        if not image:
            logger.warning("No image provided for OCR")
            return ""
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self.executor, self._extract_text_sync, image)
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def _extract_text_sync(self, image: Image.Image) -> str:
        """Synchronous OCR text extraction"""
        try:
            processed_image = self._preprocess_image(image)
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?@#$%^&*()_+-=[]{}|;:,.<>?/~` '
            text = pytesseract.image_to_string(processed_image, config=custom_config)
            cleaned_text = self._clean_ocr_text(text)
            return cleaned_text
        except Exception as e:
            logger.error(f"OCR Error: {e}")
            return ""
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR accuracy"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = opencv_image.shape[:2]
        if height < 100 or width < 100:
            scale_factor = max(200/height, 200/width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            opencv_image = cv2.resize(opencv_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return Image.fromarray(cleaned)
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean and normalize OCR extracted text"""
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?@#$%^&*()_+\-=\[\]{}|;:,.<>?/~`]', '', text)
        replacements = {
            '0': 'O', '1': 'l', '5': 'S'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        words = text.split()
        cleaned_words = [word.strip() for word in words if len(word.strip()) > 1]
        return ' '.join(cleaned_words).strip()
    
    async def analyze_image_content(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image content and properties"""
        if not image:
            logger.warning("No image provided for analysis")
            return {}
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self.executor, self._analyze_image_sync, image)
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {}
    
    def _analyze_image_sync(self, image: Image.Image) -> Dict[str, Any]:
        """Synchronous image analysis"""
        try:
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            large_contours = [c for c in contours if cv2.contourArea(c) > 100]
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
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return {}
    
    async def extract_tables_from_image(self, image: Image.Image) -> List[List[str]]:
        """Extract table data from image"""
        if not image:
            logger.warning("No image provided for table extraction")
            return []
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self.executor, self._extract_tables_sync, image)
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return []
    
    def _extract_tables_sync(self, image: Image.Image) -> List[List[str]]:
        """Synchronous table extraction"""
        try:
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 4:
                table_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
                text = pytesseract.image_to_string(image, config=table_config)
                lines = text.strip().split('\n')
                table_data = []
                for line in lines:
                    if line.strip():
                        cells = [cell.strip() for cell in re.split(r'\s{2,}', line) if cell.strip()]
                        if cells:
                            table_data.append(cells)
                return table_data
            return []
        except Exception as e:
            logger.error(f"Table extraction error: {e}")
            return []
    
    async def batch_process_images(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Process multiple images in batch"""
        if not images:
            logger.warning("No images provided for batch processing")
            return []
        tasks = [self._process_single_image(image, i) for i, image in enumerate(images)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing image {i}: {result}")
                processed_results.append({
                    'image_index': i,
                    'error': str(result),
                    'ocr_text': '',
                    'analysis': {},
                    'tables': [],
                    'has_text': False,
                    'has_tables': False
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_single_image(self, image: Image.Image, index: int) -> Dict[str, Any]:
        """Process a single image completely"""
        try:
            ocr_text = await self.extract_text_from_image(image)
            analysis = await self.analyze_image_content(image)
            tables = await self.extract_tables_from_image(image)
            
            return {
                'image_index': index,
                'ocr_text': ocr_text,
                'analysis': analysis,
                'tables': tables,
                'has_text': len(ocr_text.strip()) > 0,
                'has_tables': len(tables) > 0
            }
        except Exception as e:
            logger.error(f"Error processing single image {index}: {e}")
            return {
                'image_index': index,
                'error': str(e),
                'ocr_text': '',
                'analysis': {},
                'tables': [],
                'has_text': False,
                'has_tables': False
            }