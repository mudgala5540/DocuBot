import google.generativeai as genai
import os
from typing import List, Dict, Any
import asyncio
import json
import re
from datetime import datetime
import threading
import logging
import hashlib
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMHandler:
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.total_tokens_used = 0
        self.requests_made = 0
        self._lock = threading.Lock()
        self.response_cache = {}
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Seconds between requests
    
    async def generate_response(self, query: str, relevant_chunks: List[Dict[str, Any]], max_tokens: int = 2000) -> str:
        """Generate response using relevant document chunks"""
        if not query.strip():
            logger.warning("Empty query provided")
            return "Please provide a valid query."
        
        cache_key = hashlib.md5((query + json.dumps(relevant_chunks)).encode()).hexdigest()
        if cache_key in self.response_cache:
            logger.info(f"Returning cached response for query: {query}")
            return self.response_cache[cache_key]
        
        context = self._prepare_context(relevant_chunks)
        prompt = self._create_enhanced_agentic_prompt(query, context)
        
        try:
            # Rate limiting
            with self._lock:
                current_time = time.time()
                if current_time - self.last_request_time < self.min_request_interval:
                    await asyncio.sleep(self.min_request_interval - (current_time - self.last_request_time))
                self.last_request_time = time.time()
                self.requests_made += 1
            
            response = await self._generate_with_retry_safe(prompt, max_tokens)
            self.response_cache[cache_key] = response
            logger.info(f"Generated response for query: {query}")
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def _prepare_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Prepare context with prioritized chunks"""
        if not chunks:
            logger.warning("No chunks provided for context")
            return "No relevant context found."
        
        chunks.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        context_parts = []
        total_length = 0
        max_context_length = 25000  # Increased for larger documents
        
        for chunk in chunks[:30]:
            chunk_text = f"[PAGE {chunk.get('page', 'N/A')}]\n{chunk['text']}\n"
            if total_length + len(chunk_text) > max_context_length:
                break
            context_parts.append(chunk_text)
            total_length += len(chunk_text)
        
        return "\n".join(context_parts)
    
    def _create_enhanced_agentic_prompt(self, query: str, context: str) -> str:
        """Enhanced prompt for consistent and accurate responses"""
        prompt = f"""You are IntelliDoc Agent, a professional document analysis AI. Your role is to provide accurate, detailed, and well-structured responses based solely on the provided document context.

**Query Processing Steps:**
1. **Query Classification**:
   - Identify if the query is a greeting (e.g., "hi", "hello"), irrelevant/nonsense (e.g., "asdf", "weather"), or document-related.
2. **Response Strategy**:
   - For greetings: Respond politely without using document context (e.g., "Hello! How can I assist you with your documents?").
   - For irrelevant/nonsense: Redirect to document-related topics (e.g., "I specialize in document analysis. Please ask about your uploaded documents.").
   - For document queries: Provide a detailed answer using only the context, including specific details, bullet points, and source citations.

**DOCUMENT CONTEXT:**
{context}

**USER QUERY:** {query}

**RESPONSE FORMAT:**
- **Answer**: [Direct and concise answer to the query]
- **Details**: [Comprehensive explanation with bullet points if applicable]
- **Source**: [List all page numbers referenced, e.g., (Source: Page 1, Page 3)]
"""
        return prompt

    async def _generate_with_retry_safe(self, prompt: str, max_tokens: int, max_retries: int = 3) -> str:
        """Generate response with retry logic"""
        def sync_generate():
            genai_sync = genai  # Avoid re-importing
            genai_sync.configure(api_key=self.api_key)
            model_sync = genai_sync.GenerativeModel('gemini-1.5-flash')
            
            for attempt in range(max_retries):
                try:
                    generation_config = genai_sync.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.1,
                        top_p=0.95,
                        top_k=40
                    )
                    response = model_sync.generate_content(prompt, generation_config=generation_config)
                    return response.text
                except Exception as e:
                    logger.error(f"LLM generation failed on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(2 ** attempt)
            return "Failed to generate response after multiple attempts."
        
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, sync_generate)

    async def summarize_document(self, chunks: List[Dict[str, Any]], image_data: List[Dict[str, Any]] = None) -> str:
        """Generate a comprehensive summary of the entire document"""
        if not chunks:
            logger.warning("No chunks provided for summarization")
            return "No document content available for summarization. Please upload a valid document."
        
        # Dynamic chunk sampling based on document length
        total_chunks = len(chunks)
        sample_size = min(50, max(10, total_chunks // 2))  # Dynamic sample size
        step = max(1, total_chunks // sample_size)
        sampled_chunks = chunks[::step][:sample_size]
        
        if len(sampled_chunks) < 5:
            logger.warning(f"Insufficient chunks ({len(sampled_chunks)}) for comprehensive summary")
            return "Insufficient content to generate a meaningful summary. The document may be too short or contain insufficient text."
        
        image_context = ""
        if image_data:
            for img in image_data[:15]:  # Increased limit for images
                if img.get('ocr_text') or img.get('tables'):
                    image_context += f"[PAGE {img['page']}]\n"
                    if img.get('ocr_text'):
                        image_context += f"Image Text: {img['ocr_text'][:400]}\n"
                    if img.get('tables'):
                        image_context += f"Table Data: {json.dumps(img['tables'][:3], indent=2)}\n"
        
        context = self._prepare_context(sampled_chunks) + "\n" + image_context
        
        prompt = f"""You are IntelliDoc Agent, tasked with creating a comprehensive summary of a document based on the provided content.

**DOCUMENT CONTENT:**
{context}

**INSTRUCTIONS:**
Provide a detailed summary with the following sections:
1. **Overview**: Describe the document type and its primary purpose.
2. **Key Topics**: List the main subjects or themes covered.
3. **Critical Findings**: Highlight key conclusions, results, or recommendations.
4. **Important Data**: Include significant numbers, statistics, dates, or metrics.
5. **Structure**: Describe the document's organization and main sections.
6. **Visual Elements**: Summarize any charts, tables, or images (if provided).
7. **Action Items**: Note any next steps or recommendations.

**FORMATTING:**
- Use clear markdown headings for each section.
- Use bullet points for key details.
- Cite page numbers for all referenced information.
- If information is missing, state: "No relevant information found in the provided context."
- Ensure all sections are addressed, even if briefly, to maintain consistency.

**OUTPUT:**
Provide a concise yet comprehensive summary in markdown format.
"""
        try:
            response = await self._generate_with_retry_safe(prompt, 2500)
            logger.info("Document summary generated successfully")
            return response
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {str(e)}. Please try re-uploading the document or contact support."
    
    async def extract_key_information(self, chunks: List[Dict[str, Any]], info_type: str = "general") -> Dict[str, Any]:
        """Extract specific types of information from documents"""
        if not chunks:
            logger.warning("No chunks provided for information extraction")
            return {"error": "No document content available"}
        
        context = self._prepare_context(chunks[:35])
        
        if info_type == "financial":
            prompt = f"""Extract financial information from the document:

{context}

**Categories**:
- Monetary amounts (e.g., $ amounts, costs, revenues)
- Financial metrics (e.g., ROI, profit margins)
- Financial dates (e.g., fiscal years, deadlines)
- Financial entities (e.g., companies, banks)
- Performance data (e.g., growth rates, trends)
- Budget items (e.g., allocations, expenditures)

**Format**: Use markdown with clear categories and citations."""
        elif info_type == "technical":
            prompt = f"""Extract technical information from the document:

{context}

**Categories**:
- Technical specifications (e.g., models, versions)
- Procedures/processes (e.g., workflows)
- Equipment/tools (e.g., names, specifications)
- Technical parameters (e.g., configurations)
- Safety information (e.g., warnings)
- Standards/compliance (e.g., regulations)

**Format**: Use markdown with clear categories and citations."""
        else:
            prompt = f"""Extract key information from the document:

{context}

**Categories**:
- Important people (e.g., names, roles)
- Key dates (e.g., deadlines, milestones)
- Locations (e.g., addresses, facilities)
- Critical numbers (e.g., statistics, quantities)
- Main conclusions (e.g., findings, decisions)
- Action items (e.g., tasks, responsibilities)
- Contact details (e.g., emails, phone numbers)

**Format**: Use markdown with clear categories and citations."""
        
        try:
            response = await self._generate_with_retry_safe(prompt, 1500)
            logger.info(f"Extracted {info_type} information successfully")
            return {"extracted_info": response, "info_type": info_type}
        except Exception as e:
            logger.error(f"Error extracting information: {e}")
            return {"error": f"Error extracting information: {str(e)}"}
    
    async def answer_with_reasoning(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer with detailed reasoning"""
        if not query.strip():
            logger.warning("Empty query provided for reasoning")
            return {"error": "Please provide a valid query"}
        
        context = self._prepare_context(relevant_chunks)
        
        prompt = f"""You are IntelliDoc Agent, an expert document analyst.

**CONTEXT:**
{context}

**QUESTION:** {query}

**RESPONSE FORMAT:**
- **Direct Answer**: [Concise answer]
- **Reasoning**: [Step-by-step explanation]
- **Evidence**: [Specific quotes or data with page citations]
- **Confidence**: [High/Medium/Low with explanation]
- **Insights**: [Additional relevant information]
- **Limitations**: [Gaps or constraints in the analysis]
"""
        try:
            response = await self._generate_with_retry_safe(prompt, 2000)
            sections = self._parse_structured_response(response)
            logger.info(f"Generated reasoned response for query: {query}")
            return {
                "answer": response,
                "structured_response": sections,
                "query": query,
                "sources_used": len(relevant_chunks)
            }
        except Exception as e:
            logger.error(f"Error generating reasoned response: {e}")
            return {"error": f"Error generating reasoned response: {str(e)}"}
    
    def _parse_structured_response(self, response: str) -> Dict[str, str]:
        """Parse structured response into sections"""
        sections = {}
        patterns = {
            "direct_answer": r"(?:\*\*Direct Answer:\*\*|Direct Answer)\s*:?\s*(.*?)(?=\*\*Reasoning:|\Z)",
            "reasoning": r"(?:\*\*Reasoning:\*\*|Reasoning)\s*:?\s*(.*?)(?=\*\*Evidence:|\Z)",
            "evidence": r"(?:\*\*Evidence:\*\*|Evidence)\s*:?\s*(.*?)(?=\*\*Confidence:|\Z)",
            "confidence": r"(?:\*\*Confidence:\*\*|Confidence)\s*:?\s*(.*?)(?=\*\*Insights:|\Z)",
            "insights": r"(?:\*\*Insights:\*\*|Insights)\s*:?\s*(.*?)(?=\*\*Limitations:|\Z)",
            "limitations": r"(?:\*\*Limitations:\*\*|Limitations)\s*:?\s*(.*?)(?=\Z)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                sections[key] = match.group(1).strip()
        
        return sections
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        with self._lock:
            return {
                "total_requests": self.requests_made,
                "estimated_tokens": self.total_tokens_used,
                "model_used": "gemini-1.5-flash",
                "last_request": datetime.now().isoformat(),
                "cache_size": len(self.response_cache)
            }
    
    async def batch_process_queries(self, queries: List[str], relevant_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple queries efficiently"""
        results = []
        for query in queries:
            try:
                response = await self.generate_response(query, relevant_chunks, max_tokens=500)
                results.append({
                    "query": query,
                    "response": response,
                    "status": "success"
                })
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                results.append({
                    "query": query,
                    "response": None,
                    "status": "error",
                    "error": str(e)
                })
            await asyncio.sleep(self.min_request_interval)
        return results