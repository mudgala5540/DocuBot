import google.generativeai as genai
import os
from typing import List, Dict, Any
import asyncio
import json
import re
from datetime import datetime
import threading

class LLMHandler:
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=self.api_key)
        
        # Use Gemini Flash for cost-effectiveness
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Token tracking for cost monitoring
        self.total_tokens_used = 0
        self.requests_made = 0
        
        # Thread lock for thread safety
        self._lock = threading.Lock()
    
    async def generate_response(self, query: str, relevant_chunks: List[Dict[str, Any]], max_tokens: int = 2000) -> str:
        """Generate response using relevant document chunks"""
        
        # Prepare context from relevant chunks
        context = self._prepare_context(relevant_chunks)
        
        # Call the enhanced agentic prompt
        prompt = self._create_enhanced_agentic_prompt(query, context) 
        
        try:
            # Generate response
            response = await self._generate_with_retry_safe(prompt, max_tokens)
            
            # Track usage
            with self._lock:
                self.requests_made += 1
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _prepare_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Prepare context, prioritizing higher-scored chunks."""
        if not chunks:
            return "No relevant context found."
        
        # Sort chunks by similarity score to use the best ones first
        chunks.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)

        context_parts = []
        total_length = 0
        max_context_length = 8000

        for chunk in chunks:
            # Enhanced format with more metadata
            chunk_text = f"--- Document Content from Page {chunk.get('page', 'N/A')} ---\n{chunk['text']}\n"
            if total_length + len(chunk_text) > max_context_length:
                break
            context_parts.append(chunk_text)
            total_length += len(chunk_text)
        
        return "\n".join(context_parts)
    
    def _create_enhanced_agentic_prompt(self, query: str, context: str) -> str:
        """
        Enhanced prompt with better intent classification and stricter source citation requirements
        """
        prompt = f"""You are IntelliDoc Agent, a professional document analysis assistant. You are highly intelligent, precise, and strictly factual.

**CRITICAL INSTRUCTIONS - FOLLOW EXACTLY:**

**Step 1: Classify the User's Query Intent**
Determine if the user's query is:
  a. **Casual/Greeting:** Simple greetings, thanks, small talk (e.g., "hi", "hello", "thank you", "how are you?")
  b. **Irrelevant/Nonsense:** Random text, gibberish, or topics unrelated to documents (e.g., "asdfgh", "what's the weather?", "tell me a joke")
  c. **Document Question:** A genuine question that could be answered from the provided document content

**Step 2: Respond Based on Intent**

**For Casual/Greeting queries:**
- Respond politely and briefly as an AI assistant
- DO NOT mention documents or sources
- Example: For "thank you" → "You're welcome! How can I help you with your documents?"

**For Irrelevant/Nonsense queries:**
- Politely redirect to document-related questions
- Example: "I'm designed to answer questions about your uploaded documents. Please ask something related to their content."

**For Document Questions:**
- You MUST answer ONLY using the provided document context below
- If the context contains relevant information, provide a clear, comprehensive answer
- **MANDATORY:** You MUST end your response with exact source citations in this format: (Source: Page X, Page Y, Page Z)
- List ALL pages that contain information relevant to your answer
- If the context does NOT contain the answer, respond EXACTLY: "Based on the provided documents, I could not find an answer to that question."
- Never guess or use external knowledge - only use the provided context

**DOCUMENT CONTEXT:**
{context}

**USER QUERY:** {query}

**YOUR RESPONSE:**"""
        return prompt

    async def _generate_with_retry_safe(self, prompt: str, max_tokens: int, max_retries: int = 3) -> str:
        """Generate response with retry logic - SAFE VERSION"""
        
        def sync_generate():
            """Synchronous generation function to run in thread pool"""
            import google.generativeai as genai_sync
            
            # Configure in this thread
            genai_sync.configure(api_key=self.api_key)
            model_sync = genai_sync.GenerativeModel('gemini-1.5-flash')
            
            for attempt in range(max_retries):
                try:
                    generation_config = genai_sync.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.0,
                        top_p=0.95,
                        top_k=40
                    )
                    
                    response = model_sync.generate_content(
                        prompt,
                        generation_config=generation_config
                    )
                    
                    return response.text
                    
                except Exception as e:
                    print(f"LLM generation failed on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise e
                    import time
                    time.sleep(2 ** attempt)
            
            return "Failed to generate response after multiple attempts."
        
        # Run the synchronous function in a thread pool
        loop = asyncio.get_event_loop()
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(executor, sync_generate)
            return result

    async def summarize_document(self, chunks: List[Dict[str, Any]]) -> str:
        """Generate a summary of the entire document"""
        
        # Select representative chunks
        sample_chunks = chunks[:10]  # First 10 chunks for overview
        context = self._prepare_context(sample_chunks)
        
        prompt = f"""Analyze the following document content and provide a comprehensive summary:

DOCUMENT CONTENT:
{context}

Please provide:
1. Main topics and themes
2. Key findings or conclusions
3. Document structure and organization
4. Important data or statistics mentioned
5. Any notable images or visual elements referenced

Keep the summary concise but informative."""

        try:
            response = await self._generate_with_retry_safe(prompt, 800)
            return response
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    async def extract_key_information(self, chunks: List[Dict[str, Any]], info_type: str = "general") -> Dict[str, Any]:
        """Extract specific types of information from documents"""
        
        context = self._prepare_context(chunks[:15])  # Use more chunks for extraction
        
        if info_type == "financial":
            prompt = f"""Extract financial information from the following document:

{context}

Find and extract:
- Numbers, amounts, percentages
- Financial terms and metrics
- Dates related to financial events
- Company names and financial entities

Return the information in a structured format."""

        elif info_type == "technical":
            prompt = f"""Extract technical information from the following document:

{context}

Find and extract:
- Technical specifications
- Procedures and processes
- Equipment or tool names
- Technical measurements and parameters
- Safety information

Return the information in a structured format."""

        else:  # general
            prompt = f"""Extract key information from the following document:

{context}

Find and extract:
- Important names, dates, and places
- Key facts and figures
- Main conclusions or recommendations
- Action items or next steps
- Contact information

Return the information in a structured format."""

        try:
            response = await self._generate_with_retry_safe(prompt, 600)
            return {"extracted_info": response, "info_type": info_type}
        except Exception as e:
            return {"error": f"Error extracting information: {str(e)}"}
    
    async def answer_with_reasoning(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer with step-by-step reasoning"""
        
        context = self._prepare_context(relevant_chunks)
        
        prompt = f"""You are an expert document analyst. Answer the following question with clear reasoning.

CONTEXT:
{context}

QUESTION: {query}

Please provide:
1. DIRECT ANSWER: A clear, direct answer to the question
2. REASONING: Step-by-step explanation of how you arrived at the answer
3. EVIDENCE: Specific quotes or references from the document that support your answer
4. CONFIDENCE: Rate your confidence in the answer (High/Medium/Low) and explain why
5. LIMITATIONS: Any limitations or assumptions in your answer

Format your response clearly with these sections."""

        try:
            response = await self._generate_with_retry_safe(prompt, 1200)
            
            # Parse the structured response
            sections = self._parse_structured_response(response)
            
            return {
                "answer": response,
                "structured_response": sections,
                "query": query,
                "sources_used": len(relevant_chunks)
            }
        except Exception as e:
            return {"error": f"Error generating reasoned response: {str(e)}"}
    
    def _parse_structured_response(self, response: str) -> Dict[str, str]:
        """Parse structured response into sections"""
        sections = {}
        
        patterns = {
            "direct_answer": r"(?:DIRECT ANSWER|1\..*?):(.*?)(?=(?:REASONING|2\.)|$)",
            "reasoning": r"(?:REASONING|2\..*?):(.*?)(?=(?:EVIDENCE|3\.)|$)",
            "evidence": r"(?:EVIDENCE|3\..*?):(.*?)(?=(?:CONFIDENCE|4\.)|$)",
            "confidence": r"(?:CONFIDENCE|4\..*?):(.*?)(?=(?:LIMITATIONS|5\.)|$)",
            "limitations": r"(?:LIMITATIONS|5\..*?):(.*?)$"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                sections[key] = match.group(1).strip()
        
        return sections
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for cost monitoring"""
        with self._lock:
            return {
                "total_requests": self.requests_made,
                "estimated_tokens": self.total_tokens_used,
                "model_used": "gemini-1.5-flash",
                "last_request": datetime.now().isoformat()
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
                results.append({
                    "query": query,
                    "response": None,
                    "status": "error",
                    "error": str(e)
                })
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)
        
        return results