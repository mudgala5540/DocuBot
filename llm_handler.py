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
        
        # FIX: Call the new, more robust agentic prompt
        prompt = self._create_agentic_prompt(query, context) 
        
        try:
            # Generate response - CRITICAL FIX: Run in thread pool to avoid event loop issues
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
        
        # IMPROVEMENT: Sort chunks by similarity score to use the best ones first
        chunks.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)

        context_parts = []
        total_length = 0
        max_context_length = 8000 # Give the model ample context

        for chunk in chunks:
            # IMPROVEMENT: Simpler format for the LLM
            chunk_text = f"--- Context from Page {chunk.get('page', 'N/A')} ---\n{chunk['text']}\n"
            if total_length + len(chunk_text) > max_context_length:
                break
            context_parts.append(chunk_text)
            total_length += len(chunk_text)
        
        return "\n".join(context_parts)
    

    def _create_agentic_prompt(self, query: str, context: str) -> str:
        """
        Creates a resilient, multi-layered prompt that instructs the AI to first
        classify the user's intent before answering. This is the core of the agent's robustness.
        """
        prompt = f"""You are a professional, highly intelligent document analysis assistant. Your persona is helpful, concise, and strictly factual.

Your task is to follow a sequence of rules to respond to the user's query.

**--- Rules of Operation ---**

**Rule 1: Analyze the User's Query Intent.**
First, categorize the user's query into one of three types:
  a. **Chit-Chat:** Is it a simple greeting, pleasantry, or conversational filler? (e.g., "hi", "how are you?", "thanks", "who are you?")
  b. **Nonsense/Irrelevant:** Is it gibberish, a random string of characters, or a topic completely unrelated to business documents? (e.g., "asdfasdf", "what is the color of the sky?", "tell me a joke")
  c. **Document-Related Query:** Is it a specific question that could plausibly be answered by the provided document context?

**Rule 2: Formulate Your Response Based on Intent.**

*   **If the intent is Chit-Chat:**
    - Respond politely and briefly as an AI assistant.
    - DO NOT use the document context.
    - DO NOT mention the documents.
    - Example: If user says "thank you", you say "You're welcome! How else can I help?"

*   **If the intent is Nonsense/Irrelevant:**
    - State clearly and politely that your function is to answer questions about the uploaded documents.
    - DO NOT attempt to answer the irrelevant question.
    - Example: "I can only answer questions related to the documents you've provided. Please ask a question about their content."

*   **If the intent is a Document-Related Query:**
    - Scrutinize the provided "DOCUMENT CONTEXT" below to find the answer.
    - **Your answer MUST be derived 100% from this context.** Do not use any outside knowledge.
    - If the context contains the answer, provide it clearly and concisely. Use bullet points for lists or steps.
    - **If the context DOES NOT contain the answer, you MUST state: "Based on the provided documents, I could not find an answer to that question."** Do not guess or hallucinate.
    - **CRITICAL:** After your answer, you MUST add a source citation on a new line, listing the exact page numbers you used. The format must be: `(Source: Page X, Page Y)`

**--- Execution ---**

**DOCUMENT CONTEXT( context from documents ):** 
{context}

**USER QUESTION:**
{query}

**ASSISTANT RESPONSE (Follow the rules above):**
"""
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

    async def _generate_with_retry(self, prompt: str, max_tokens: int, max_retries: int = 3) -> str:
        """Generate response with retry logic"""
        
        for attempt in range(max_retries):
            try:
                # IMPROVEMENT: Set temperature to 0.0 for maximum factuality
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.0,
                    top_p=0.95,
                    top_k=40
                )
                
                # IMPROVEMENT: Use the asynchronous version of the call for better performance
                response = await self.model.generate_content_async(
                    prompt,
                    generation_config=generation_config
                )
                
                return response.text
                
            except Exception as e:
                print(f"LLM generation failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(2 ** attempt)
        
        return "Failed to generate response after multiple attempts."
    
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