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
        
        # Create the enhanced agentic prompt
        prompt = self._create_enhanced_agentic_prompt(query, context) 
        
        try:
            # Generate response - Run in thread pool to avoid event loop issues
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
        max_context_length = 10000  # Increased context length for better responses

        for chunk in chunks:
            # Include page numbers for better source tracking
            chunk_text = f"[PAGE {chunk.get('page', 'N/A')}]\n{chunk['text']}\n"
            if total_length + len(chunk_text) > max_context_length:
                break
            context_parts.append(chunk_text)
            total_length += len(chunk_text)
        
        return "\n".join(context_parts)
    
    def _create_enhanced_agentic_prompt(self, query: str, context: str) -> str:
        """
        Enhanced agentic prompt that better handles query classification and provides
        more comprehensive responses with proper source citation.
        """
        prompt = f"""You are IntelliDoc Agent, a professional document analysis AI assistant. You are highly intelligent, helpful, and always provide accurate information based solely on the provided documents.

**CRITICAL INSTRUCTION: Follow this exact process for every query:**

**STEP 1: ANALYZE THE USER'S QUERY TYPE**

Classify the query into one of these categories:
- **GREETING/CASUAL**: Simple greetings, thanks, pleasantries (e.g., "hi", "hello", "thank you", "how are you")
- **IRRELEVANT/NONSENSE**: Random strings, completely unrelated topics, gibberish (e.g., "asdfgh", "what's the weather", "tell me a joke")  
- **DOCUMENT_QUERY**: Any question that could potentially be answered using the provided document context

**STEP 2: RESPOND BASED ON QUERY TYPE**

**For GREETING/CASUAL queries:**
- Respond politely and professionally as an AI assistant
- DO NOT mention or use the document context
- Keep response brief and friendly
- Example: "Hello! I'm here to help you with questions about your documents. What would you like to know?"

**For IRRELEVANT/NONSENSE queries:**
- Politely redirect to document-related topics
- DO NOT attempt to answer the irrelevant question
- Example: "I specialize in answering questions about the documents you've uploaded. Please ask me something about their content."

**For DOCUMENT_QUERY:**
- Use ONLY the provided document context below to answer
- Provide comprehensive, detailed answers when the information is available
- If the context contains the answer, include:
  * Clear, well-structured response
  * Specific details and data points
  * Bullet points or numbered lists when appropriate
  * Professional tone with confidence
- If the context does NOT contain sufficient information, state clearly: "Based on the provided documents, I could not find enough information to answer that question."
- ALWAYS end with source citation: (Source: Page X, Page Y, Page Z) listing ALL pages referenced
- Be thorough - if there are multiple relevant sections, include information from all of them

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
                        temperature=0.1,  # Slightly increased for more natural responses
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
        """Generate a comprehensive summary of the entire document"""
        
        # Select representative chunks from throughout the document
        sample_chunks = chunks[:15]  # Increased for better coverage
        context = self._prepare_context(sample_chunks)
        
        prompt = f"""Analyze the following document content and provide a comprehensive summary:

DOCUMENT CONTENT:
{context}

Please provide a detailed summary including:

1. **Document Overview**: What type of document is this and what is its main purpose?

2. **Key Topics and Themes**: What are the main subjects covered?

3. **Important Findings**: What are the key conclusions, results, or recommendations?

4. **Critical Data**: Any important numbers, statistics, dates, or metrics mentioned

5. **Structure and Organization**: How is the document organized and what are its main sections?

6. **Visual Elements**: Any charts, diagrams, tables, or images referenced

7. **Action Items**: Any next steps, recommendations, or calls to action

Make the summary comprehensive but well-organized and easy to read."""

        try:
            response = await self._generate_with_retry_safe(prompt, 1200)
            return response
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    async def extract_key_information(self, chunks: List[Dict[str, Any]], info_type: str = "general") -> Dict[str, Any]:
        """Extract specific types of information from documents"""
        
        context = self._prepare_context(chunks[:20])  # Use more chunks for extraction
        
        if info_type == "financial":
            prompt = f"""Extract all financial information from the following document:

{context}

Find and extract:
- **Monetary amounts**: All dollar amounts, costs, revenues, budgets
- **Financial metrics**: ROI, profit margins, percentages, ratios
- **Financial dates**: Fiscal years, reporting periods, deadlines
- **Financial entities**: Company names, banks, financial institutions
- **Performance data**: Growth rates, comparisons, trends
- **Budget items**: Line items, allocations, expenditures

Format the response with clear categories and specific details."""

        elif info_type == "technical":
            prompt = f"""Extract all technical information from the following document:

{context}

Find and extract:
- **Technical specifications**: Models, versions, capacities, measurements
- **Procedures and processes**: Step-by-step instructions, workflows
- **Equipment and tools**: Names, models, specifications
- **Technical parameters**: Settings, configurations, tolerances
- **Safety information**: Warnings, precautions, requirements
- **Standards and compliance**: Regulations, certifications, standards

Format the response with clear categories and specific technical details."""

        else:  # general
            prompt = f"""Extract all key information from the following document:

{context}

Find and extract:
- **Important people**: Names, titles, roles, contact information
- **Key dates**: Deadlines, milestones, schedules, timelines
- **Locations**: Addresses, facilities, geographic references
- **Critical numbers**: Statistics, quantities, measurements, percentages
- **Main conclusions**: Key findings, recommendations, decisions
- **Action items**: Tasks, next steps, responsibilities
- **Contact details**: Phone numbers, emails, addresses

Format the response with clear categories and comprehensive details."""

        try:
            response = await self._generate_with_retry_safe(prompt, 800)
            return {"extracted_info": response, "info_type": info_type}
        except Exception as e:
            return {"error": f"Error extracting information: {str(e)}"}
    
    async def answer_with_reasoning(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer with detailed step-by-step reasoning"""
        
        context = self._prepare_context(relevant_chunks)
        
        prompt = f"""You are an expert document analyst. Answer the following question with comprehensive reasoning and analysis.

CONTEXT:
{context}

QUESTION: {query}

Please provide a detailed analysis with these sections:

**1. DIRECT ANSWER:**
Provide a clear, direct answer to the question.

**2. DETAILED REASONING:**
Explain step-by-step how you arrived at this answer, including:
- What information you found in the documents
- How different pieces of information connect
- Any patterns or trends you identified
- Logical deductions made from the evidence

**3. SUPPORTING EVIDENCE:**
List specific quotes, data points, or references from the document that support your answer.

**4. CONFIDENCE ASSESSMENT:**
Rate your confidence level (High/Medium/Low) and explain:
- Why you have this confidence level
- What factors support or limit your certainty
- Any assumptions you had to make

**5. ADDITIONAL INSIGHTS:**
Any related information, context, or implications that might be valuable.

**6. LIMITATIONS:**
Any gaps in the available information or limitations in your analysis.

Format your response clearly with these sections for easy reading."""

        try:
            response = await self._generate_with_retry_safe(prompt, 1500)
            
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
        """Parse structured response into sections with improved regex patterns"""
        sections = {}
        
        patterns = {
            "direct_answer": r"(?:\*\*1\.\s*DIRECT ANSWER:\*\*|1\.\s*DIRECT ANSWER|DIRECT ANSWER)\s*:?\s*(.*?)(?=\*\*2\.|2\.|$)",
            "reasoning": r"(?:\*\*2\.\s*DETAILED REASONING:\*\*|2\.\s*DETAILED REASONING|DETAILED REASONING)\s*:?\s*(.*?)(?=\*\*3\.|3\.|$)",
            "evidence": r"(?:\*\*3\.\s*SUPPORTING EVIDENCE:\*\*|3\.\s*SUPPORTING EVIDENCE|SUPPORTING EVIDENCE)\s*:?\s*(.*?)(?=\*\*4\.|4\.|$)",
            "confidence": r"(?:\*\*4\.\s*CONFIDENCE ASSESSMENT:\*\*|4\.\s*CONFIDENCE ASSESSMENT|CONFIDENCE ASSESSMENT)\s*:?\s*(.*?)(?=\*\*5\.|5\.|$)",
            "insights": r"(?:\*\*5\.\s*ADDITIONAL INSIGHTS:\*\*|5\.\s*ADDITIONAL INSIGHTS|ADDITIONAL INSIGHTS)\s*:?\s*(.*?)(?=\*\*6\.|6\.|$)",
            "limitations": r"(?:\*\*6\.\s*LIMITATIONS:\*\*|6\.\s*LIMITATIONS|LIMITATIONS)\s*:?\s*(.*?)$"
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