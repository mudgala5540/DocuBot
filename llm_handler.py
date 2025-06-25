import google.generativeai as genai
import os
import asyncio
import json
import re
from datetime import datetime
import threading
from typing import List, Dict, Any, Tuple

class LLMHandler:
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=self.api_key)
        
        # Use Gemini Flash for speed and cost-effectiveness
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        self.requests_made = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        self._lock = threading.Lock()

    def _prepare_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Prepare context, prioritizing higher-scored chunks."""
        if not chunks:
            return "No relevant context found."
        
        # Chunks should already be sorted by similarity score
        context_parts = []
        total_length = 0
        # Gemini 1.5 Flash has a large context window, we can be generous
        max_context_length = 32000 

        for chunk in chunks:
            chunk_text = f"[Source: Page {chunk.get('page', 'N/A')}]\n{chunk['text']}\n---\n"
            if total_length + len(chunk_text) > max_context_length:
                break
            context_parts.append(chunk_text)
            total_length += len(chunk_text)
        
        return "\n".join(context_parts)

    def _create_agentic_prompt(self, query: str, context: str) -> str:
        """
        A highly structured, agentic prompt that forces the LLM to classify the query,
        think step-by-step, and provide a structured JSON response. This is the core
        of the agent's intelligence.
        """
        return f"""
You are IntelliDoc Agent, an expert AI assistant for document analysis. Your task is to analyze user queries and respond based *only* on the provided document context.

Follow these steps precisely:

**Step 1: Analyze the Query**
Classify the user's query into one of the following categories:
- `GREETING`: For simple hellos, thanks, or other pleasantries.
- `IRRELEVANT`: For questions completely unrelated to the document, nonsense, or gibberish.
- `DOCUMENT_QUERY`: For any question that can be answered from the document context.

**Step 2: Formulate the Response**
- For `GREETING`: Respond with a brief, friendly, and professional greeting. Do not use the document context.
- For `IRRELEVANT`: Politely state that you can only answer questions about the uploaded documents and redirect the user. Do not attempt to answer the irrelevant question.
- For `DOCUMENT_QUERY`:
    1. Carefully review the provided "DOCUMENT CONTEXT".
    2. Synthesize a comprehensive, detailed answer using *only* the information from the context.
    3. If the context is insufficient, state clearly: "Based on the provided documents, I could not find enough information to answer that question."
    4. **Crucially**, identify every page number you used to formulate your answer from the `[Source: Page X]` tags.

**Step 3: Construct the Final JSON Output**
You MUST respond with a single, valid JSON object. Do not add any text before or after the JSON. The JSON object must have the following structure:

```json
{{
  "query_type": "...",
  "thought_process": "A brief, step-by-step explanation of how you arrived at your answer.",
  "response_text": "The final, user-facing answer.",
  "cited_pages": [
    // An array of integer page numbers, e.g., [3, 7, 12].
    // This MUST be accurate and based on the [Source: Page X] tags.
    // Leave empty if the query is not a DOCUMENT_QUERY or no specific page was used.
  ]
}}
DOCUMENT CONTEXT:
{context}
USER QUERY: {query}
YOUR JSON RESPONSE:
"""

async def _generate_with_retry(self, prompt: str, max_retries: int = 3) -> genai.types.GenerateContentResponse:
    """Internal generation function with retry logic."""
    loop = asyncio.get_running_loop()
    
    def sync_generate():
        # This synchronous function will be run in a thread pool executor
        last_exception = None
        for attempt in range(max_retries):
            try:
                # Configuration for predictable JSON output
                generation_config = genai.types.GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json" # Enforce JSON output
                )
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                return response
            except Exception as e:
                last_exception = e
                print(f"LLM generation failed on attempt {attempt + 1}: {e}")
                asyncio.sleep(2 ** attempt) # Exponential backoff
        raise last_exception

    # Run the synchronous SDK call in an executor to avoid blocking the event loop
    return await loop.run_in_executor(None, sync_generate)


async def get_agentic_response(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generates a structured, agentic response, including classification, answer, and citations.
    """
    context = self._prepare_context(relevant_chunks)
    prompt = self._create_agentic_prompt(query, context)
    
    try:
        response = await self._generate_with_retry(prompt)
        
        # Update token usage stats
        with self._lock:
            self.requests_made += 1
            if response.usage_metadata:
                self.total_input_tokens += response.usage_metadata.prompt_token_count
                self.total_output_tokens += response.usage_metadata.candidate_token_count

        # Clean and parse the JSON response
        raw_text = response.text.strip()
        # The Gemini API can sometimes wrap the JSON in ```json ... ```, so we clean it.
        clean_text = re.sub(r'^```json\s*|\s*```$', '', raw_text, flags=re.DOTALL)
        parsed_response = json.loads(clean_text)
        
        return parsed_response

    except Exception as e:
        print(f"Error generating or parsing LLM response: {e}")
        # Return a fallback error structure
        return {
            "query_type": "ERROR",
            "thought_process": f"An error occurred: {str(e)}",
            "response_text": "I'm sorry, I encountered a critical error while trying to generate a response. Please try again.",
            "cited_pages": []
        }

def get_usage_stats(self) -> Dict[str, Any]:
    """Get usage statistics for cost monitoring."""
    with self._lock:
        return {
            "total_requests": self.requests_made,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "model_used": self.model.model_name,
            "last_request_time": datetime.now().isoformat()
        }
