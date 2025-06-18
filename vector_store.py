import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import faiss
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize vector store with sentence transformer model"""
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.documents = []
        self.index = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            # Fallback to a smaller model
            self.model = SentenceTransformer("all-MiniLM-L12-v2")
    
    async def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to vector store"""
        if not documents:
            return
        
        self.documents = documents
        
        # Extract text for embedding
        texts = [doc['text'] for doc in documents]
        
        # Generate embeddings
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            lambda: self.model.encode(texts, show_progress_bar=True)
        )
        
        self.embeddings = np.array(embeddings)
        
        # Create FAISS index for efficient similarity search
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.index.add(normalized_embeddings.astype('float32'))
    
    async def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self.documents or self.index is None:
            return []
        
        # Generate query embedding
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            self.executor,
            self.model.encode,
            [query]
        )
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search using FAISS
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['similarity_score'] = float(score)
                doc['rank'] = i + 1
                results.append(doc)
        
        return results
    
    async def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.7) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic and keyword matching"""
        if not self.documents:
            return []
        
        # Semantic search
        semantic_results = await self.similarity_search(query, k * 2)
        
        # Keyword search
        keyword_results = self._keyword_search(query, k * 2)
        
        # Combine results with weighted scoring
        combined_scores = {}
        
        # Add semantic scores
        for result in semantic_results:
            doc_id = result['chunk_id']
            combined_scores[doc_id] = {
                'doc': result,
                'semantic_score': result['similarity_score'],
                'keyword_score': 0
            }
        
        # Add keyword scores
        for result in keyword_results:
            doc_id = result['chunk_id']
            if doc_id in combined_scores:
                combined_scores[doc_id]['keyword_score'] = result['keyword_score']
            else:
                combined_scores[doc_id] = {
                    'doc': result,
                    'semantic_score': 0,
                    'keyword_score': result['keyword_score']
                }
        
        # Calculate hybrid scores
        final_results = []
        for doc_id, scores in combined_scores.items():
            hybrid_score = (alpha * scores['semantic_score'] + 
                          (1 - alpha) * scores['keyword_score'])
            
            doc = scores['doc'].copy()
            doc['hybrid_score'] = hybrid_score
            doc['semantic_score'] = scores['semantic_score']
            doc['keyword_score'] = scores['keyword_score']
            final_results.append(doc)
        
        # Sort by hybrid score and return top k
        final_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return final_results[:k]
    
    def _keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Simple keyword-based search"""
        query_words = set(query.lower().split())
        results = []
        
        for doc in self.documents:
            text_words = set(doc['text'].lower().split())
            
            # Calculate keyword overlap score
            overlap = len(query_words.intersection(text_words))
            if overlap > 0:
                score = overlap / len(query_words)
                doc_copy = doc.copy()
                doc_copy['keyword_score'] = score
                results.append(doc_copy)
        
        # Sort by keyword score
        results.sort(key=lambda x: x['keyword_score'], reverse=True)
        return results[:k]
    
    def save_index(self, filepath: str):
        """Save vector store to disk"""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'model_name': self.model_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        # Save FAISS index separately
        if self.index is not None:
            faiss.write_index(self.index, filepath + '.faiss')
    
    def load_index(self, filepath: str):
        """Load vector store from disk"""
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data['documents']
            self.embeddings = data['embeddings']
            
            # Load FAISS index
            faiss_path = filepath + '.faiss'
            if os.path.exists(faiss_path):
                self.index = faiss.read_index(faiss_path)
            
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            'num_documents': len(self.documents),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'model_name': self.model_name,
            'index_type': type(self.index).__name__ if self.index else None
        }