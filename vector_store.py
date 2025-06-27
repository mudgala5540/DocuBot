import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.documents = []
        self.index = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model with fallback."""
        try:
            self.model = SentenceTransformer(self.model_name, device='cpu')
            logger.info(f"Loaded model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            self.model = SentenceTransformer("all-MiniLM-L12-v2", device='cpu')
            logger.info("Loaded fallback model: all-MiniLM-L12-v2")
    
    async def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to vector store with optimized batching."""
        if not documents:
            logger.warning("No documents provided to add to vector store")
            return
        
        self.documents = documents
        texts = [doc['text'] for doc in documents]
        
        try:
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(
                self.executor,
                lambda: self.model.encode(texts, show_progress_bar=True, batch_size=128)  # Increased batch size
            )
            self.embeddings = np.array(embeddings)
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.index.add(normalized_embeddings.astype('float32'))
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            self.documents = []
            self.embeddings = None
            self.index = None
    
    async def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents with improved error handling."""
        if not query.strip():
            logger.warning("Empty query provided for similarity search")
            return []
        if not self.documents or self.index is None:
            logger.warning("No documents or index available for similarity search")
            return []
        
        try:
            loop = asyncio.get_running_loop()
            query_embedding = await loop.run_in_executor(
                self.executor,
                self.model.encode,
                [query]
            )
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc['similarity_score'] = float(score)
                    doc['rank'] = i + 1
                    results.append(doc)
            
            logger.info(f"Similarity search returned {len(results)} results for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    async def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.7) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic and keyword matching with improved scoring."""
        if not query.strip():
            logger.warning("Empty query provided for hybrid search")
            return []
        if not self.documents:
            logger.warning("No documents available for hybrid search")
            return []
        
        try:
            semantic_results = await self.similarity_search(query, k * 2)
            keyword_results = self._keyword_search(query, k * 2)
            
            combined_scores = {}
            for result in semantic_results:
                doc_id = result['chunk_id']
                combined_scores[doc_id] = {
                    'doc': result,
                    'semantic_score': result['similarity_score'],
                    'keyword_score': 0
                }
            
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
            
            final_results = []
            for doc_id, scores in combined_scores.items():
                hybrid_score = (alpha * scores['semantic_score'] + 
                              (1 - alpha) * scores['keyword_score'])
                doc = scores['doc'].copy()
                doc['hybrid_score'] = hybrid_score
                doc['semantic_score'] = scores['semantic_score']
                doc['keyword_score'] = scores['keyword_score']
                final_results.append(doc)
            
            final_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
            logger.info(f"Hybrid search returned {len(final_results[:k])} results for query: {query}")
            return final_results[:k]
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return []
    
    def _keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Simple keyword-based search with improved scoring."""
        query_words = set(query.lower().split())
        results = []
        
        for doc in self.documents:
            text_words = set(doc['text'].lower().split())
            overlap = len(query_words.intersection(text_words))
            if overlap > 0:
                score = overlap / (len(query_words) + 0.1)  # Avoid division by zero
                doc_copy = doc.copy()
                doc_copy['keyword_score'] = score
                results.append(doc_copy)
        
        results.sort(key=lambda x: x['keyword_score'], reverse=True)
        return results[:k]
    
    def save_index(self, filepath: str):
        """Save vector store to disk with improved error handling."""
        try:
            data = {
                'documents': self.documents,
                'embeddings': self.embeddings,
                'model_name': self.model_name
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            if self.index is not None:
                faiss.write_index(self.index, filepath + '.faiss')
            logger.info(f"Saved vector store to {filepath}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
    
    def load_index(self, filepath: str):
        """Load vector store from disk with improved error handling."""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Index file {filepath} not found")
                return False
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.documents = data['documents']
            self.embeddings = data['embeddings']
            faiss_path = filepath + '.faiss'
            if os.path.exists(faiss_path):
                self.index = faiss.read_index(faiss_path)
            logger.info(f"Loaded vector store from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            'num_documents': len(self.documents),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'model_name': self.model_name,
            'index_type': type(self.index).__name__ if self.index else None
        }