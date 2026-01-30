# embedding_model.py
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import re
import pickle
import hashlib
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Union
import logging

class EmbeddingModel:
    """Handles text embedding, caching, and JSON parsing operations"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "embedding_cache"):
        self.model_name = model_name
        self.model = None
        self.messages = []  # Store messages for Streamlit
        self.cache_dir = cache_dir
        self._embedding_cache = {}  # In-memory cache
        self._initialize_cache_dir()
        self._initialize_model()
    
    def _initialize_cache_dir(self):
        """Initialize cache directory if it doesn't exist"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._add_message(f"📁 Cache directory: {os.path.abspath(self.cache_dir)}")
        except Exception as e:
            self._add_message(f"⚠️ Could not create cache directory: {e}")
    
    def _initialize_model(self):
        """Initialize the embedding model"""
        try:
            self._add_message("🔄 Loading embedding model...")
            self.model = SentenceTransformer(self.model_name)
            self._add_message(f"✅ Loaded model: {self.model_name}")
        except Exception as e:
            self._add_message(f"❌ Error loading model {self.model_name}: {e}")
            raise
    
    def _add_message(self, message: str):
        """Add a message to the messages list"""
        self.messages.append(message)
        print(message)
    
    def clear_messages(self):
        """Clear all stored messages"""
        self.messages = []
    
    def get_messages(self) -> List[str]:
        """Get all stored messages"""
        return self.messages
    
    def _get_cache_key(self, texts: List[str]) -> str:
        """
        Generate a unique cache key for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            MD5 hash string as cache key
        """
        # Create a unique string from texts and model name
        text_string = "|||".join(sorted(texts))  # Sort for consistent ordering
        cache_string = f"{self.model_name}|||{text_string}"
        
        # Generate MD5 hash
        return hashlib.md5(cache_string.encode('utf-8')).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Load embeddings from cache
        
        Args:
            cache_key: Cache key to lookup
            
        Returns:
            Tuple of (embeddings, metadata) or None if not found
        """
        # Check in-memory cache first
        if cache_key in self._embedding_cache:
            self._add_message(f"📦 Loading from in-memory cache (key: {cache_key[:8]}...)")
            return self._embedding_cache[cache_key]
        
        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Validate cached data
                if self.validate_embeddings(cached_data['embeddings']):
                    self._add_message(f"📂 Loaded from disk cache (key: {cache_key[:8]}...)")
                    
                    # Also store in memory for faster future access
                    self._embedding_cache[cache_key] = (
                        cached_data['embeddings'], 
                        cached_data['metadata']
                    )
                    
                    return cached_data['embeddings'], cached_data['metadata']
                else:
                    self._add_message("⚠️ Invalid cached embeddings, recomputing...")
                    os.remove(cache_file)  # Remove invalid cache file
            except Exception as e:
                self._add_message(f"⚠️ Error loading cache file {cache_file}: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, embeddings: np.ndarray, texts: List[str]):
        """
        Save embeddings to cache
        
        Args:
            cache_key: Cache key
            embeddings: Embeddings to cache
            texts: Original texts
        """
        try:
            # Prepare metadata
            metadata = {
                'model_name': self.model_name,
                'text_count': len(texts),
                'embedding_dim': embeddings.shape[1],
                'timestamp': datetime.now().isoformat(),
                'text_sample': texts[0][:100] if texts else ""  # Store sample for identification
            }
            
            # Save to in-memory cache
            self._embedding_cache[cache_key] = (embeddings, metadata)
            
            # Save to disk cache
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            cache_data = {
                'embeddings': embeddings,
                'metadata': metadata,
                'texts': texts  # Optionally store texts if needed
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            self._add_message(f"💾 Saved to cache (key: {cache_key[:8]}...)")
            
        except Exception as e:
            self._add_message(f"⚠️ Could not save to cache: {e}")
    
    def encode_texts(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None,
        show_progress_bar: bool = True,
        use_cache: bool = True,
        force_recompute: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Encode a list of texts into embeddings with caching
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding (None for auto)
            show_progress_bar: Whether to show progress bar
            use_cache: Whether to use caching
            force_recompute: Force recomputation even if cached
            
        Returns:
            Tuple of (embeddings, messages)
        """
        self.clear_messages()  # Clear previous messages
        
        if not texts:
            self._add_message("❌ No texts provided for encoding")
            return np.array([]), self.messages
        
        if self.model is None:
            self._add_message("❌ Model not initialized")
            return np.array([]), self.messages
        
        # Check cache if enabled and not forcing recomputation
        if use_cache and not force_recompute:
            cache_key = self._get_cache_key(texts)
            cached_result = self._load_from_cache(cache_key)
            
            if cached_result is not None:
                embeddings, metadata = cached_result
                self._add_message(f"✅ Loaded {len(embeddings)} embeddings from cache")
                self._add_message(f"   Model: {metadata['model_name']}")
                self._add_message(f"   Dimension: {metadata['embedding_dim']}")
                self._add_message(f"   Cached: {metadata['timestamp']}")
                return embeddings, self.messages
        
        # Not in cache or force recompute
        self._add_message(f"🔄 Encoding {len(texts)} texts...")
        
        try:
            # Try encoding with default settings first
            embeddings = self.model.encode(
                texts, 
                show_progress_bar=show_progress_bar, 
                convert_to_numpy=True,
                batch_size=batch_size,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            self._add_message(f"✅ Encoded {len(embeddings)} texts successfully")
            
        except Exception as e:
            self._add_message(f"❌ Error during encoding: {e}")
            
            if batch_size is None:
                self._add_message("🔄 Trying with smaller batch size...")
                # Encode in batches
                batch_size = 100
                embeddings_list = []
                num_batches = (len(texts) + batch_size - 1) // batch_size
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    batch_embeddings = self.model.encode(
                        batch, 
                        show_progress_bar=False, 
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                    embeddings_list.append(batch_embeddings)
                    self._add_message(f"  Encoded batch {i//batch_size + 1}/{num_batches}")
                
                embeddings = np.vstack(embeddings_list)
                self._add_message(f"✅ Encoded {len(embeddings)} texts in batches")
            else:
                # Re-raise if batch_size was specified and still failed
                raise
        
        # Save to cache if enabled
        if use_cache:
            cache_key = self._get_cache_key(texts)
            self._save_to_cache(cache_key, embeddings, texts)
        
        return embeddings, self.messages
    
    def compute_similarity(
        self, 
        query_embedding: np.ndarray, 
        corpus_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and corpus embeddings
        
        Args:
            query_embedding: Single embedding vector (already normalized)
            corpus_embeddings: Matrix of corpus embeddings (already normalized)
            
        Returns:
            Array of similarity scores
        """
        # Both embeddings should already be normalized (we set normalize_embeddings=True)
        # Compute cosine similarity directly
        similarities = np.dot(corpus_embeddings, query_embedding)
        
        return similarities
    
    def find_similar(
        self,
        query: str,
        corpus_texts: List[str],
        corpus_embeddings: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ) -> Tuple[List[Dict], List[str]]:
        """
        Find similar texts to a query
        
        Args:
            query: Query text
            corpus_texts: List of corpus texts
            corpus_embeddings: Pre-computed embeddings for corpus
            top_k: Number of similar items to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            Tuple of (similar_items, messages)
        """
        self.clear_messages()
        
        if not corpus_texts or len(corpus_embeddings) == 0:
            self._add_message("❌ No corpus data available")
            return [], self.messages
        
        self._add_message(f"🔍 Finding similar items for query: '{query[:50]}...'")
        
        # Encode query (with caching)
        query_embedding, _ = self.encode_texts([query], show_progress_bar=False)
        if len(query_embedding) == 0:
            return [], self.messages
        
        # Compute similarities
        similarities = self.compute_similarity(query_embedding[0], corpus_embeddings)
        
        # Apply threshold
        valid_indices = np.where(similarities >= similarity_threshold)[0]
        
        if len(valid_indices) == 0:
            self._add_message(f"⚠️ No items above similarity threshold {similarity_threshold}")
            return [], self.messages
        
        # Get top-k indices from valid ones
        top_k = min(top_k, len(valid_indices))
        top_valid_indices = valid_indices[np.argsort(similarities[valid_indices])[-top_k:][::-1]]
        
        # Prepare results
        results = []
        for idx in top_valid_indices:
            similarity_score = float(similarities[idx])
            results.append({
                "index": int(idx),
                "text": corpus_texts[idx],
                "similarity": similarity_score,
                "similarity_percent": round(similarity_score * 100, 2)
            })
        
        self._add_message(f"✅ Found {len(results)} similar items (threshold: {similarity_threshold})")
        
        return results, self.messages
    
    def save_embeddings(
        self, 
        embeddings: np.ndarray, 
        texts: List[str], 
        file_path: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Save embeddings and texts to disk
        
        Args:
            embeddings: Embeddings array
            texts: Corresponding texts
            file_path: Path to save file
            metadata: Additional metadata to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if metadata is None:
                metadata = {}
            
            data = {
                'embeddings': embeddings,
                'texts': texts,
                'model_name': self.model_name,
                'embedding_dim': embeddings.shape[1] if len(embeddings) > 0 else 0,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            self._add_message(f"💾 Saved embeddings to {file_path}")
            return True
            
        except Exception as e:
            self._add_message(f"❌ Error saving embeddings: {e}")
            return False
    
    def load_embeddings(self, file_path: str) -> Tuple[np.ndarray, List[str], Dict]:
        """
        Load embeddings and texts from disk
        
        Args:
            file_path: Path to saved embeddings file
            
        Returns:
            Tuple of (embeddings, texts, metadata)
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            embeddings = data['embeddings']
            texts = data['texts']
            metadata = data.get('metadata', {})
            
            # Validate loaded data
            if not self.validate_embeddings(embeddings):
                raise ValueError("Invalid embeddings in file")
            
            if len(embeddings) != len(texts):
                raise ValueError("Mismatch between embeddings and texts count")
            
            self._add_message(f"📂 Loaded {len(embeddings)} embeddings from {file_path}")
            return embeddings, texts, metadata
            
        except Exception as e:
            self._add_message(f"❌ Error loading embeddings: {e}")
            return np.array([]), [], {}
    
    def clear_cache(self, cache_type: str = "all") -> int:
        """
        Clear cache entries
        
        Args:
            cache_type: "memory", "disk", or "all"
            
        Returns:
            Number of cache entries cleared
        """
        count = 0
        
        if cache_type in ["memory", "all"]:
            count += len(self._embedding_cache)
            self._embedding_cache.clear()
            self._add_message(f"🧹 Cleared {count} in-memory cache entries")
        
        if cache_type in ["disk", "all"]:
            disk_count = 0
            try:
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.pkl'):
                        os.remove(os.path.join(self.cache_dir, filename))
                        disk_count += 1
                self._add_message(f"🧹 Cleared {disk_count} disk cache entries")
                count += disk_count
            except Exception as e:
                self._add_message(f"⚠️ Error clearing disk cache: {e}")
        
        return count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'memory_cache_size': len(self._embedding_cache),
            'model_name': self.model_name,
            'cache_dir': self.cache_dir
        }
        
        # Disk cache stats
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
            stats['disk_cache_size'] = len(cache_files)
            
            # Calculate total disk usage
            total_size = 0
            for filename in cache_files:
                filepath = os.path.join(self.cache_dir, filename)
                total_size += os.path.getsize(filepath)
            stats['disk_cache_bytes'] = total_size
            
        except Exception as e:
            stats['disk_cache_error'] = str(e)
        
        return stats
    
    @staticmethod
    def safe_json_parse(text: str) -> Dict:
        """
        Safely parse JSON from LLM response
        
        Args:
            text: Text containing JSON
            
        Returns:
            Parsed JSON as dictionary
        """
        if not text:
            return {}
        
        # Remove code blocks
        text = re.sub(r"```json\s*|```\s*", "", text).strip()
        
        # Try to find and fix common JSON issues
        # Fix missing quotes around keys
        text = re.sub(r'(\w+)\s*:', r'"\1":', text)
        
        # Fix trailing commas
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        # Try to parse as JSON
        try:
            parsed = json.loads(text)
            return parsed
        except json.JSONDecodeError as e:
            # Try to find JSON object/array
            json_pattern = r'(\{.*\}|\[.*\])'
            matches = re.findall(json_pattern, text, re.DOTALL)
            
            if matches:
                try:
                    return json.loads(matches[0])
                except:
                    # Last attempt: try eval for simple dicts (be careful!)
                    try:
                        # Only use eval for trusted content in development
                        import ast
                        parsed = ast.literal_eval(matches[0])
                        if isinstance(parsed, (dict, list)):
                            return parsed
                    except:
                        return {}
        
        return {}
    
    @staticmethod
    def validate_embeddings(embeddings: np.ndarray) -> bool:
        """
        Validate embeddings array
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(embeddings, np.ndarray):
            return False
        
        if len(embeddings.shape) != 2:
            return False
        
        if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            return False
        
        # Check if embeddings are normalized (optional)
        # norms = np.linalg.norm(embeddings, axis=1)
        # if not np.allclose(norms, 1.0, atol=1e-6):
        #     return False
        
        return True
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model
        
        Returns:
            Embedding dimension
        """
        if self.model is None:
            return 0
        
        # Test with a dummy sentence to get dimension
        test_embedding, _ = self.encode_texts(["test"], show_progress_bar=False, use_cache=False)
        return test_embedding.shape[1] if len(test_embedding) > 0 else 0
    
    def batch_find_similar(
        self,
        queries: List[str],
        corpus_texts: List[str],
        corpus_embeddings: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ) -> Tuple[List[List[Dict]], List[str]]:
        """
        Find similar texts for multiple queries
        
        Args:
            queries: List of query texts
            corpus_texts: List of corpus texts
            corpus_embeddings: Pre-computed embeddings for corpus
            top_k: Number of similar items to return per query
            similarity_threshold: Minimum similarity score
            
        Returns:
            Tuple of (list_of_results_per_query, messages)
        """
        self.clear_messages()
        
        if not queries:
            self._add_message("❌ No queries provided")
            return [], self.messages
        
        self._add_message(f"🔍 Finding similar items for {len(queries)} queries...")
        
        # Encode all queries at once (with caching)
        query_embeddings, _ = self.encode_texts(queries, show_progress_bar=False)
        
        if len(query_embeddings) == 0:
            return [], self.messages
        
        all_results = []
        
        for i, query in enumerate(queries):
            query_embedding = query_embeddings[i]
            similarities = self.compute_similarity(query_embedding, corpus_embeddings)
            
            # Apply threshold
            valid_indices = np.where(similarities >= similarity_threshold)[0]
            
            if len(valid_indices) > 0:
                top_k_current = min(top_k, len(valid_indices))
                top_indices = valid_indices[np.argsort(similarities[valid_indices])[-top_k_current:][::-1]]
                
                query_results = []
                for idx in top_indices:
                    similarity_score = float(similarities[idx])
                    query_results.append({
                        "index": int(idx),
                        "text": corpus_texts[idx],
                        "similarity": similarity_score,
                        "similarity_percent": round(similarity_score * 100, 2)
                    })
                all_results.append(query_results)
            else:
                all_results.append([])
        
        self._add_message(f"✅ Processed {len(queries)} queries")
        return all_results, self.messages