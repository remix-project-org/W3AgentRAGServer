import os, time
import numpy as np
import threading
import logging
from typing import List, Dict, Any

# Import the previously created classes
from src.repoManagement import GitHubRepositoryManager
from sentence_transformers import SentenceTransformer

class IntegratedRAGSystem:
    def __init__(self, 
                 data_dir: str = './data',
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 update_interval_hours: int = 24):
        """
        Initialize Integrated RAG System
        
        :param data_dir: Directory to store repository data
        :param embedding_model: Embedding model to use
        :param update_interval_hours: Interval for periodic repository updates
        """
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.repo_manager = GitHubRepositoryManager(
            data_dir=data_dir
        )
        
        self.embedding_model = SentenceTransformer(embedding_model)
        self.knowledge_base = []
        self.knowledge_embeddings = None
        
        self.repo_manager.update_repositories(self.embedding_model)
        self.refresh_knowledge_base()
        
        self.start_background_updates(update_interval_hours)
    
    def refresh_knowledge_base(self):
        try:
            self.knowledge_base = []
            
            for repo_config in self.repo_manager.repos_config:
                repo_documents = self.repo_manager.extract_text_from_repository(repo_config['local_path'])
                
                for doc in repo_documents:
                    self.knowledge_base.append({
                        'source': doc['path'],
                        'content': self.preprocess_text(doc['content'])
                    })
            
            self.knowledge_embeddings = self.compute_knowledge_embeddings()
            self.logger.info(f"Knowledge base refreshed. Total documents: {len(self.knowledge_base)}")
        
        except Exception as e:
            self.logger.error(f"Error refreshing knowledge base: {e}")
    
    def preprocess_text(self, text: str, max_length: int = 2000) -> str:
        """
        Preprocess text for embedding and retrieval
        
        :param text: Input text
        :param max_length: Maximum length of text to keep
        :return: Preprocessed text
        """
        text = ' '.join(text.split())
        
        return text[:max_length]
    
    def compute_knowledge_embeddings(self) -> np.ndarray:
        """
        Compute embeddings for knowledge base
        
        :return: NumPy array of embeddings
        """
        if not self.knowledge_base:
            return np.array([])
        
        return np.array([
            self.embedding_model.encode(entry['content'], show_progress_bar=False) 
            for entry in self.knowledge_base
        ])
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant documents based on embedding similarity
        
        :param query: User's input query
        :param top_k: Number of top results to return
        :return: List of most relevant documents
        """
        if len(self.knowledge_base) == 0 or self.knowledge_embeddings is None:
            return []
        
        query_embedding = self.embedding_model.encode(query, show_progress_bar=False)
        
        similarities = np.dot(self.knowledge_embeddings, query_embedding) / (
            np.linalg.norm(self.knowledge_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        threshold = 0.5  # Define a similarity threshold
        thr_top_indices = [i for i in similarities.argsort()[::-1] if similarities[i] >= threshold][:top_k]
        # print(f"Thresholded top indices: {thr_top_indices}")
        # print(f"Similarities thresholded: {similarities[thr_top_indices]}")
        # print(f"Top indices: {top_indices}")
        # print(f"Similarities: {similarities[top_indices]}")

        return [self.knowledge_base[i] for i in thr_top_indices]
    
    def generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Generate a response based on retrieved documents
        
        :param query: User's input query
        :param retrieved_docs: List of retrieved relevant documents
        :return: Generated response
        """
        # Simple response generation strategy
        if not retrieved_docs:
            return ""
        
        context = "\n\n".join([
            f"Source: {doc['source']}\nContent: {doc['content']}"
            for doc in retrieved_docs
        ])
        
        return context
    
    def start_background_updates(self, interval_hours: int = 24):
        def update_routine():
            while True:
                try:
                    time.sleep(interval_hours * 3600)
                    updated_repos = self.repo_manager.update_repositories(self.embedding_model)
                    if updated_repos:
                        self.logger.info(f"Repositories updated: {updated_repos}")
                        self.refresh_knowledge_base()
                
                except Exception as e:
                    self.logger.error(f"Error in update routine: {e}")
                
        
        update_thread = threading.Thread(target=update_routine, daemon=True)
        update_thread.start()

