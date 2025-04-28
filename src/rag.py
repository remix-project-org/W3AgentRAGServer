import os, time
import numpy as np
import threading
import logging
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai.llms import OpenAI

from dotenv import load_dotenv
load_dotenv()


from src.repoManagement import GitHubRepositoryManager
class IntegratedRAGSystem:
    def __init__(self, 
                 data_dir: str = './data',
                 embedding_model: str = 'text-embedding-ada-002',
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
        
        self.repo_manager = GitHubRepositoryManager(data_dir=data_dir)
        self.repo_manager.dispatcher.on("new_vector_store", self.handle_new_vector_store)
        
        self.vector_store = self.load_vector_store()
        self.retriever = self.initialize_qa_chain()

        self.refresh_knowledge_base()
        self.start_background_updates(update_interval_hours)
    
    def handle_new_vector_store(self, vector_store: FAISS):
        # especialy for updates
        print("Event New vector store created")
        self.vector_store = vector_store

    def load_vector_store(self) -> FAISS:
        """
        Load the FAISS vector store from disk.
        """
        if os.path.exists(self.repo_manager.vector_store_path):
            return FAISS.load_local(self.repo_manager.vector_store_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        else:
            return self.repo_manager.build_vector_store()
    
    def initialize_qa_chain(self) -> RetrievalQA:
        """
        Initialize the RetrievalQA chain using LangChain.
        """
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5}) if self.vector_store else None
        #llm = OpenAI(model="gpt-4-turbo")
        #return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        return retriever
    
    def refresh_knowledge_base(self):
        try:
            for repo_config in self.repo_manager.repos_config:
                self.repo_manager.clone_or_update_repository(repo_config)
            self.logger.info(f"Knowledge base refreshed")
        except Exception as e:
            self.logger.error(f"Error refreshing knowledge base: {e}")

    def retrieve_relevant_documents(self, query: str):
        """
        Retrieve relevant documents 
        """
        try:
            retrieved_docs = self.retriever.get_relevant_documents(query)
            if not retrieved_docs:
                self.logger.info("No relevant documents found")
                return []
            
            # Generate response based on retrieved documents
            response = [{"content": doc.page_content, "source": os.path.relpath(doc.metadata.get("source", ""), self.repo_manager.data_dir)} for doc in retrieved_docs]
            return response
        
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}")
            return []
    
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

