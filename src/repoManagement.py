import os
import json
import shutil
import git
import requests
from typing import List, Dict, Any
import numpy as np
import subprocess
import time
from datetime import datetime, timedelta
import logging

class GitHubRepositoryManager:
    def __init__(self, 
                 data_dir: str = './data',
                 config_path: str = './repo_config.json'):
        """
        Initialize GitHub Repository Manager
        
        :param data_dir: Directory to store repository data
        :param config_path: Path to repository configuration file
        :param embedding_model: Embedding model to use
        """
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.data_dir = os.path.abspath(data_dir)
        self.config_path = os.path.abspath(config_path)
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.repos_config = self.load_or_create_repo_config()
        
        self.embedding_cache_dir = os.path.join(self.data_dir, 'embeddings')
        os.makedirs(self.embedding_cache_dir, exist_ok=True)
    
    def load_or_create_repo_config(self) -> List[Dict[str, Any]]:
        """
        Load existing repository configuration or create a default one
        """
        default_repos = [
            {
                "name": "openzeppelin-contracts",
                "url": "https://github.com/OpenZeppelin/openzeppelin-contracts.git",
                "local_path": os.path.join(self.data_dir, "openzeppelin-contracts"),
                "last_updated": None,
                "update_frequency_hours": 24
            },
            {
                "name": "uniswap-v2-core",
                "url": "https://github.com/Uniswap/v2-core.git",
                "local_path": os.path.join(self.data_dir, "uniswap-v2-core"),
                "last_updated": None,
                "update_frequency_hours": 24
            }
        ]
        
        if not os.path.exists(self.config_path):
            with open(self.config_path, 'w') as f:
                json.dump(default_repos, f, indent=2)
            return default_repos
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def clone_or_update_repository(self, repo_config: Dict[str, Any]) -> bool:
        try:
            local_path = repo_config['local_path']
            
            # Check if repository exists locally
            if not os.path.exists(local_path):
                self.logger.info(f"Cloning {repo_config['name']} to {local_path}")
                git.Repo.clone_from(repo_config['url'], local_path)
                return True
            
            # Update existing repository
            repo = git.Repo(local_path)
            origin = repo.remotes.origin
            
            # Check if update is needed based on last update time
            current_time = datetime.now()
            if repo_config['last_updated']:
                last_updated = datetime.fromisoformat(repo_config['last_updated'])
                update_delta = timedelta(hours=repo_config['update_frequency_hours'])
                
                if current_time - last_updated < update_delta:
                    self.logger.info(f"Skipping update for {repo_config['name']} - not time yet")
                    return False
            
            # Pull latest changes
            origin.pull()
            self.logger.info(f"Updated {repo_config['name']}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error updating {repo_config['name']}: {e}")
            return False
    
    def extract_text_from_repository(self, repo_path: str) -> List[Dict[str, str]]:
        documents = []
        
        text_extensions = ['.sol', '.md', '.txt', '.adoc']
        for root, _, files in os.walk(repo_path):
            for file in files:
                if any(file.endswith(ext) for ext in text_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            documents.append({
                                'path': os.path.relpath(file_path, self.data_dir),
                                'content': content
                            })
                    except Exception as e:
                        self.logger.warning(f"Could not read {file_path}: {e}")
        
        return documents
    
    def compute_repository_embeddings(self, repo_config: Dict[str, Any], embedding_model) -> np.ndarray:
        embedding_cache_file = os.path.join(
            self.embedding_cache_dir, 
            f"{repo_config['name']}_embeddings.npy"
        )
        
        if os.path.exists(embedding_cache_file):
            return np.load(embedding_cache_file)
        
        documents = self.extract_text_from_repository(repo_config['local_path'])
        
        embeddings = embedding_model.encode([
            doc['content'] for doc in documents
            ], 
            show_progress_bar=False
        )
        
        np.save(embedding_cache_file, embeddings)
        return embeddings
    
    def update_repositories(self, embedding_model):
        updated_repos = []
        
        for repo_config in self.repos_config:
            # Update repository
            if self.clone_or_update_repository(repo_config):
                # Recompute embeddings if updated
                self.compute_repository_embeddings(repo_config, embedding_model)
                
                # Update last updated timestamp
                repo_config['last_updated'] = datetime.now().isoformat()
                updated_repos.append(repo_config['name'])
        
        # Save updated configuration
        with open(self.config_path, 'w') as f:
            json.dump(self.repos_config, f, indent=2)
        
        return updated_repos
    
    def run_periodic_updates(self, interval_hours: int = 24):
        """
        Run periodic updates in the background
        
        :param interval_hours: Hours between update checks
        """
        while True:
            time.sleep(interval_hours * 3600)  # Convert hours to seconds
            self.update_repositories()

def main():
    # Create repository manager
    repo_manager = GitHubRepositoryManager()
    
    # Perform initial update
    updated_repos = repo_manager.update_repositories()
    print(f"Updated repositories: {updated_repos}")

if __name__ == '__main__':
    main()