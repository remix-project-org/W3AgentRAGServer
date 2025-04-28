import os, time
import json
import logging
from typing import List, Dict, Any
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from datetime import datetime
from datetime import timedelta
import git

class EventDispatcher:
    def __init__(self):
        self.listeners = {}

    def on(self, event_name, callback):
        self.listeners.setdefault(event_name, []).append(callback)

    def emit(self, event_name, *args, **kwargs):
        for callback in self.listeners.get(event_name, []):
            callback(*args, **kwargs)

class GitHubRepositoryManager:
    def __init__(self, 
                 data_dir: str = './data',
                 config_path: str = './repo_config.json'):
        """
        Initialize GitHub Repository Manager
        
        :param data_dir: Directory to store repository data
        :param config_path: Path to repository configuration file
                """
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.dispatcher = EventDispatcher()
        self.data_dir = os.path.abspath(data_dir)
        self.config_path = os.path.abspath(config_path)
        self.vector_store_path = os.path.join(data_dir, "vector_store")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.repos_config = self.load_or_create_repo_config()
        self.update_repositories()
    
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

                if 'release_tag' in repo_config and repo_config['release_tag']:
                    repo = git.Repo(local_path)
                    repo.git.fetch('--tags')
                    repo.git.checkout(repo_config['release_tag'])
                    self.logger.info(f"Checked out release tag {repo_config['release_tag']} for {repo_config['name']}")
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
            
            # Pull latest changes or checkout release tag if specified
            if repo_config.get('release_tag'):
                if repo_config['release_tag'] not in repo.git.tag():
                    origin.pull()
                    self.logger.info(f"Release tag {repo_config['release_tag']} not found, pulling latest changes")
                    return True
                self.logger.info(f"Checking out release tag {repo_config['release_tag']} for {repo_config['name']}")
                repo.git.fetch('--tags')
                repo.git.checkout(repo_config['release_tag'])
            else:
                origin.pull()
                self.logger.info(f"Updated {repo_config['name']}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error updating {repo_config['name']}: {e}")
            return False
    
    def extract_text_from_repository(self, repo_path: str) -> List[Dict[str, str]]:
        documents = []
        text_extensions = ['.mdx', '.md', '.txt', '.adoc']
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
        
    def build_vector_store(self) -> FAISS:
        """
        Build a FAISS vector .
        """
        print("################### build vector store with data path", self.data_dir)
        extensions = ["mdx", "md", "txt", "adoc"]
        documents = []

        for ext in extensions:
            print(f"Loading files with extension: {ext}")
            loader = DirectoryLoader(
                path=self.data_dir,
                glob=f"**/*.{ext}",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                show_progress=True,
                recursive=True
            )
            documents.extend(loader.load())

        self.logger.info(f"Loaded {len(documents)} documents from path {self.data_dir}")
        
        embeddings = OpenAIEmbeddings()
        import faiss
        faiss.omp_set_num_threads(1)  # Optional: Limit FAISS to a single thread
        vector_store = FAISS.from_documents(documents, embeddings)
        
        vector_store.save_local(self.vector_store_path)
        self.dispatcher.emit("new_vector_store", vector_store)
        return vector_store     
   
    def update_repositories(self):
        updated_repos = []
        
        for repo_config in self.repos_config:
            # Update repository
            if self.clone_or_update_repository(repo_config):
                # Update last updated timestamp
                repo_config['last_updated'] = datetime.now().isoformat()
                updated_repos.append(repo_config['name'])
                self.build_vector_store()
        
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