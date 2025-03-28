import json
from flask import Flask, request, jsonify
from src.rag import IntegratedRAGSystem

# Flask Application Setup
app = Flask(__name__)
rag_system = IntegratedRAGSystem()

@app.route('/query', methods=['POST'])
def handle_query():
    """
    Handle RAG query via HTTP POST request
    """
    try:
        # Get query from request
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        retrieved_docs = rag_system.retrieve_relevant_documents(query)
        response = rag_system.generate_response(query, retrieved_docs)
        
        return jsonify({
            "query": query,
            "retrieved_documents": retrieved_docs,
            "response": response
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/repositories', methods=['GET', 'POST'])
def manage_repositories():
    if request.method == 'GET':
        return jsonify(rag_system.repo_manager.repos_config)
    
    elif request.method == 'POST':
        new_repo = request.get_json()
        
        required_keys = ['name', 'url', 'local_path', 'update_frequency_hours']
        if not all(key in new_repo for key in required_keys):
            return jsonify({"error": "Invalid repository configuration"}), 400
        
        rag_system.repo_manager.repos_config.append(new_repo)
        
        with open(rag_system.repo_manager.config_path, 'w') as f:
            json.dump(rag_system.repo_manager.repos_config, f, indent=2)
        
        rag_system.repo_manager.clone_or_update_repository(new_repo)
        rag_system.refresh_knowledge_base()
        
        return jsonify({"message": "Repository added successfully", "repository": new_repo})

@app.route('/knowledge_base', methods=['GET'])
def get_knowledge_base():
    return jsonify({
        "total_documents": len(rag_system.knowledge_base),
        "sample_documents": rag_system.knowledge_base[:10]  # Return first 10 documents
    })

def main():
    print("Starting Integrated RAG System API Server...")
    app.run(host='0.0.0.0', port=7860, debug=True)

if __name__ == '__main__':
    main()