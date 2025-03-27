# Web3 RAG Server
This repository implements a RAG server for Web3 related requests.

## How it works
Data for RAG is provided by the user either manualy or by using the data submission interface. Git repositories are prefered as data source. 

## Start the servers
Intall the requirements with `pip install -r requirements.txt`
```
TOKENIZERS_PARALLELISM=false python main.py
```

## Server endpoints

### `query`
Handles RAG query via HTTP POST request

```
curl -X POST http://localhost:6789/query \
     -H "Content-Type: application/json" \
     -d '{"query":"Write an ERC721 soulbound contract"}'
```

### `repositories`
`GET` Queries the current git data repository config or `POST` insert a new repository to be crawled and added to the knowledge base.

```
curl -X POST http://localhost:6789/repositories \
     -H "Content-Type: application/json" \
     -d '{
         "name": "hardhat",
         "url": "https://github.com/NomicFoundation/hardhat.git",
         "local_path": "data/hardhat",
         "update_frequency_hours": 24
     }'
```

```
curl -X GET http://localhost:6789/repositories \
     -H "Content-Type: application/json"
```


### `knowledge_base`
Get the number of documents in the knowledge base. 
```
curl -X GET http://localhost:6789/knowledge_base \
     -H "Content-Type: application/json" 
```