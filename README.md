# RAG
to strat working you need to install docker and ollama
# Docker
use this container - opensearchproject/opensearch:3
docker run --rm -d -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=your_password" opensearchproject/opensearch:3
# Models
run this code in terminal to download model: ollama run hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M
in your terminal in your directory  run python saving_model.py to download embedding model
# Start
in your terminal in your directory  run mlflow ui to start mlflow
to start pipeline run in your terminal in your directory  run python main_pipe.py 


