# 🧠 RAG System Setup Guide

## Requirements
Before you begin, make sure you have **Docker** and **Ollama** installed.

---

##Docker Setup

Use the following container image:

```bash
docker run --rm -d \
  -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=your_password" \
  opensearchproject/opensearch:3
```
## 1. Download the LLM
```bash
ollama run hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M
```
## 2. Download the Embedding Model
```bash
python saving_model.py
```
## 3. Launch MLflow UI
```bash
mlflow ui
```
## 3. Run the Pipeline
```bash
python main_pipe.py
```
