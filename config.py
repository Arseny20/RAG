# config.py

CONFIG = {
    "experiment": {
        "name": "rag_big_chunk_charsplit",
        "seed": 42,
    },
    "embedding": {
        "model_name": "multilingual-e5-large",
        "batch_size": 8,
        
    },

    "tokenizer": {
        "pretrained_path_template": "./models/tokenizer/{model_name}",
        "return_tensors": "pt",
        "padding": True,
        "max_length": 512,
        "truncation": True,
 
    },
    "text_splitter": {
        "type": "CharacterTextSplitter",
        "separator": "\n",
        "chunk_size": 256,
        "chunk_overlap": 100,
        "length_function": "len",
    },

    "retriever": {
        "top_k": 7,
        "index_name": "docs",
        "opensearch": {
            "knn_engine": "faiss",
            "knn_method": "hnsw",
            "space_type": "innerproduct",
            "number_of_shards": 1,
            "number_of_replicas": 1,
        },
    },

    "llm": {
        "provider": "ollama",
        "model_name": "hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M",
        "temperature": 0.7,
        "seed": 42,
    },

    "evaluation": {
        "gpt_file": "questions_answers.csv",
        "qwen_output_file": "qwen_gpt.csv",
        "data_folder": "./docs",
        "metrics_folder": "./metrics/"
    },

    "prompt": {
        "template": (
            "Ты умный помощник по документам. "
            "Твоя задача — ответить на вопрос, опираясь на контекст ниже. "
            "Отвечай коротко и точно. "
            "Отвечай на русском. Если вопрос не связан с контекстом, отвечай что данная информация не связана с документом.\n\n"
            "Контекст:\n{context}\n\n"
            "Вопрос:\n{question}\n\n"
        )
    },
}
