from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter
import os
import json
import re
from config import CONFIG
def preprocessing(data_folder,embedding_model):
    
    text_splitter = CharacterTextSplitter(
        separator=CONFIG['text_splitter']['separator'],
        chunk_size=CONFIG['text_splitter']['chunk_size'],
        chunk_overlap=CONFIG['text_splitter']['chunk_overlap'],
        length_function=len
    )
    all_docs_lazy = []
    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        loader = PyMuPDFLoader(
            file_path,
            mode="page",
        )
        all_docs_lazy.append(loader.lazy_load())

   
    chunks = []
    for docs_lazy in all_docs_lazy:
        docs = []
        for doc in docs_lazy:
            doc.page_content = re.sub(r"\s+", " ", doc.page_content)
            docs.append(doc)
        chunks.extend(text_splitter.split_documents(docs))

    text = []
    metadata = []
    for chunk in chunks:
        chunk.page_content = re.sub("\n","",chunk.page_content)
        text.append(chunk.page_content)
        metadata_item = chunk.metadata
        filename = os.path.basename(metadata_item['file_path'])
        metadata.append({"document_name":filename,"page":metadata_item['page'],"total_pages":metadata_item['total_pages'],"author":metadata_item['author']})
    temp_text = ["passage: " + t for t in text]
    embeddings_list = embedding_model.embed_documents(temp_text)
    docs_opensearch = ''
    index = "docs"
    idx = 0
    for item_text, item_embed,item_meta in zip(text[:-1],embeddings_list[:-1],metadata[:-1]):
        docs_opensearch +=f'{{"index": {{"_index": "{index}", "_id": {idx}}}}} \n ' + f' {json.dumps({"text": item_text, "embedding": item_embed, "metadata": item_meta}, ensure_ascii=False)} \n '
        idx += 1
    docs_opensearch += f'{{"index": {{"_index": "{index}", "_id": {idx}}}}}\n' + f' {json.dumps({"text": text[idx], "embedding": embeddings_list[idx], "metadata": metadata[idx]}, ensure_ascii=False)}'

    return docs_opensearch
def flatten_dict(d, parent_key=""):
    """
    Turns nested dicts into flat dict keys for mlflow logging.
    Example:
    {"llm":{"model_name":"qwen","temp":0}}
    -> {"llm.model_name":"qwen", "llm.temp":0}
    """
    items = {}
    for k, v in d.items():
        key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, key))
        else:
            items[key] = v
    return items
