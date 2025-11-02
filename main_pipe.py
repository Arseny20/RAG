from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from db_engine import OpenSearchDB
import os
import time
from embeddig_model import E5Embeddings
from preprocessing import preprocessing,flatten_dict
from metrics import metric_pipeline
import re
from tqdm import tqdm
import torch
import gc
import pandas as pd
import mlflow
import mlflow.utils.mlflow_tags as mlflow_tags 
from config import CONFIG
def generate_response(df_gpt, llm, db, index_name, k_retrieval, prompt_template):
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm

    questions = df_gpt['question'].tolist()
    row_answers = []
    answers = []
    contexts = []

    total_ctx_chars = 0
    start_time = time.time()

    for question in tqdm(questions):
        results = db.query_documents(question, index_name, k=k_retrieval)

        if not results:
            context = ""
            answer_clean = "В контексте это не упоминается."
            answer_raw = answer_clean
        else:
            context = " ".join([hit["_source"]["text"] for hit in results])
            message = chain.invoke({
                "context": context,
                "question": question,
            })
            answer_clean = re.sub(
                r"<think>.*?</think>", "", message.content,
                flags=re.DOTALL
            ).strip()
            answer_raw = message.content

        total_ctx_chars += len(context)
        row_answers.append(answer_raw)
        answers.append(answer_clean)
        contexts.append(context)

    elapsed = time.time() - start_time

    df_gpt['llm_clean_response'] = answers
    df_gpt['llm_raw_response'] = row_answers
    df_gpt['retrieved_context'] = contexts

    return df_gpt, {
        "num_questions": len(questions),
        "avg_context_chars": (total_ctx_chars / max(len(questions), 1)),
        "total_runtime_sec": elapsed,
    }

def pipeline():
    gc.collect()                # free unused CPU memory
    torch.cuda.empty_cache()    # release cached GPU memory
    torch.cuda.ipc_collect()
    df_gpt = pd.read_csv(f'{CONFIG["evaluation"]["metrics_folder"]}{CONFIG["evaluation"]["gpt_file"]}').sample(
        frac=1, random_state=CONFIG["experiment"]["seed"]
    ).reset_index(drop=True)
    print(df_gpt.head())
    data_folder = CONFIG["evaluation"]["data_folder"]
    embedder = E5Embeddings(
        model_name=CONFIG["embedding"]["model_name"],
        batch_size=CONFIG["embedding"]["batch_size"],
    )
    CONFIG["embedding"]["dim"] = embedder.dim
    CONFIG["embedding"]["device"] = str(embedder.device)
    CONFIG["tokenizer"]["vocab_size"] = getattr(embedder.tokenizer, "vocab_size", None)
    CONFIG["tokenizer"]["model_max_length"] = getattr(
        embedder.tokenizer, "model_max_length", None
    )

    db = OpenSearchDB(embedder)
    index_name = CONFIG["retriever"]["index_name"]
    db.create_body(index_name)
    data = preprocessing(data_folder,embedder)
    db.client.bulk(body=data)
    torch.cuda.empty_cache()
    llm = ChatOllama(
        model=CONFIG["llm"]["model_name"],
        temperature=CONFIG["llm"]["temperature"],
        seed=CONFIG["llm"]["seed"],
    )
    mlflow.set_experiment(CONFIG["experiment"]["name"])
    with mlflow.start_run():
        flat_cfg = flatten_dict(CONFIG)
        for key, value in flat_cfg.items():
            mlflow.log_param(key, value)
        df_result, metrics = generate_response(
            df_gpt=df_gpt,
            llm=llm,
            db=db,
            index_name=index_name,
            k_retrieval=CONFIG["retriever"]["top_k"],
            prompt_template=CONFIG["prompt"]["template"],
        )
        for m_name, m_val in metrics.items():
            mlflow.log_metric(m_name, float(m_val))

        answers_out = f'{CONFIG["evaluation"]["metrics_folder"]}{CONFIG["evaluation"]["qwen_output_file"]}'
        df_result.to_csv(answers_out, index=False, encoding="utf-8-sig")
        mlflow.log_artifact(answers_out)

        eval_out = f'{CONFIG["evaluation"]["metrics_folder"]}{CONFIG["evaluation"]["gpt_file"]}'
        mlflow.log_artifact(eval_out)
        print(df_result.head())

if __name__ == "__main__":
    pipeline()
    metric_pipeline()

    