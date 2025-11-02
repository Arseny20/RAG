import pandas as pd
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
import re
from langchain_community.chat_models import ChatOpenAI
from config_metrics import CONFIG
import os
import mlflow
from collections import Counter
import matplotlib.pyplot as plt
from preprocessing import flatten_dict
from dotenv import load_dotenv

def generate_judge(llm,df,template):

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm

    questions = df['question'].tolist()
    contexts = df['retrieved_context'].tolist()
    answers_gpt = df['answer'].tolist()
    answers = df['llm_clean_response'].tolist()
    
    rows = []
    for i in tqdm(range(len(questions))):
        msg = chain.invoke({
            "context": contexts[i],
            "question": questions[i],
            "answer": answers[i],
            "answer_gpt": answers_gpt[i],
        })
        msg_str = msg.content
        answer_clean = re.sub(
                r"<think>.*?</think>", "", msg_str,
                flags=re.DOTALL
            ).strip()
        try:
            score_int = int(answer_clean)
            error = "ok"
        except Exception as e:
            score_int = None
            error = f"error: {type(e).__name__}"

        rows.append({
            "score_raw": answer_clean,
            "score_int": score_int,
            "error": error
        })

    df_result = pd.concat([df, pd.DataFrame(rows)], axis=1)
    
    return df_result
def judge_metric(values: list, title: str = "Rating_Histogram",status='all',categories=[0,1,2,3]):
    counts = Counter(values)

    xs = list(range(len(categories)))
    ys = [counts.get(cat, 0) for cat in categories]

    total = sum(ys) if sum(ys) > 0 else 1
    pcts = [y * 100.0 / total for y in ys]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(xs, ys, edgecolor="black")

    ax.set_title(title.replace("_", " ")+status)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(c) for c in categories])

    # Leave headroom for percentage labels
    ymax = max(ys) if ys else 1
    ax.set_ylim(0, ymax * 1.15 + 0.5)

    # Annotate each bar with percentage
    for bar, pct in zip(bars, pcts):
        x = bar.get_x() + bar.get_width() / 2.0
        y = bar.get_height()
        ax.text(x, y + (ymax * 0.03 + 0.05), f"{pct:.1f}%", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()

    # Save file
    fname = f"{title}_{status}.png"
    fig.savefig(f'{CONFIG["metrics"]["graphics_folder"]}{fname}', dpi=150)
    plt.close(fig)
    mlflow.log_param(f"{title}_num_unique_{status}", len(categories))
    mlflow.log_param(f"{title}_total_values_{status}", int(total))
    
    accuracy = (counts.get(2, 0) + counts.get(3, 0)) / total
    mlflow.log_metric(f"accuracy_{status}", round(accuracy, 4))
    for c, y in zip(categories, ys):
        mlflow.log_metric(f"count_{c}_{status}", y)
    for c, p in zip(categories, pcts):
        mlflow.log_metric(f"percent_{c}_{status}", round(p, 4))
    return fname
def metric_pipeline():
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    print(api_key)
    df_metrics = pd.read_csv(f'{CONFIG["metrics"]["metrics_folder"]}{CONFIG["metrics"]["qwen_output_file"]}')

    open_llm = ChatOpenAI(
        model_name= CONFIG["llm_judge"]["model_name"],
        temperature=CONFIG["llm_judge"]["temperature"],
        seed = CONFIG["llm_judge"]["seed"], # или "gpt-4o" / "gpt-4-turbo"
        openai_api_key=api_key
    )
    mlflow.set_experiment(CONFIG["experiment"]["name"])
    df_judge = generate_judge(open_llm,df_metrics,CONFIG["prompt"]["templateJudge"])
    with mlflow.start_run():
        flat_cfg = flatten_dict(CONFIG)
        for key, value in flat_cfg.items():
            mlflow.log_param(key, value)
        df_judge.to_csv(f'{CONFIG["metrics"]["metrics_folder"]}{CONFIG["metrics"]["metrics_file"]}', index=False, encoding="utf-8-sig")
        judge_out = f'{CONFIG["metrics"]["metrics_folder"]}{CONFIG["metrics"]["metrics_file"]}'
        mlflow.log_artifact(judge_out)
        histogram = judge_metric(df_judge['score_int'].tolist())
        mlflow.log_artifact(f'{CONFIG["metrics"]["graphics_folder"]}{histogram}')
        status = df_judge['status'].unique().tolist()
        for item in status:
            histogram = judge_metric(df_judge.loc[df_judge['status'] ==item,"score_int"].tolist(),title=f"Rating_Histogram",status=item)
            mlflow.log_artifact(f'{CONFIG["metrics"]["graphics_folder"]}{histogram}')
            
        print(df_judge.head())