import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_TASK_NAMES = ["tuple", "dependency", "question"]

# ---------------------------
# Data loading
# ---------------------------


def load_tifa160_data(path: str = "tifa160-dev-anns.csv") -> pd.DataFrame:
    path = Path(__file__).parent / "data" / path
    return pd.read_csv(path)


# ---------------------------
# Example building
# ---------------------------


def create_train_example(
    prompt: str,
    task: str = "tuple",
    tuples: Optional[List[str]] = None,
    dependencies: Optional[List[str]] = None,
    questions: Optional[List[str]] = None,
) -> Dict[str, str]:
    assert task in _TASK_NAMES, f"task == {task}"

    inputs, outputs = [], []
    n_outputs = len(tuples or [])

    # Keep textual ICL for continuity; model returns structured JSON per schema.
    if task == "tuple":
        inputs.append(prompt)
        for i in range(n_outputs):
            outputs.append(f"{i + 1} | {tuples[i]}".strip())

    elif task == "dependency":
        inputs.append(prompt)
        for i in range(n_outputs):
            inputs.append(f"{i + 1} | {tuples[i]}".strip())
        for i in range(n_outputs):
            outputs.append(f"{i + 1} | {dependencies[i]}".strip())

    elif task == "question":
        inputs.append(prompt)
        for i in range(n_outputs):
            inputs.append(f"{i + 1} | {tuples[i]}".strip())
        for i in range(n_outputs):
            outputs.append(f"{i + 1} | {questions[i]}".strip())

    return {"input": "\n".join(inputs), "output": "\n".join(outputs)}


def tifa_id2example(df: pd.DataFrame, id: str, task: str = "tuple") -> Dict[str, str]:
    sub = df[df.item_id == id]
    prompt = sub.text.iloc[0]
    all_tuples = sub.tuple.tolist()
    all_dependencies = sub.dependency.tolist()
    all_questions = sub.question_natural_language.tolist()

    return create_train_example(
        prompt=prompt,
        task=task,
        tuples=all_tuples,
        dependencies=all_dependencies,
        questions=all_questions,
    )


def get_tifa_examples(data_df: pd.DataFrame, ids: List[str], task: str = "tuple"):
    return [tifa_id2example(data_df, id, task=task) for id in ids]


# ---------------------------
# Training examples
# ---------------------------
TIFA160_ICL_TRAIN_IDS = [
    "coco_361740",
    "drawbench_155",
    "partiprompt_86",
    "paintskill_374",
    "coco_552592",
    "partiprompt_1414",
    "coco_627537",
    "coco_744388",
    "partiprompt_1108",
    "coco_397109",
    "coco_666114",
    "coco_62896",
    "paintskill_235",
    "drawbench_159",
    "partiprompt_893",
    "coco_322041",
    "coco_292534",
    "drawbench_57",
    "partiprompt_555",
    "coco_488166",
    "partiprompt_726",
    "coco_323167",
    "coco_625027",
]
assert len(TIFA160_ICL_TRAIN_IDS) == 23


###############################
# Load TIFA160 Likert Scores
###############################

human_tifa160_likert_df = pd.read_csv(
    Path(__file__).parent / "data/tifa160-likert-anns.csv"
)


def load_human_likert_ann(t2i_model, item_id):
    """Load the Likert scores of human annotations on DSG-1k prompts"""

    assert t2i_model in [
        "mini-dalle",
        "sd1dot1",
        "sd1dot5",
        "sd2dot1",
        "vq-diffusion",
    ], t2i_model

    human_likert_df = human_tifa160_likert_df[
        human_tifa160_likert_df.t2i_model == t2i_model
    ]
    item_df = human_likert_df[human_likert_df.item_id == item_id]

    worker_ids = item_df.worker_id.unique().tolist()
    n_workers = len(worker_ids)

    human_likert_output = {
        "item_id": item_id,
        "t2i_model": t2i_model,
        "worker_ids": worker_ids,
        "n_workers": n_workers,
        "likert_scores": item_df.answer.tolist(),
    }

    return human_likert_output


dsg_id_to_tifa_id = {}
tifa_id_to_dsg_id = {}
for i, row in human_tifa160_likert_df.iterrows():
    dsg_id_to_tifa_id[row["item_id"]] = row["source_id"]
    tifa_id_to_dsg_id[row["source_id"]] = row["item_id"]


###############################
# Load DSG annotations
###############################

dsg_df = pd.read_csv(Path(__file__).parent / "data/dsg-1k-anns.csv")

dsg_itemid2data = {}
for idx, row in dsg_df.iterrows():
    item_id = row["item_id"]

    if item_id not in dsg_itemid2data:
        data = []
    else:
        data = dsg_itemid2data[item_id]

    # add the row
    data.append(row)
    dsg_itemid2data[item_id] = data
# merge the data
for item_id, data in dsg_itemid2data.items():
    dsg_itemid2data[item_id] = pd.concat(data, axis=1).T

dsg_id2tuple = {}
dsg_id2question = {}
dsg_id2dependency = {}

for item_id, item_df in dsg_itemid2data.items():
    try:
        qid2tup = {}
        for idx, row in item_df.iterrows():
            qid = row["proposition_id"]
            output = row["tuple"]
            qid2tup[qid] = output

    except Exception:
        qid2tup = {}
    dsg_id2tuple[item_id] = qid2tup

for item_id, item_df in dsg_itemid2data.items():
    try:
        qid2q = {}
        for idx, row in item_df.iterrows():
            qid = row["proposition_id"]
            output = row["question_natural_language"]
            qid2q[qid] = output

    except Exception:
        qid2q = {}
    dsg_id2question[item_id] = qid2q

for item_id, item_df in dsg_itemid2data.items():
    try:
        qid2dep = {}
        for idx, row in item_df.iterrows():
            qid = row["proposition_id"]
            output = row["dependency"]
            if type(output) == str:
                output = list(output.split(","))
                output = [int(x.strip()) for x in output]
            qid2dep[qid] = output

    except Exception:
        qid2dep = {}
    dsg_id2dependency[item_id] = qid2dep
