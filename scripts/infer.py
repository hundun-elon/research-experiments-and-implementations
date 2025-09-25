"""
scripts/infer.py

Main entrypoint for inference/experiments.

Adds:
 - W&B logging of configs, metrics, and outputs
"""

import argparse
import json
import os
import random
import numpy as np
from typing import Dict, Any, List

from tqdm import tqdm
import wandb

# Local imports
from dsp_langchain.dspy_pipelines import RubricProgram, SelfRefine, RAG

# Optionally import Hugging Face / DSPy LM setup
import dspy
from transformers import AutoTokenizer, AutoModelForCausalLM


# ----------------------------
# Utilities
# ----------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def save_jsonl(items: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


# ----------------------------
# Direct & rubric-guided baselines
# ----------------------------
def run_direct(item: Dict[str, Any], model, tokenizer) -> Dict[str, Any]:
    prompt = f"""You are grading a student's answer.
Student answer: {item['student_answer']}
Give a numeric score only, between 0 and {sum(item['rubric'].values())}.
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    score_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    try:
        score = float(score_text.split()[-1])
    except Exception:
        score = 0.0

    out = dict(item)
    out["model_score"] = score
    out["model_raw_output"] = score_text
    return out


def run_rubric_guided(item: Dict[str, Any], model, tokenizer) -> Dict[str, Any]:
    rubric = item["rubric"]
    prompt = "Grade the following student answer using this rubric:\n\n"
    for crit, max_score in rubric.items():
        prompt += f"- {crit} (0-{max_score})\n"
    prompt += f"\nStudent answer: {item['student_answer']}\n\nReturn JSON: {{criterion: score}}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        scores = json.loads(text)
    except Exception:
        scores = {crit: 0 for crit in rubric}

    out = dict(item)
    out["per_criterion"] = scores
    out["model_score"] = float(sum(scores.values()))
    out["model_raw_output"] = text
    return out


# ----------------------------
# DSPy pipeline selector
# ----------------------------
def get_dspy_pipeline(pipeline_name: str):
    if pipeline_name == "rubric_program":
        return RubricProgram()
    elif pipeline_name == "self_refine":
        return SelfRefine()
    elif pipeline_name == "rag":
        return RAG(exemplars=[])
    else:
        raise ValueError(f"Unknown DSPy pipeline: {pipeline_name}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["direct", "rubric_guided", "dspy_pipeline"],
        required=True,
    )
    parser.add_argument(
        "--dspy-pipeline",
        type=str,
        choices=["rubric_program", "self_refine", "rag"],
        default="rubric_program",
    )
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    # ----------------------------
    # Init W&B
    # ----------------------------
    wandb.init(
        project="llm-grading",
        config=vars(args),
        tags=[args.strategy, args.model_name],
    )

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load model
    print(f"Loading model {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to("cuda")

    # Configure DSPy LM backend
    dspy.configure(lm=args.model_name)

    # Load data
    items = load_jsonl(args.input_jsonl)

    results = []
    if args.strategy == "direct":
        for item in tqdm(items):
            results.append(run_direct(item, model, tokenizer))
    elif args.strategy == "rubric_guided":
        for item in tqdm(items):
            results.append(run_rubric_guided(item, model, tokenizer))
    elif args.strategy == "dspy_pipeline":
        pipeline = get_dspy_pipeline(args.dspy_pipeline)
        for item in tqdm(items):
            results.append(pipeline.run(item))

    # Save to disk
    save_jsonl(results, args.output_jsonl)
    print(f"Saved {len(results)} results to {args.output_jsonl}")

    # ----------------------------
    # Log to W&B
    # ----------------------------
    for r in results:
        wandb.log({
            "model_score": r.get("model_score", 0.0),
            "ground_truth": r.get("human_score", 0.0),
        })

    wandb.save(args.output_jsonl)
    wandb.finish()
