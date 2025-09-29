# """
# scripts/infer.py

# Main entrypoint for inference/experiments.

# Adds:
#  - W&B logging of configs, metrics, and outputs
# """

# import argparse
# import json
# import os
# import random
# import numpy as np
# from typing import Dict, Any, List

# from tqdm import tqdm
# import wandb

# # Local imports
# from dsp_langchain.dspy_pipelines import RubricProgram, SelfRefine, RAG

# # Optionally import Hugging Face / DSPy LM setup
# import dspy
# from transformers import AutoTokenizer, AutoModelForCausalLM


# # ----------------------------
# # Utilities
# # ----------------------------
# def load_jsonl(path: str) -> List[Dict[str, Any]]:
#     with open(path, "r") as f:
#         return [json.loads(line) for line in f]


# def save_jsonl(items: List[Dict[str, Any]], path: str):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, "w") as f:
#         for item in items:
#             f.write(json.dumps(item) + "\n")


# # ----------------------------
# # Direct & rubric-guided baselines
# # ----------------------------
# def run_direct(item: Dict[str, Any], model, tokenizer) -> Dict[str, Any]:
#     prompt = f"""You are grading a student's answer.
# Student answer: {item['student_answer']}
# Give a numeric score only, between 0 and {sum(item['rubric'].values())}.
# """
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     outputs = model.generate(**inputs, max_new_tokens=20)
#     score_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
#     try:
#         score = float(score_text.split()[-1])
#     except Exception:
#         score = 0.0

#     out = dict(item)
#     out["model_score"] = score
#     out["model_raw_output"] = score_text
#     return out


# def run_rubric_guided(item: Dict[str, Any], model, tokenizer) -> Dict[str, Any]:
#     rubric = item["rubric"]
#     prompt = "Grade the following student answer using this rubric:\n\n"
#     for crit, max_score in rubric.items():
#         prompt += f"- {crit} (0-{max_score})\n"
#     prompt += f"\nStudent answer: {item['student_answer']}\n\nReturn JSON: {{criterion: score}}"

#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     outputs = model.generate(**inputs, max_new_tokens=200)
#     text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     try:
#         scores = json.loads(text)
#     except Exception:
#         scores = {crit: 0 for crit in rubric}

#     out = dict(item)
#     out["per_criterion"] = scores
#     out["model_score"] = float(sum(scores.values()))
#     out["model_raw_output"] = text
#     return out


# # ----------------------------
# # DSPy pipeline selector
# # ----------------------------
# def get_dspy_pipeline(pipeline_name: str):
#     if pipeline_name == "rubric_program":
#         return RubricProgram()
#     elif pipeline_name == "self_refine":
#         return SelfRefine()
#     elif pipeline_name == "rag":
#         return RAG(exemplars=[])
#     else:
#         raise ValueError(f"Unknown DSPy pipeline: {pipeline_name}")


# # ----------------------------
# # Main
# # ----------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-name", type=str, required=True)
#     parser.add_argument("--input-jsonl", type=str, required=True)
#     parser.add_argument("--output-jsonl", type=str, required=True)
#     parser.add_argument(
#         "--strategy",
#         type=str,
#         choices=["direct", "rubric_guided", "dspy_pipeline"],
#         required=True,
#     )
#     parser.add_argument(
#         "--dspy-pipeline",
#         type=str,
#         choices=["rubric_program", "self_refine", "rag"],
#         default="rubric_program",
#     )
#     parser.add_argument("--seed", type=int, default=13)
#     args = parser.parse_args()

#     # ----------------------------
#     # Init W&B
#     # ----------------------------
#     wandb.init(
#         project="llm-grading",
#         config=vars(args),
#         tags=[args.strategy, args.model_name],
#     )

#     # Reproducibility
#     random.seed(args.seed)
#     np.random.seed(args.seed)

#     # Load model
#     print(f"Loading model {args.model_name} ...")
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name)
#     model = AutoModelForCausalLM.from_pretrained(args.model_name).to("cuda")

#     # Configure DSPy LM backend
#     dspy.configure(lm=args.model_name)

#     # Load data
#     items = load_jsonl(args.input_jsonl)

#     results = []
#     if args.strategy == "direct":
#         for item in tqdm(items):
#             results.append(run_direct(item, model, tokenizer))
#     elif args.strategy == "rubric_guided":
#         for item in tqdm(items):
#             results.append(run_rubric_guided(item, model, tokenizer))
#     elif args.strategy == "dspy_pipeline":
#         pipeline = get_dspy_pipeline(args.dspy_pipeline)
#         for item in tqdm(items):
#             results.append(pipeline.run(item))

#     # Save to disk
#     save_jsonl(results, args.output_jsonl)
#     print(f"Saved {len(results)} results to {args.output_jsonl}")

#     # ----------------------------
#     # Log to W&B
#     # ----------------------------
#     for r in results:
#         wandb.log({
#             "model_score": r.get("model_score", 0.0),
#             "ground_truth": r.get("human_score", 0.0),
#         })

#     wandb.save(args.output_jsonl)
#     wandb.finish()


#!/usr/bin/env python3
"""
ASAP Automated Essay Grading Pipeline with DSPy
Supports multiple grading strategies: direct, rubric-guided, self-refinement
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
from datetime import datetime

# Import metrics from your existing script
try:
    from metrics import evaluate_predictions
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: metrics.py not found. Metrics will not be computed.")

# Optional imports
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    print("Warning: DSPy not available. Using fallback implementation.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. CPU-only mode.")


# ============================================================================
# TEMPLATE DEFINITIONS
# ============================================================================

CATEGORY_PROMPTS = {
    "source_based": (
        "Write an argumentative essay in response to the provided reading(s). "
        "Your essay should analyze the source material, develop a clear position, "
        "and support it with evidence from the text(s)."
    ),
    "independent": (
        "Write an argumentative essay in response to the given topic. "
        "Your essay should present a clear point of view, demonstrate critical thinking, "
        "and support your position with appropriate examples and reasoning."
    )
}


# ============================================================================
# DSPY SIGNATURES AND MODULES
# ============================================================================

if DSPY_AVAILABLE:
    class DirectGradingSignature(dspy.Signature):
        """Grade an essay directly without explicit reasoning steps."""
        task_prompt: str = dspy.InputField(desc="The essay writing task")
        essay_text: str = dspy.InputField(desc="The student's essay")
        score: int = dspy.OutputField(desc="Score from 1-6")
        feedback: str = dspy.OutputField(desc="Brief grading feedback")
        confidence: float = dspy.OutputField(desc="Confidence in score (0-1)")

    class RubricGuidedSignature(dspy.Signature):
        """Grade an essay using rubric criteria with reasoning."""
        task_prompt: str = dspy.InputField(desc="The essay writing task")
        essay_text: str = dspy.InputField(desc="The student's essay")
        rubric_text: str = dspy.InputField(desc="Grading rubric descriptions")
        rationale: str = dspy.OutputField(desc="Step-by-step reasoning")
        score: int = dspy.OutputField(desc="Score from 1-6")
        feedback: str = dspy.OutputField(desc="Detailed feedback")
        confidence: float = dspy.OutputField(desc="Confidence in score (0-1)")

    class SelfRefineSignature(dspy.Signature):
        """Critique and refine an initial grading decision."""
        task_prompt: str = dspy.InputField(desc="The essay writing task")
        essay_text: str = dspy.InputField(desc="The student's essay")
        initial_score: int = dspy.InputField(desc="Initial score")
        initial_rationale: str = dspy.InputField(desc="Initial reasoning")
        rubric_text: str = dspy.InputField(desc="Grading rubric")
        critique: str = dspy.OutputField(desc="Critique of initial grading")
        refined_score: int = dspy.OutputField(desc="Refined score from 1-6")
        refined_feedback: str = dspy.OutputField(desc="Refined feedback")
        confidence: float = dspy.OutputField(desc="Confidence in refined score (0-1)")


# ============================================================================
# GRADING STRATEGIES
# ============================================================================

class DirectGrader:
    """Direct grading without explicit reasoning steps."""
    
    def __init__(self, lm=None):
        self.lm = lm
        if DSPY_AVAILABLE and lm:
            self.predictor = dspy.Predict(DirectGradingSignature)
        else:
            self.predictor = None
    
    def grade(self, task_prompt: str, essay_text: str, **kwargs) -> Dict:
        if self.predictor:
            try:
                result = self.predictor(task_prompt=task_prompt, essay_text=essay_text)
                score = self._extract_score(result.score)
                confidence = self._extract_confidence(result.confidence)
                return {
                    "score": score,
                    "feedback": result.feedback,
                    "rationale": "",
                    "confidence": confidence
                }
            except Exception as e:
                print(f"DSPy prediction error: {e}")
                return self._fallback_grade(task_prompt, essay_text)
        else:
            return self._fallback_grade(task_prompt, essay_text)
    
    def _fallback_grade(self, task_prompt: str, essay_text: str) -> Dict:
        """Fallback when DSPy is not available."""
        prompt = f"""Task: {task_prompt}

Essay: {essay_text}

Grade this essay on a scale of 1-6 where:
1 = Very little or no mastery
2 = Little mastery
3 = Developing mastery
4 = Adequate mastery
5 = Reasonably consistent mastery
6 = Clear and consistent mastery

Provide:
1. Score (1-6):
2. Brief feedback:"""
        
        # Simple fallback scoring (in production, would call actual LLM)
        return {
            "score": 3,  # Default middle score
            "feedback": "Automated grading unavailable - using fallback",
            "rationale": "",
            "confidence": 0.5
        }
    
    def _extract_score(self, score_str) -> int:
        """Extract numeric score, handling various formats."""
        try:
            if isinstance(score_str, int):
                score = score_str
            else:
                score_str = str(score_str).strip()
                # Extract first digit found
                import re
                match = re.search(r'\d+', score_str)
                score = int(match.group()) if match else 3
            
            # Clamp to valid range
            return max(1, min(6, score))
        except:
            return 3
    
    def _extract_confidence(self, conf_str) -> float:
        """Extract confidence score, handling various formats."""
        try:
            if isinstance(conf_str, (int, float)):
                conf = float(conf_str)
            else:
                import re
                match = re.search(r'0?\.\d+|\d+\.?\d*', str(conf_str))
                conf = float(match.group()) if match else 0.5
            
            # Clamp to [0, 1]
            return max(0.0, min(1.0, conf))
        except:
            return 0.5


class RubricGuidedGrader:
    """Rubric-guided grading with Chain-of-Thought reasoning."""
    
    def __init__(self, lm=None):
        self.lm = lm
        if DSPY_AVAILABLE and lm:
            self.predictor = dspy.ChainOfThought(RubricGuidedSignature)
        else:
            self.predictor = None
    
    def grade(self, task_prompt: str, essay_text: str, rubric_text: str, **kwargs) -> Dict:
        if self.predictor:
            try:
                result = self.predictor(
                    task_prompt=task_prompt,
                    essay_text=essay_text,
                    rubric_text=rubric_text
                )
                score = self._extract_score(result.score)
                confidence = self._extract_confidence(result.confidence)
                return {
                    "score": score,
                    "feedback": result.feedback,
                    "rationale": result.rationale,
                    "confidence": confidence
                }
            except Exception as e:
                print(f"DSPy ChainOfThought error: {e}")
                return self._fallback_grade(task_prompt, essay_text, rubric_text)
        else:
            return self._fallback_grade(task_prompt, essay_text, rubric_text)
    
    def _fallback_grade(self, task_prompt: str, essay_text: str, rubric_text: str) -> Dict:
        """Fallback when DSPy is not available."""
        return {
            "score": 3,
            "feedback": "Rubric-guided grading unavailable - using fallback",
            "rationale": "Automated reasoning unavailable",
            "confidence": 0.5
        }
    
    def _extract_score(self, score_str) -> int:
        """Extract numeric score."""
        try:
            if isinstance(score_str, int):
                score = score_str
            else:
                import re
                match = re.search(r'\d+', str(score_str))
                score = int(match.group()) if match else 3
            return max(1, min(6, score))
        except:
            return 3
    
    def _extract_confidence(self, conf_str) -> float:
        """Extract confidence score."""
        try:
            if isinstance(conf_str, (int, float)):
                conf = float(conf_str)
            else:
                import re
                match = re.search(r'0?\.\d+|\d+\.?\d*', str(conf_str))
                conf = float(match.group()) if match else 0.5
            return max(0.0, min(1.0, conf))
        except:
            return 0.5


class SelfRefinementGrader:
    """Self-refinement grading: initial reasoning + critique + revision."""
    
    def __init__(self, lm=None):
        self.lm = lm
        if DSPY_AVAILABLE and lm:
            self.initial_predictor = dspy.ChainOfThought(RubricGuidedSignature)
            self.refine_predictor = dspy.Predict(SelfRefineSignature)
        else:
            self.initial_predictor = None
            self.refine_predictor = None
    
    def grade(self, task_prompt: str, essay_text: str, rubric_text: str, **kwargs) -> Dict:
        if self.initial_predictor and self.refine_predictor:
            try:
                # Step 1: Initial grading
                initial = self.initial_predictor(
                    task_prompt=task_prompt,
                    essay_text=essay_text,
                    rubric_text=rubric_text
                )
                initial_score = self._extract_score(initial.score)
                
                # Step 2: Self-critique and refinement
                refined = self.refine_predictor(
                    task_prompt=task_prompt,
                    essay_text=essay_text,
                    initial_score=initial_score,
                    initial_rationale=initial.rationale,
                    rubric_text=rubric_text
                )
                refined_score = self._extract_score(refined.refined_score)
                confidence = self._extract_confidence(refined.confidence)
                
                return {
                    "score": refined_score,
                    "feedback": refined.refined_feedback,
                    "rationale": f"Initial: {initial.rationale}\nCritique: {refined.critique}",
                    "confidence": confidence
                }
            except Exception as e:
                print(f"Self-refinement error: {e}")
                return self._fallback_grade(task_prompt, essay_text, rubric_text)
        else:
            return self._fallback_grade(task_prompt, essay_text, rubric_text)
    
    def _fallback_grade(self, task_prompt: str, essay_text: str, rubric_text: str) -> Dict:
        """Fallback when DSPy is not available."""
        return {
            "score": 3,
            "feedback": "Self-refinement grading unavailable - using fallback",
            "rationale": "Automated refinement unavailable",
            "confidence": 0.5
        }
    
    def _extract_score(self, score_str) -> int:
        """Extract numeric score."""
        try:
            if isinstance(score_str, int):
                score = score_str
            else:
                import re
                match = re.search(r'\d+', str(score_str))
                score = int(match.group()) if match else 3
            return max(1, min(6, score))
        except:
            return 3
    
    def _extract_confidence(self, conf_str) -> float:
        """Extract confidence score."""
        try:
            if isinstance(conf_str, (int, float)):
                conf = float(conf_str)
            else:
                import re
                match = re.search(r'0?\.\d+|\d+\.?\d*', str(conf_str))
                conf = float(match.group()) if match else 0.5
            return max(0.0, min(1.0, conf))
        except:
            return 0.5


# ============================================================================
# DATA LOADING
# ============================================================================

def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed line: {e}")
    return data


def load_rubrics(rubric_path: Path) -> Dict[str, Dict[int, str]]:
    """Load rubrics organized by category and score."""
    rubrics_raw = load_jsonl(rubric_path)
    rubrics = {}
    
    for entry in rubrics_raw:
        category = entry.get('category', '')
        score = entry.get('score', 0)
        description = entry.get('description', '')
        
        if category not in rubrics:
            rubrics[category] = {}
        
        rubrics[category][score] = description
    
    return rubrics


def format_rubric_text(rubrics: Dict[int, str]) -> str:
    """Format rubric dictionary into readable text."""
    lines = ["Grading Rubric:"]
    for score in sorted(rubrics.keys(), reverse=True):
        lines.append(f"\nScore {score}: {rubrics[score]}")
    return "\n".join(lines)


# ============================================================================
# GRADING PIPELINE
# ============================================================================

class GradingPipeline:
    """Main grading pipeline."""
    
    def __init__(self, strategy: str, model_name: str, lm=None):
        self.strategy = strategy
        self.model_name = model_name
        
        # Initialize grader based on strategy
        if strategy == "direct":
            self.grader = DirectGrader(lm)
        elif strategy == "rubric_guided":
            self.grader = RubricGuidedGrader(lm)
        elif strategy == "self_refinement":
            self.grader = SelfRefinementGrader(lm)
        # Option 5: CPU-only inference (no VLLM)
        elif os.environ.get("USE_CPU"):
            print(f"Loading model on CPU (slow): {args.model_name}...")
            from transformers import pipeline
            
            model_mapping = {
                "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
                "phi-3": "microsoft/Phi-3-mini-4k-instruct",
            }
            model_id = model_mapping.get(args.model_name, args.model_name)
            
            # Load with 8-bit quantization for CPU
            pipe = pipeline("text-generation", model=model_id, device=-1)
            lm = dspy.HFModel(model=model_id, is_client=False)
            print(" CPU inference configured (may be slow)")
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def grade_essay(self, essay: Dict, rubrics: Dict[str, Dict[int, str]]) -> Dict:
        """Grade a single essay."""
        essay_id = essay.get('id', 'unknown')
        essay_text = essay.get('answer', '')
        human_score = essay.get('human_score', 0)
        category = essay.get('category', 'independent')
        
        # Get task prompt
        task_prompt = CATEGORY_PROMPTS.get(category, CATEGORY_PROMPTS['independent'])
        
        # Prepare rubric text if needed
        rubric_text = ""
        if self.strategy in ["rubric_guided", "self_refinement"]:
            category_rubrics = rubrics.get(category, {})
            rubric_text = format_rubric_text(category_rubrics)
        
        # Grade the essay
        try:
            result = self.grader.grade(
                task_prompt=task_prompt,
                essay_text=essay_text,
                rubric_text=rubric_text
            )
            
            return {
                "id": essay_id,
                "category": category,
                "human_score": human_score,
                "model_score": result["score"],
                "confidence": result.get("confidence", 0.5),
                "rationale": result.get("rationale", ""),
                "feedback": result.get("feedback", ""),
                "strategy": self.strategy,
                "model": self.model_name
            }
        except Exception as e:
            print(f"Error grading essay {essay_id}: {e}")
            return {
                "id": essay_id,
                "category": category,
                "human_score": human_score,
                "model_score": 0,
                "confidence": 0.0,
                "rationale": "",
                "feedback": f"Error: {str(e)}",
                "strategy": self.strategy,
                "model": self.model_name
            }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ASAP Essay Grading with DSPy")
    parser.add_argument("--strategy", type=str, required=True,
                       choices=["direct", "rubric_guided", "self_refinement"],
                       help="Grading strategy")
    parser.add_argument("--model-name", type=str, required=True,
                       help="Model name identifier")
    parser.add_argument("--asap-dir", type=str, default="asap-dataset",
                       help="Directory containing ASAP dataset")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSONL file path")
    parser.add_argument("--wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of essays to process (for testing)")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to local model (if using local LLM)")
    parser.add_argument("--skip-metrics", action="store_true",
                       help="Skip metrics computation (useful for debugging)")
    
    args = parser.parse_args()
    
    # Setup paths
    asap_dir = Path(args.asap_dir)
    essays_path = asap_dir / "train.jsonl"
    rubrics_path = asap_dir / "rubric.jsonl"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B if requested
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project="asap-essay-grading",
            name=f"{args.model_name}_{args.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "strategy": args.strategy,
                "model": args.model_name,
                "dataset": "ASAP"
            }
        )
    
    # Load data
    print(f"Loading essays from {essays_path}...")
    essays = load_jsonl(essays_path)
    print(f"Loaded {len(essays)} essays")
    
    print(f"Loading rubrics from {rubrics_path}...")
    rubrics = load_rubrics(rubrics_path)
    print(f"Loaded rubrics for categories: {list(rubrics.keys())}")
    
    # Limit essays if specified
    if args.limit:
        essays = essays[:args.limit]
        print(f"Limited to {len(essays)} essays")
    
    # Initialize language model (placeholder - would configure actual LLM here)
    lm = None
    if DSPY_AVAILABLE:
        # Example: Configure DSPy LM
        # For OpenAI:
        # lm = dspy.OpenAI(model=args.model_name)
        # For HuggingFace:
        # lm = dspy.HFModel(model=args.model_path or args.model_name)
        # For local models:
        # lm = dspy.HFClientTGI(model=args.model_path, port=8080)
        # dspy.settings.configure(lm=lm)
        print("Note: Configure DSPy LM in production (see commented examples in code)")
    
    # Initialize pipeline
    print(f"Initializing {args.strategy} grading pipeline...")
    pipeline = GradingPipeline(args.strategy, args.model_name, lm)
    
    # Process essays
    results = []
    print(f"\nGrading {len(essays)} essays...")
    for i, essay in enumerate(essays):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(essays)}")
        
        result = pipeline.grade_essay(essay, rubrics)
        results.append(result)
    
    # Save results
    print(f"\nSaving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Saved {len(results)} graded essays to {output_path}")
    
    # Compute metrics using your metrics.py script
    if not args.skip_metrics and METRICS_AVAILABLE:
        print("\nComputing metrics using metrics.py...")
        try:
            metrics = evaluate_predictions(results)
            
            print("\n" + "="*60)
            print("EVALUATION METRICS")
            print("="*60)
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
            print("="*60)
            
            # Save metrics
            metrics_path = output_path.parent / f"{output_path.stem}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\nMetrics saved to {metrics_path}")
            
            # Log to W&B if enabled
            if args.wandb and WANDB_AVAILABLE:
                wandb.log(metrics)
                
                # Create distribution plots
                import matplotlib.pyplot as plt
                human_scores = [r['human_score'] for r in results if r['human_score'] > 0]
                model_scores = [r['model_score'] for r in results if r['model_score'] > 0]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Distribution comparison
                ax1.hist(human_scores, bins=6, alpha=0.5, label='Human', range=(0.5, 6.5))
                ax1.hist(model_scores, bins=6, alpha=0.5, label='Model', range=(0.5, 6.5))
                ax1.set_xlabel('Score')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Score Distribution Comparison')
                ax1.legend()
                
                # Scatter plot
                ax2.scatter(human_scores, model_scores, alpha=0.5)
                ax2.plot([1, 6], [1, 6], 'r--', label='Perfect Agreement')
                ax2.set_xlabel('Human Score')
                ax2.set_ylabel('Model Score')
                ax2.set_title('Human vs Model Scores')
                ax2.legend()
                ax2.set_xlim(0.5, 6.5)
                ax2.set_ylim(0.5, 6.5)
                
                plt.tight_layout()
                wandb.log({"score_analysis": wandb.Image(fig)})
                plt.close()
                
        except Exception as e:
            print(f"Error computing metrics: {e}")
            print("Results were saved successfully, but metrics computation failed.")
    elif not args.skip_metrics:
        print("\nWarning: metrics.py not available. Skipping metrics computation.")
        print("To compute metrics, ensure metrics.py is in the same directory or in your PYTHONPATH.")
    
    # Finish W&B run
    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    print("\nDone!")


if __name__ == "__main__":
    main()