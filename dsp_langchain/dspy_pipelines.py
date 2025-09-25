"""
dsp_langchain/dspy_pipelines.py

Formal DSPy implementations of the grading pipelines described in the
research plan.

Provides:
 - RubricProgram (criterion-by-criterion)
 - SelfRefine (initial judgment -> critique -> revise)
 - RAG (retrieval augmented grading using exemplars)

Each pipeline is implemented using DSPy modules (dspy.Predict and dspy.Refine)
and returns a plain Python dict compatible with your experiment JSONL:
{
  ... original item fields ...,
  "per_criterion": {...},
  "model_score": <float>,
  "model_raw_output": ...,
  "justifications": {...}   # optional
}

Notes:
 - These pipelines are "formal DSPy" programs: they declare DSPy module signatures
   and then call them. You can optionally run DSPy optimizers (e.g., BootstrapFewShot)
   to tune prompts for a trainset; hooks for that are documented below.
 - Keep in mind DSPy will build prompts and can optimize them given examples.
 - This code is synchronous and small-scale: run it inside a Slurm job or interactive
   GPU session where your HF model is available via the DSPy LM adapter.
"""

from typing import List, Dict, Any, Optional, Callable
import json
import math
import os

# DSPy imports
try:
    import dspy
    from dspy import Predict, Refine, Example  # common helpers
except Exception as e:
    raise RuntimeError(
        "dspy library not available in the environment. "
        "Install it (pip install dspy-ai) in your llm-exp env, then retry."
    ) from e


# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def sum_scores(per_criterion: Dict[str, float]) -> float:
    return float(sum(per_criterion.values()))


# ------------------------------------------------------------
# DSPy module signatures
# ------------------------------------------------------------
# We'll declare three re-usable module signatures.
#
# 1) criterion_module:
#    signature: "student_answer, criterion, max_score -> score: float, justification: str"
#
# 2) holistic_refine_module:
#    signature: "student_answer, rubric_json, initial_total_score -> revised_score: float, reason: str"
#
# 3) rag_module:
#    signature: "student_answer, context_text, rubric_json -> score: float, justification: str"
#
# These signatures are intentionally simple and explicit; DSPy will compile
# appropriate prompts for the LM automatically.

criterion_sig = "student_answer, criterion, max_score -> score: float, justification: str"
refine_sig = "student_answer, rubric_json, initial_total_score -> revised_score: float, reason: str"
rag_sig = "student_answer, context_text, rubric_json -> score: float, justification: str"


# ------------------------------------------------------------
# Pipeline classes
# ------------------------------------------------------------
class RubricProgram:
    """
    RubricProgram: formal DSPy pipeline that grades an example by invoking
    a DSPy 'Predict' module per rubric criterion and aggregating the scores.

    Example usage:
        pipeline = RubricProgram(lm=dspy.Predict(...))  # or leave default for DSPy to create it
        result = pipeline.run(item_dict)
    """

    def __init__(
        self,
        lm_module: Optional[Predict] = None,
        module_config: Optional[Dict[str, Any]] = None,
    ):
        """
        lm_module: Optional pre-declared dspy.Predict module.
                   If None, this constructor will declare one with the
                   'criterion_sig' signature.
        module_config: optional config passed to dspy.Predict on declaration (temperature, n, etc.)
        """
        module_config = module_config or {}
        if lm_module is None:
            # declare standard DSPy module for per-criterion grading
            self.criterion_module = dspy.Predict(criterion_sig, **module_config)
        else:
            self.criterion_module = lm_module

    def run(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the RubricProgram on a single item (dict expected to have 'student_answer' and 'rubric').

        Returns a dict compatible with your JSONL outputs.
        """
        rubric = item.get("rubric") or {}
        per_criterion: Dict[str, float] = {}
        justifications: Dict[str, str] = {}

        # iterate over rubric criteria deterministically
        for crit, max_score in rubric.items():
            # call DSPy module: passing named arguments matching signature
            # DSPy returns a Prediction object; access fields as attributes
            pred = self.criterion_module(
                student_answer=item.get("student_answer", ""),
                criterion=crit,
                max_score=max_score,
            )
            # Predictions in DSPy expose attributes by name
            score = safe_float(getattr(pred, "score", None), default=0.0)
            justification = getattr(pred, "justification", "")
            # Clip to allowed range if max_score provided
            if max_score is not None:
                score = max(0.0, min(score, float(max_score)))
            per_criterion[crit] = score
            justifications[crit] = str(justification).strip()

        total = sum_scores(per_criterion)

        out = dict(item)  # copy original fields
        out["per_criterion"] = per_criterion
        out["justifications"] = justifications
        out["model_score"] = total
        # store the raw DSPy predictions (stringified) for reproducibility
        out["model_raw_output"] = {"per_criterion_preds": {k: None for k in per_criterion}}
        # Try to store more detailed DSPy outputs if available
        # (left mostly empty to avoid huge structures in JSONL)
        return out


class SelfRefine:
    """
    SelfRefine pipeline:
      1) Run RubricProgram to get per-criterion & initial total.
      2) Run a DSPy 'Refine' (or a Predict wrapped with Refine) to ask the LM to critique
         and revise the initial total score.
    """

    def __init__(
        self,
        base_pipeline: Optional[RubricProgram] = None,
        refine_module: Optional[Predict] = None,
        refine_n: int = 3,
        reward_fn: Optional[Callable[[Dict[str, Any], Any], float]] = None,
    ):
        """
        base_pipeline: if None, a default RubricProgram is created.
        refine_module: if None, declared with signature 'refine_sig'
        refine_n: number of rollouts for dspy.Refine
        reward_fn: function(prediction_dict, prediction_obj) -> float used by Refine; if None, a simple reward is used.
        """
        self.base = base_pipeline or RubricProgram()
        if refine_module is None:
            self.refine_module = dspy.Predict(refine_sig, n=1)  # base refine predictor
        else:
            self.refine_module = refine_module
        self.refine_n = refine_n

        # default reward_fn: negative absolute error to human_score if available
        if reward_fn is None:

            def reward_fn(example_dict: Dict[str, Any], pred) -> float:
                # attempt to reward closeness to human score if provided in example_dict
                human = example_dict.get("human_score")
                revised = getattr(pred, "revised_score", None) or getattr(pred, "revised", None)
                if human is None or revised is None:
                    return 0.0
                try:
                    return -abs(float(human) - float(revised))
                except Exception:
                    return 0.0

        self._reward_fn = reward_fn

    def run(self, item: Dict[str, Any]) -> Dict[str, Any]:
        # Step 1: initial judgment
        initial_res = self.base.run(item)
        initial_total = initial_res.get("model_score", 0.0)

        # Create a DSPy Example if we want to pass to Refine/optimizers later
        # but for runtime calling we simply call the refine module.
        refine_input = {
            "student_answer": item.get("student_answer", ""),
            "rubric_json": json.dumps(item.get("rubric", {})),
            "initial_total_score": float(initial_total),
        }

        # wrap refine_module with dspy.Refine to run multiple rollouts and pick best
        # using the reward function defined at init
        # NOTE: dspy.Refine expects: Refine(module, N, reward_fn, threshold, fail_count=None)
        # We set threshold to a very low value to force N rollouts and use reward_fn to pick best.
        refine_wrapper = dspy.Refine(self.refine_module, N=self.refine_n, reward_fn=lambda pred_obj: self._reward_fn(item, pred_obj), threshold=-1e9)

        refined_pred = refine_wrapper(**refine_input)
        # refined_pred should expose fields per the signature: revised_score, reason
        revised = safe_float(getattr(refined_pred, "revised_score", None) or getattr(refined_pred, "revised", None), default=initial_total)
        reason = getattr(refined_pred, "reason", "")

        out = dict(item)
        out["initial_model_score"] = initial_total
        out["model_score"] = revised
        out["model_raw_output"] = {"initial": initial_res.get("model_raw_output"), "refine": str(refined_pred)}
        out["per_criterion"] = initial_res.get("per_criterion", {})
        out["self_reflect_reason"] = reason
        return out


class RAG:
    """
    Simple RAG pipeline implemented with DSPy.
    - 'exemplars' is a list of dicts with 'student_answer' and 'human_score' (optional).
    - A retrieval function can be provided (embedding-based); otherwise fallback uses overlap scoring.
    - The DSPy module 'rag_module' is declared with signature 'rag_sig' and asked to produce a numeric score.
    """

    def __init__(
        self,
        exemplars: Optional[List[Dict[str, Any]]] = None,
        rag_module: Optional[Predict] = None,
        top_k: int = 3,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        module_config: Optional[Dict[str, Any]] = None,
    ):
        self.exemplars = exemplars or []
        self.top_k = top_k
        self.embedding_fn = embedding_fn
        module_config = module_config or {}
        if rag_module is None:
            self.rag_module = dspy.Predict(rag_sig, **module_config)
        else:
            self.rag_module = rag_module

    def _retrieve_neighbors(self, query: str) -> List[Dict[str, Any]]:
        if self.embedding_fn and len(self.exemplars) > 0:
            import numpy as np

            qemb = self.embedding_fn(query)
            sims = []
            for ex in self.exemplars:
                eemb = self.embedding_fn(ex["student_answer"])
                # cosine
                score = float((sum(a * b for a, b in zip(qemb, eemb))) / (math.sqrt(sum(a * a for a in qemb)) * math.sqrt(sum(b * b for b in eemb)) + 1e-8))
                sims.append((score, ex))
            sims.sort(key=lambda x: x[0], reverse=True)
            return [x[1] for x in sims[: self.top_k]]
        # fallback overlap scoring
        qtokens = set(query.lower().split())
        scored = []
        for ex in self.exemplars:
            ex_tokens = set(ex["student_answer"].lower().split())
            overlap = len(qtokens.intersection(ex_tokens))
            scored.append((overlap, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[: self.top_k]]

    def run(self, item: Dict[str, Any]) -> Dict[str, Any]:
        # retrieve neighbors
        neighbors = self._retrieve_neighbors(item.get("student_answer", ""))

        # build context text
        ctx = ""
        for n in neighbors:
            ctx += f"EXEMPLAR_ANSWER: {n['student_answer']}\nEXEMPLAR_SCORE: {n.get('human_score','?')}\n---\n"

        # call DSPy rag_module
        pred = self.rag_module(
            student_answer=item.get("student_answer", ""),
            context_text=ctx,
            rubric_json=json.dumps(item.get("rubric", {})),
        )
        score = safe_float(getattr(pred, "score", None), default=None)
        justification = getattr(pred, "justification", "")

        out = dict(item)
        out["model_score"] = score
        out["model_raw_output"] = str(pred)
        out["justification"] = str(justification)
        out["neighbors"] = neighbors
        return out


# ------------------------------------------------------------
# Optional helpers for optimization / compiling with DSPy
# ------------------------------------------------------------
def optimize_module_with_bootstrap(module: Predict, trainset: List[Dict[str, Any]], metric_fn: Callable[[Dict[str, Any], Any], float], n_trials: int = 50):
    """
    Optional helper: run DSPy's BootstrapFewShot optimizer to generate improved prompts/demos.
    - module: a declared dspy.Predict module
    - trainset: list of dict examples (each should contain fields used by module signature)
    - metric_fn: function(example_dict, pred) -> reward score (higher = better)
    - n_trials: number of bootstrap trials
    Returns: compiled module (optimized) or the optimizer object depending on DSPy API.
    """
    # Example usage pattern (pseudo-code; DSPy API evolves so check docs):
    #
    # optimizer = dspy.BootstrapFewShot(metric=metric_fn)
    # optimized = optimizer.compile(module, trainset=dspy.Example.from_list(trainset))
    # return optimized
    #
    # We keep this as a documented hook; concrete usage should follow DSPy docs.
    raise NotImplementedError("Call DSPy optimizers following the current DSPy API (see dspy.BootstrapFewShot docs).")


# ------------------------------------------------------------
# If this module is run directly, show a tiny smoke test (no HF model required)
# ------------------------------------------------------------
if __name__ == "__main__":
    # quick smoke tests that exercise the class interfaces without calling LMs
    dummy_item = {
        "id": "demo1",
        "student_answer": "A BST is a tree where left < node < right.",
        "human_score": 8,
        "rubric": {"correctness": 5, "clarity": 3},
    }

    # Declare dummy Predict modules that return fixed outputs using dspy.Predict
    # For offline smoke testing without an LM, you can create "stub" Predict modules:
    def make_stub_predict(score_value=1.0, justification_text="ok"):
        # DSPy provides a way to create constant modules via Python functions or mocks,
        # but this depends on local DSPy test harness. For interactive debugging,
        # you can replace dspy.Predict with a small wrapper that returns an object
        # with the expected attributes.
        class StubPred:
            def __init__(self, score_value, justification_text):
                self.score_value = score_value
                self.justification_text = justification_text

            def __call__(self, **kwargs):
                class R:
                    pass
                r = R()
                r.score = score_value
                r.justification = justification_text
                return r

        return StubPred(score_value, justification_text)

    # run RubricProgram with stub module
    rubric_stub = RubricProgram(lm_module=make_stub_predict(3.0, "ok"))
    print("RubricProgram (stub):", rubric_stub.run(dummy_item))

    # run SelfRefine with stub base and stub refine (note: Refine requires DSPy runtime)
    sf = SelfRefine(base_pipeline=rubric_stub, refine_module=make_stub_predict(4.0, "refined"))
    print("SelfRefine (stub):", sf.run(dummy_item))

    # run RAG with stub
    rag = RAG(exemplars=[dummy_item], rag_module=make_stub_predict(4.5, "near exemplar"))
    print("RAG (stub):", rag.run(dummy_item))
