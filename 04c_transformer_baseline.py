#!/usr/bin/env python3
"""
04c_transformer_baseline.py
============================
Zero-shot transformer baseline for veracity classification.

Why this baseline:
  Reviewer 2 noted that the keyword baseline is too weak. We add a
  pretrained transformer (BART-large-MNLI) as a stronger, *non-trained*
  baseline that classifies each claim against the same 5 veracity labels
  using natural language inference, *without* access to retrieved
  biomedical evidence. This isolates the contribution of the RAG
  evidence-grounding stage: any gap between this baseline and the
  LLM+RAG pipeline is attributable to evidence grounding, not to
  linguistic representation.

Input:  data/03_classified/classified_claims.jsonl
Output: data/04_evaluation/results/baseline_transformer_predictions.jsonl
        data/04_evaluation/results/baseline_transformer_results.json

Run on CPU: ~1-2 hours for 8,250 claims (acceptable, runs once)
Run on GPU: ~5-10 minutes
"""

import json
import logging
import time
from pathlib import Path
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/03_classified/classified_claims.jsonl")
OUTPUT_DIR = Path("data/04_evaluation/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = OUTPUT_DIR / "baseline_transformer_predictions.jsonl"
RESULTS_PATH = OUTPUT_DIR / "baseline_transformer_results.json"

# Five-class hypothesis templates that the NLI model will score
# against each claim. The verbalizations matter for zero-shot quality.
LABEL_HYPOTHESES = {
    "SUPPORTED":    "This health claim is supported by scientific evidence.",
    "UNSUPPORTED":  "This health claim has no scientific evidence supporting it.",
    "EXAGGERATED":  "This health claim overstates or exaggerates the actual effect.",
    "CONTRADICTED": "This health claim is contradicted by scientific evidence.",
    "DANGEROUS":    "This health claim promotes a dangerous or harmful practice.",
}
LABELS = list(LABEL_HYPOTHESES.keys())
HYPOTHESES = list(LABEL_HYPOTHESES.values())


def build_premise(claim: dict) -> str:
    """Build the input text from a claim record."""
    quote   = claim.get("verbatim_quote", "") or ""
    product = claim.get("product", "") or ""
    effect  = claim.get("claimed_effect", "") or ""
    if quote:
        return f"Claim about {product}: {quote}"
    return f"Claim about {product}: {product} {effect}"


def main():
    logger.info("=" * 60)
    logger.info("Stage 04c: Zero-Shot Transformer Baseline")
    logger.info("=" * 60)

    # Lazy imports so the script fails clearly if transformers isn't installed
    try:
        from transformers import pipeline
        import torch
    except ImportError:
        logger.error("Run first: pip install transformers torch")
        return

    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Device: {'GPU (CUDA)' if device == 0 else 'CPU'}")

    logger.info("Loading model: facebook/bart-large-mnli (downloads ~1.6 GB on first run)...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
    )

    if not INPUT_PATH.exists():
        logger.error(f"Input not found: {INPUT_PATH}")
        return

    claims = [json.loads(l) for l in open(INPUT_PATH, encoding="utf-8")]
    logger.info(f"Loaded {len(claims)} claims")

    # Score each claim
    confusion = defaultdict(lambda: Counter())  # confusion[true][pred]
    pred_counter = Counter()
    n_processed = 0
    n_skipped = 0
    t0 = time.time()

    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as fout:
        for i, claim in enumerate(claims):
            true_label = (
                claim.get("classification", {})
                .get("llm_grounded", {})
                .get("veracity")
            )
            if true_label not in LABELS:
                n_skipped += 1
                continue

            premise = build_premise(claim)
            if not premise.strip():
                n_skipped += 1
                continue

            try:
                result = classifier(
                    premise,
                    candidate_labels=HYPOTHESES,
                    multi_label=False,
                )
                # Map back from hypothesis text to label key
                top_hyp = result["labels"][0]
                top_score = result["scores"][0]
                pred_label = next(
                    k for k, v in LABEL_HYPOTHESES.items() if v == top_hyp
                )
            except Exception as e:
                logger.warning(f"Skipping claim {i} due to error: {e}")
                n_skipped += 1
                continue

            confusion[true_label][pred_label] += 1
            pred_counter[pred_label] += 1
            n_processed += 1

            fout.write(json.dumps({
                "source_post_id": claim.get("source_post_id"),
                "product": claim.get("product"),
                "claimed_effect": claim.get("claimed_effect"),
                "true_label_llm_rag": true_label,
                "pred_label_transformer": pred_label,
                "pred_score": float(top_score),
            }) + "\n")

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-9)
                eta = (len(claims) - i - 1) / max(rate, 1e-9)
                logger.info(
                    f"  {i+1}/{len(claims)} "
                    f"({rate:.1f} claims/s, ETA {eta/60:.1f} min)"
                )

    elapsed = time.time() - t0
    logger.info(f"Done. Processed {n_processed}, skipped {n_skipped}, "
                f"in {elapsed/60:.1f} min")

    # ====================================================
    # Compute metrics
    # ====================================================
    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[lbl][lbl] for lbl in LABELS)
    accuracy = correct / total if total else 0.0

    per_class = {}
    f1_list = []
    for lbl in LABELS:
        tp = confusion[lbl][lbl]
        fp = sum(confusion[other][lbl] for other in LABELS if other != lbl)
        fn = sum(confusion[lbl][other] for other in LABELS if other != lbl)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[lbl] = {
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1":        round(f1, 3),
            "support":   sum(confusion[lbl].values()),
        }
        f1_list.append(f1)
    macro_f1 = sum(f1_list) / len(f1_list)

    # Compare prediction distributions
    rag_dist = Counter()
    for c in claims:
        v = c.get("classification", {}).get("llm_grounded", {}).get("veracity")
        if v in LABELS:
            rag_dist[v] += 1

    results = {
        "model": "facebook/bart-large-mnli (zero-shot, no fine-tuning)",
        "n_claims_evaluated": n_processed,
        "n_skipped": n_skipped,
        "runtime_minutes": round(elapsed / 60, 2),
        "accuracy_vs_llm_rag": round(accuracy, 3),
        "macro_f1_vs_llm_rag": round(macro_f1, 3),
        "per_class_metrics": per_class,
        "confusion_matrix": {
            true_lbl: dict(confusion[true_lbl]) for true_lbl in LABELS
        },
        "transformer_baseline_distribution": dict(pred_counter),
        "llm_rag_distribution": dict(rag_dist),
        "labels": LABELS,
        "label_hypotheses": LABEL_HYPOTHESES,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Print pretty summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Accuracy vs LLM+RAG: {accuracy:.3f}")
    logger.info(f"Macro F1 vs LLM+RAG: {macro_f1:.3f}")
    logger.info("\nPer-class metrics (treating LLM+RAG as reference):")
    for lbl in LABELS:
        m = per_class[lbl]
        logger.info(
            f"  {lbl:14s} P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"F1={m['f1']:.3f}  (n={m['support']})"
        )
    logger.info("\nDistribution comparison:")
    logger.info(f"  {'Label':14s}  {'LLM+RAG':>8s}  {'Transformer':>12s}")
    for lbl in LABELS:
        logger.info(
            f"  {lbl:14s}  {rag_dist.get(lbl, 0):>8d}  "
            f"{pred_counter.get(lbl, 0):>12d}"
        )
    logger.info(f"\nResults saved to: {RESULTS_PATH}")
    logger.info(f"Predictions saved to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
