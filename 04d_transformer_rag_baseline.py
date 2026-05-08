#!/usr/bin/env python3
"""
04d_transformer_with_rag_baseline.py
=====================================
Zero-shot transformer baseline WITH retrieved evidence (BART + RAG).

This is the third baseline:
  Baseline 1: keyword matching          (no LLM, no RAG)
  Baseline 2: BART-MNLI zero-shot       (transformer, no RAG)
  Baseline 3: BART-MNLI zero-shot + RAG (transformer, RAG)  <-- THIS SCRIPT
  Full:       Claude Haiku + RAG        (LLM, RAG)

By feeding the same retrieved biomedical evidence to BART that the
LLM+RAG pipeline uses, this baseline isolates the contribution of the
language-model component: any gap between BART+RAG and Claude+RAG is
attributable to LLM capability, not to evidence access.

BART-MNLI has a 1024-token context limit, so we feed:
  - The claim verbatim (truncated to ~200 chars)
  - The product and claimed effect
  - The top-1 PubMed article (title + first 400 chars of abstract)
  - A short FDA adverse-event summary (top 3 reactions)
  - A flag for whether an NIH fact sheet exists

This is a fair, compact summary of the same evidence available to the
LLM+RAG pipeline.

Input:  data/03_classified/classified_claims.jsonl
Output: data/04_evaluation/results/baseline_transformer_rag_predictions.jsonl
        data/04_evaluation/results/baseline_transformer_rag_results.json

Run: ~15 minutes on GPU, ~2-3 hours on CPU.
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

PREDICTIONS_PATH = OUTPUT_DIR / "baseline_transformer_rag_predictions.jsonl"
RESULTS_PATH = OUTPUT_DIR / "baseline_transformer_rag_results.json"

LABEL_HYPOTHESES = {
    "SUPPORTED":    "This health claim is supported by scientific evidence.",
    "UNSUPPORTED":  "This health claim has no scientific evidence supporting it.",
    "EXAGGERATED":  "This health claim overstates or exaggerates the actual effect.",
    "CONTRADICTED": "This health claim is contradicted by scientific evidence.",
    "DANGEROUS":    "This health claim promotes a dangerous or harmful practice.",
}
LABELS = list(LABEL_HYPOTHESES.keys())
HYPOTHESES = list(LABEL_HYPOTHESES.values())


def build_premise_with_evidence(claim: dict) -> str:
    """
    Build a premise that includes both the claim and a compact summary
    of the retrieved evidence, fitting within BART's 1024-token limit.
    """
    quote   = (claim.get("verbatim_quote") or "")[:200]
    product = claim.get("product") or ""
    effect  = (claim.get("claimed_effect") or "")[:150]

    parts = []
    parts.append(f"Claim about {product}: {quote if quote else effect}")

    ev = claim.get("evidence", {}) or {}

    # Top PubMed article: title + first 400 chars of abstract
    pubmed = ev.get("pubmed_articles", []) or []
    if pubmed:
        top = pubmed[0]
        title = (top.get("title") or "").strip()
        abstract = (top.get("abstract") or "").strip()[:400]
        if title or abstract:
            parts.append(f"PubMed evidence: {title} {abstract}".strip())
    else:
        parts.append("PubMed evidence: no relevant systematic reviews or trials found.")

    # FDA adverse events: top 3 reactions
    fda = ev.get("fda_events", []) or []
    if fda:
        reactions = []
        for event in fda[:3]:
            for r in (event.get("reactions") or [])[:3]:
                if r and r not in reactions:
                    reactions.append(r)
        if reactions:
            parts.append(
                f"FDA adverse events recorded: {', '.join(reactions[:6])}."
            )

    # NIH fact sheet
    nih = ev.get("nih_reference", {}) or {}
    if nih:
        parts.append("An NIH fact sheet exists for this product.")

    return " | ".join(parts)


def main():
    logger.info("=" * 60)
    logger.info("Stage 04d: Zero-Shot Transformer Baseline + RAG")
    logger.info("=" * 60)

    try:
        from transformers import pipeline
        import torch
    except ImportError:
        logger.error("Run: pip install transformers torch")
        return

    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Device: {'GPU (CUDA)' if device == 0 else 'CPU'}")

    logger.info("Loading model: facebook/bart-large-mnli...")
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

    confusion = defaultdict(lambda: Counter())
    pred_counter = Counter()
    n_processed = 0
    n_skipped = 0
    n_truncated = 0
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

            premise = build_premise_with_evidence(claim)
            if not premise.strip():
                n_skipped += 1
                continue

            # BART has a hard 1024-token limit; flag (not truncate manually,
            # the pipeline handles it but we count for transparency)
            if len(premise) > 3000:  # ~750 tokens upper bound, safe margin
                premise = premise[:3000]
                n_truncated += 1

            try:
                result = classifier(
                    premise,
                    candidate_labels=HYPOTHESES,
                    multi_label=False,
                    truncation=True,  # let pipeline truncate to model max
                )
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
                "pred_label_transformer_rag": pred_label,
                "pred_score": float(top_score),
                "had_pubmed_evidence": bool(claim.get("evidence", {}).get("pubmed_articles")),
                "had_fda_evidence": bool(claim.get("evidence", {}).get("fda_events")),
                "had_nih_evidence": bool(claim.get("evidence", {}).get("nih_reference")),
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
                f"pre-truncated {n_truncated}, in {elapsed/60:.1f} min")

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

    rag_dist = Counter()
    for c in claims:
        v = c.get("classification", {}).get("llm_grounded", {}).get("veracity")
        if v in LABELS:
            rag_dist[v] += 1

    results = {
        "model": "facebook/bart-large-mnli (zero-shot, no fine-tuning) + RAG evidence",
        "evidence_format": "claim + top-1 PubMed (title + 400 chars abstract) + FDA reactions (top 6) + NIH flag",
        "n_claims_evaluated": n_processed,
        "n_skipped": n_skipped,
        "n_pre_truncated": n_truncated,
        "runtime_minutes": round(elapsed / 60, 2),
        "accuracy_vs_llm_rag": round(accuracy, 3),
        "macro_f1_vs_llm_rag": round(macro_f1, 3),
        "per_class_metrics": per_class,
        "confusion_matrix": {
            true_lbl: dict(confusion[true_lbl]) for true_lbl in LABELS
        },
        "transformer_rag_distribution": dict(pred_counter),
        "llm_rag_distribution": dict(rag_dist),
        "labels": LABELS,
        "label_hypotheses": LABEL_HYPOTHESES,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY (BART + RAG)")
    logger.info("=" * 60)
    logger.info(f"Accuracy vs LLM+RAG: {accuracy:.3f}")
    logger.info(f"Macro F1 vs LLM+RAG: {macro_f1:.3f}")
    logger.info("\nPer-class metrics:")
    for lbl in LABELS:
        m = per_class[lbl]
        logger.info(
            f"  {lbl:14s} P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"F1={m['f1']:.3f}  (n={m['support']})"
        )
    logger.info("\nDistribution comparison:")
    logger.info(f"  {'Label':14s}  {'LLM+RAG':>8s}  {'BART+RAG':>10s}")
    for lbl in LABELS:
        logger.info(
            f"  {lbl:14s}  {rag_dist.get(lbl, 0):>8d}  "
            f"{pred_counter.get(lbl, 0):>10d}"
        )
    logger.info(f"\nResults: {RESULTS_PATH}")
    logger.info(f"Predictions: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
