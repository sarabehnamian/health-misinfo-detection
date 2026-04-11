#!/usr/bin/env python3
"""
03_veracity_classification.py
=============================
Classify health claims by veracity using LLM grounded against retrieved evidence.

This is the core RAG (Retrieval-Augmented Generation) step:
- Each claim is paired with its retrieved PubMed/FDA/NIH evidence
- The LLM evaluates the claim against the evidence
- Output: veracity label, confidence, reasoning, risk tier

Input:  data/02_evidence/claims_with_evidence.jsonl
Output: data/03_classified/classified_claims.jsonl

Veracity labels:
  - SUPPORTED:    Evidence confirms the claim
  - UNSUPPORTED:  No evidence found for or against
  - EXAGGERATED:  Partial truth but overstated
  - CONTRADICTED: Evidence directly refutes the claim
  - DANGEROUS:    Contradicts safety data or promotes harmful behavior
"""

import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_DIR = Path("data/02_evidence")
OUTPUT_DIR = Path("data/03_classified")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Classification Prompt (RAG)
# ============================================================
CLASSIFICATION_PROMPT = """You are a biomedical fact-checker. Evaluate the following health claim against the provided scientific evidence.

HEALTH CLAIM:
Product: {product}
Claimed effect: {claimed_effect}
Target condition: {target_condition}
Claim strength: {claim_strength}
Verbatim quote: "{verbatim_quote}"

RETRIEVED EVIDENCE:
{evidence_text}

Based on the evidence above, classify this claim into ONE of these categories:
1. SUPPORTED - The evidence confirms this claim with high-quality studies
2. UNSUPPORTED - No relevant evidence was found to confirm or deny
3. EXAGGERATED - There is some evidence but the claim overstates the effect (e.g., "cures" vs "may help")
4. CONTRADICTED - The evidence directly refutes this claim
5. DANGEROUS - The claim promotes behavior that contradicts safety data (drug interactions, toxicity, delay of proven treatment)

Also assess the RISK TIER based on potential harm:
- LOW: Unlikely to cause harm (e.g., "vitamin C helps immunity")
- MODERATE: Could waste money or delay treatment (e.g., "turmeric replaces statins")
- HIGH: Could cause direct harm (e.g., "bleach cures COVID", "stop your medication")
- CRITICAL: Targets vulnerable populations with dangerous advice (e.g., cancer patients, pregnant women, children)

Respond with ONLY valid JSON:
{{
    "veracity": "SUPPORTED|UNSUPPORTED|EXAGGERATED|CONTRADICTED|DANGEROUS",
    "confidence": 0.0-1.0,
    "risk_tier": "LOW|MODERATE|HIGH|CRITICAL",
    "reasoning": "Brief explanation (max 150 words)",
    "key_evidence": "Most relevant piece of evidence supporting the classification",
    "recommendation": "What a health professional would advise"
}}
"""


def format_evidence(claim: dict) -> str:
    """Format retrieved evidence into text for the LLM prompt."""
    evidence = claim.get("evidence", {})
    parts = []
    
    # PubMed articles
    pubmed = evidence.get("pubmed_articles", [])
    if pubmed:
        parts.append("=== PubMed Articles ===")
        for i, article in enumerate(pubmed[:5], 1):
            parts.append(f"[{i}] {article.get('title', 'No title')}")
            parts.append(f"    Journal: {article.get('journal', 'Unknown')}, {article.get('year', '')}")
            parts.append(f"    Types: {', '.join(article.get('pub_types', []))}")
            abstract = article.get("abstract", "")
            if abstract:
                parts.append(f"    Abstract: {abstract[:500]}")
            parts.append("")
    else:
        parts.append("=== PubMed: No relevant systematic reviews or RCTs found ===")
    
    # FDA adverse events
    fda = evidence.get("fda_events", [])
    if fda:
        parts.append("=== FDA Adverse Events ===")
        for event in fda[:3]:
            reactions = ", ".join(event.get("reactions", []))
            parts.append(f"  Reactions: {reactions}")
            parts.append(f"  Serious: {event.get('serious', 'Unknown')}")
        parts.append("")
    
    # NIH reference
    nih = evidence.get("nih_reference", {})
    if nih:
        parts.append("=== NIH/NCCIH Reference ===")
        parts.append(f"  {nih.get('note', '')}")
        parts.append(f"  URL: {nih.get('reference_url', '')}")
        parts.append("")
    
    if not parts or all("No relevant" in p or "===" in p for p in parts):
        parts.append("NO EVIDENCE FOUND in PubMed, FDA, or NIH databases.")
    
    return "\n".join(parts)


def classify_claim(claim: dict, llm_provider: str = "anthropic") -> dict:
    """Classify a single claim using LLM + evidence."""
    
    evidence_text = format_evidence(claim)
    
    prompt = CLASSIFICATION_PROMPT.format(
        product=claim.get("product", "unknown"),
        claimed_effect=claim.get("claimed_effect", "unknown"),
        target_condition=claim.get("target_condition", "unknown"),
        claim_strength=claim.get("claim_strength", "unknown"),
        verbatim_quote=claim.get("verbatim_quote", "")[:200],
        evidence_text=evidence_text,
    )
    
    if llm_provider == "anthropic":
        return _classify_with_anthropic(prompt)
    else:
        return _classify_with_openai(prompt)


def _classify_with_anthropic(prompt: str) -> dict:
    """Use Claude for classification."""
    import os
    from anthropic import Anthropic
    
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        logger.warning(f"Classification error: {e}")
        return {
            "veracity": "UNSUPPORTED",
            "confidence": 0.0,
            "risk_tier": "LOW",
            "reasoning": f"Classification failed: {e}",
            "key_evidence": "",
            "recommendation": "",
        }


def _classify_with_openai(prompt: str) -> dict:
    """Use GPT-4 for classification."""
    import os
    from openai import OpenAI
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a biomedical fact-checker. Respond only with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=1000,
        )
        text = response.choices[0].message.content.strip()
        return json.loads(text)
    except Exception as e:
        logger.warning(f"Classification error: {e}")
        return {"veracity": "UNSUPPORTED", "confidence": 0.0, "risk_tier": "LOW",
                "reasoning": str(e), "key_evidence": "", "recommendation": ""}


# ============================================================
# Baseline: Keyword-only classifier (no LLM)
# ============================================================
DANGEROUS_KEYWORDS = [
    "cure cancer", "cures cancer", "kills cancer",
    "stop taking", "replace medication", "instead of chemo",
    "bleach", "turpentine", "borax", "hydrogen peroxide therapy",
    "miracle cure", "100% effective", "guaranteed",
    "big pharma doesn't want", "doctors don't want you to know",
    "FDA approved" ,  # often false claim
]

EXAGGERATION_KEYWORDS = [
    "proven to cure", "scientifically proven", "clinically proven",
    "eliminates", "eradicates", "reverses", "completely heals",
]


def classify_baseline(claim: dict) -> dict:
    """Simple keyword-based baseline classifier for comparison."""
    text = f"{claim.get('claimed_effect', '')} {claim.get('verbatim_quote', '')}".lower()
    
    for kw in DANGEROUS_KEYWORDS:
        if kw in text:
            return {"veracity": "DANGEROUS", "confidence": 0.7, "risk_tier": "HIGH",
                    "reasoning": f"Keyword match: '{kw}'", "method": "keyword_baseline"}
    
    for kw in EXAGGERATION_KEYWORDS:
        if kw in text:
            return {"veracity": "EXAGGERATED", "confidence": 0.5, "risk_tier": "MODERATE",
                    "reasoning": f"Keyword match: '{kw}'", "method": "keyword_baseline"}
    
    return {"veracity": "UNSUPPORTED", "confidence": 0.3, "risk_tier": "LOW",
            "reasoning": "No strong signal detected", "method": "keyword_baseline"}


# ============================================================
# Process All Claims
# ============================================================
def classify_all_claims(input_path: Path, output_path: Path, llm_provider: str = "anthropic"):
    """Classify all claims with both LLM+RAG and baseline."""
    
    claims = []
    with open(input_path, encoding='utf-8') as f:
        for line in f:
            claims.append(json.loads(line))
    
    logger.info(f"Classifying {len(claims)} claims...")
    
    stats = {"SUPPORTED": 0, "UNSUPPORTED": 0, "EXAGGERATED": 0, "CONTRADICTED": 0, "DANGEROUS": 0}
    risk_stats = {"LOW": 0, "MODERATE": 0, "HIGH": 0, "CRITICAL": 0}
    
    with open(output_path, 'w', encoding='utf-8') as fout:
        for i, claim in enumerate(claims):
            # LLM + RAG classification
            llm_result = classify_claim(claim, llm_provider)
            
            # Baseline classification
            baseline_result = classify_baseline(claim)
            
            # Merge results
            claim["classification"] = {
                "llm_grounded": llm_result,
                "keyword_baseline": baseline_result,
            }
            
            veracity = llm_result.get("veracity", "UNSUPPORTED")
            risk = llm_result.get("risk_tier", "LOW")
            stats[veracity] = stats.get(veracity, 0) + 1
            risk_stats[risk] = risk_stats.get(risk, 0) + 1
            
            fout.write(json.dumps(claim) + "\n")
            
            if (i + 1) % 20 == 0:
                logger.info(f"  Classified {i+1}/{len(claims)}")
            
            time.sleep(0.5)
    
    logger.info(f"\nVeracity distribution: {json.dumps(stats, indent=2)}")
    logger.info(f"Risk distribution: {json.dumps(risk_stats, indent=2)}")
    logger.info(f"Output -> {output_path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Stage 03: Veracity Classification (LLM + RAG)")
    logger.info("=" * 60)
    
    input_path = INPUT_DIR / "claims_with_evidence.jsonl"
    output_path = OUTPUT_DIR / "classified_claims.jsonl"
    
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}. Run 02_evidence_retrieval.py first.")
        exit(1)
    
    classify_all_claims(input_path, output_path, llm_provider="anthropic")
