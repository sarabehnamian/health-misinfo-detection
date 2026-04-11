#!/usr/bin/env python3
"""
02_evidence_retrieval.py
========================
Retrieve biomedical evidence for each extracted health claim.

Sources:
- PubMed/MEDLINE (Entrez esearch/efetch), query biased to SR/meta-analysis/RCT publication types
- openFDA drug adverse event reports (drug/event.json; not FDA warning letters)
- NIH ODS / NCCIH fact-sheet URLs via a static product-name map

Input:  data/01_claims/extracted_claims.jsonl
Output: data/02_evidence/claims_with_evidence.jsonl
"""

import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen, Request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_DIR = Path("data/01_claims")
OUTPUT_DIR = Path("data/02_evidence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# PubMed / Entrez API
# ============================================================
ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ENTREZ_EMAIL = "sara.behnamian@sund.ku.dk"  # Required by NCBI


def search_pubmed(query: str, max_results: int = 5) -> list:
    """
    Search PubMed for articles relevant to a health claim.
    Prioritizes systematic reviews and meta-analyses.
    """
    # Build search query prioritizing high-quality evidence
    search_query = f"({query}) AND (systematic review[pt] OR meta-analysis[pt] OR randomized controlled trial[pt])"
    
    url = (
        f"{ENTREZ_BASE}/esearch.fcgi?"
        f"db=pubmed&term={quote(search_query)}"
        f"&retmax={max_results}&sort=relevance"
        f"&email={ENTREZ_EMAIL}&tool=health_misinfo_detector"
    )
    
    try:
        with urlopen(Request(url), timeout=15) as resp:
            tree = ET.parse(resp)
        
        id_list = tree.findall(".//Id")
        pmids = [id_elem.text for id_elem in id_list]
        
        if not pmids:
            return []
        
        return fetch_pubmed_details(pmids)
        
    except Exception as e:
        logger.warning(f"PubMed search error for '{query[:50]}': {e}")
        return []


def fetch_pubmed_details(pmids: list) -> list:
    """Fetch article details (title, abstract, journal) for given PMIDs."""
    
    ids_str = ",".join(pmids)
    url = (
        f"{ENTREZ_BASE}/efetch.fcgi?"
        f"db=pubmed&id={ids_str}&rettype=xml"
        f"&email={ENTREZ_EMAIL}&tool=health_misinfo_detector"
    )
    
    articles = []
    try:
        with urlopen(Request(url), timeout=30) as resp:
            tree = ET.parse(resp)
        
        for article in tree.findall(".//PubmedArticle"):
            title_elem = article.find(".//ArticleTitle")
            abstract_elem = article.find(".//AbstractText")
            journal_elem = article.find(".//Journal/Title")
            pmid_elem = article.find(".//PMID")
            year_elem = article.find(".//PubDate/Year")
            pub_type_elems = article.findall(".//PublicationType")
            
            pub_types = [pt.text for pt in pub_type_elems if pt.text]
            
            articles.append({
                "pmid": pmid_elem.text if pmid_elem is not None else "",
                "title": title_elem.text if title_elem is not None else "",
                "abstract": abstract_elem.text[:1000] if abstract_elem is not None and abstract_elem.text else "",
                "journal": journal_elem.text if journal_elem is not None else "",
                "year": year_elem.text if year_elem is not None else "",
                "pub_types": pub_types,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid_elem.text}/" if pmid_elem is not None else "",
            })
        
    except Exception as e:
        logger.warning(f"PubMed fetch error: {e}")
    
    return articles


# ============================================================
# FDA Warning Letters and Adverse Events
# ============================================================
def search_fda_warnings(product: str) -> list:
    """
    Search FDA warning letters related to a product.
    Uses openFDA API.
    """
    url = (
        f"https://api.fda.gov/drug/event.json?"
        f"search=patient.drug.medicinalproduct:\"{quote(product)}\"&limit=3"
    )
    
    try:
        with urlopen(Request(url), timeout=15) as resp:
            data = json.loads(resp.read())
        
        results = []
        for event in data.get("results", []):
            results.append({
                "source": "FDA_adverse_event",
                "product": product,
                "reactions": [r.get("reactionmeddrapt", "") for r in event.get("patient", {}).get("reaction", [])[:5]],
                "serious": event.get("serious", ""),
                "report_date": event.get("receivedate", ""),
            })
        return results
        
    except Exception as e:
        logger.debug(f"FDA search for '{product}': {e}")
        return []


# ============================================================
# NIH Office of Dietary Supplements
# ============================================================
# Common supplements with NIH ODS fact sheet URLs
NIH_ODS_SUPPLEMENTS = {
    "vitamin d": "https://ods.od.nih.gov/factsheets/VitaminD-HealthProfessional/",
    "vitamin c": "https://ods.od.nih.gov/factsheets/VitaminC-HealthProfessional/",
    "zinc": "https://ods.od.nih.gov/factsheets/Zinc-HealthProfessional/",
    "iron": "https://ods.od.nih.gov/factsheets/Iron-HealthProfessional/",
    "magnesium": "https://ods.od.nih.gov/factsheets/Magnesium-HealthProfessional/",
    "omega-3": "https://ods.od.nih.gov/factsheets/Omega3FattyAcids-HealthProfessional/",
    "fish oil": "https://ods.od.nih.gov/factsheets/Omega3FattyAcids-HealthProfessional/",
    "probiotics": "https://ods.od.nih.gov/factsheets/Probiotics-HealthProfessional/",
    "turmeric": "https://www.nccih.nih.gov/health/turmeric",
    "ashwagandha": "https://www.nccih.nih.gov/health/ashwagandha",
    "melatonin": "https://www.nccih.nih.gov/health/melatonin-what-you-need-to-know",
    "cbd": "https://www.nccih.nih.gov/health/cannabis-marijuana-and-cannabinoids-what-you-need-to-know",
    "kratom": "https://www.nccih.nih.gov/health/kratom",
    "green tea": "https://www.nccih.nih.gov/health/green-tea",
    "echinacea": "https://www.nccih.nih.gov/health/echinacea",
    "garlic": "https://www.nccih.nih.gov/health/garlic",
    "ginseng": "https://www.nccih.nih.gov/health/asian-ginseng",
    "st john's wort": "https://www.nccih.nih.gov/health/st-johns-wort",
    "colloidal silver": "https://www.nccih.nih.gov/health/colloidal-silver",
}


def get_nih_reference(product: str) -> dict:
    """Check if NIH/NCCIH has a fact sheet for this supplement."""
    product_lower = product.lower().strip()
    
    for key, url in NIH_ODS_SUPPLEMENTS.items():
        if key in product_lower or product_lower in key:
            return {
                "source": "NIH_ODS",
                "product": product,
                "reference_url": url,
                "note": f"NIH/NCCIH fact sheet available for {key}",
            }
    return {}


# ============================================================
# Build Search Query from Claim
# ============================================================
def build_pubmed_query(claim: dict) -> str:
    """
    Construct a PubMed search query from a structured claim.
    """
    product = claim.get("product", "")
    condition = claim.get("target_condition", "")
    effect = claim.get("claimed_effect", "")
    
    # Build focused query
    parts = []
    if product:
        parts.append(product)
    if condition and condition != "general health":
        parts.append(condition)
    elif effect:
        # Extract key terms from effect
        parts.append(effect)
    
    return " ".join(parts)


# ============================================================
# Process All Claims
# ============================================================
def retrieve_evidence_for_claims(input_path: Path, output_path: Path):
    """Retrieve evidence for each extracted claim."""
    
    claims = []
    with open(input_path, encoding='utf-8') as f:
        for line in f:
            claims.append(json.loads(line))
    
    logger.info(f"Retrieving evidence for {len(claims)} claims...")
    
    with open(output_path, 'w', encoding='utf-8') as fout:
        for i, claim in enumerate(claims):
            # 1. PubMed search
            query = build_pubmed_query(claim)
            if query:
                pubmed_results = search_pubmed(query, max_results=5)
                time.sleep(0.4)  # NCBI rate limit: 3 req/sec without API key
            else:
                pubmed_results = []
            
            # 2. FDA adverse events
            product = claim.get("product", "")
            fda_results = []
            if product:
                fda_results = search_fda_warnings(product)
                time.sleep(0.3)
            
            # 3. NIH/NCCIH reference
            nih_ref = get_nih_reference(product) if product else {}
            
            # Combine evidence
            claim["evidence"] = {
                "pubmed_articles": pubmed_results,
                "fda_events": fda_results,
                "nih_reference": nih_ref,
                "pubmed_query": query,
                "evidence_count": len(pubmed_results) + len(fda_results) + (1 if nih_ref else 0),
            }
            
            fout.write(json.dumps(claim) + "\n")
            
            if (i + 1) % 20 == 0:
                logger.info(f"  Processed {i+1}/{len(claims)} claims")
    
    logger.info(f"Evidence retrieved for {len(claims)} claims -> {output_path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Stage 02: Evidence Retrieval")
    logger.info("=" * 60)
    
    input_path = INPUT_DIR / "extracted_claims.jsonl"
    output_path = OUTPUT_DIR / "claims_with_evidence.jsonl"
    
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}. Run 01_claim_extraction.py first.")
        exit(1)
    
    retrieve_evidence_for_claims(input_path, output_path)
