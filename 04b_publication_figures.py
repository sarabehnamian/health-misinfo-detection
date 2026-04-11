#!/usr/bin/env python3
"""
04_publication_figures.py — Set3 pastel palette, 200 DPI, serif, no legends on plot.
"""

import json, logging
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/03_classified/classified_claims.jsonl")
OUTPUT_DIR = Path("data/04_evaluation/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set3 pastel palette
S3 = ["#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3",
      "#fdb462","#b3de69","#fccde5","#d9d9d9","#bc80bd","#ccebc5","#ffed6f"]

VKEYS = ["SUPPORTED","EXAGGERATED","UNSUPPORTED","CONTRADICTED","DANGEROUS"]
VLABELS = ["Supported","Exaggerated","Unsupported","Contradicted","Dangerous"]
VCOLS = [S3[0], S3[5], S3[4], S3[3], S3[9]]  # teal, orange, blue, red, purple

RKEYS = ["LOW","MODERATE","HIGH","CRITICAL"]
RLABELS = ["Low","Moderate","High","Critical"]
RCOLS = [S3[6], S3[1], S3[5], S3[3]]  # green, yellow, orange, red


def load_claims(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def setup():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family":"serif","font.serif":["Times New Roman","DejaVu Serif","Georgia"],
        "font.size":13,"axes.titlesize":16,"axes.labelsize":14,
        "xtick.labelsize":12,"ytick.labelsize":12,
        "figure.dpi":200,"savefig.dpi":200,"savefig.bbox":"tight",
        "savefig.facecolor":"#fafafa","figure.facecolor":"#fafafa",
        "axes.facecolor":"white","axes.spines.top":False,"axes.spines.right":False,
        "axes.edgecolor":"#444","axes.grid":False,"text.color":"#222",
    })
    return plt

def compute_stats(claims):
    st = {"total_claims":len(claims),
          "by_platform":dict(Counter(c.get("source_platform","?") for c in claims)),
          "by_risk_category":dict(Counter(c.get("risk_category","other") for c in claims)),
          "by_claim_strength":dict(Counter(c.get("claim_strength","?") for c in claims))}
    vc,rc,confs = Counter(),Counter(),[]
    for c in claims:
        cl = c.get("classification",{}).get("llm_grounded",{})
        vc[cl.get("veracity","?")] += 1; rc[cl.get("risk_tier","?")] += 1
        cf = cl.get("confidence",0)
        if cf: confs.append(cf)
    st["veracity_distribution"]=dict(vc); st["risk_distribution"]=dict(rc)
    st["mean_confidence"]=float(np.mean(confs)) if confs else 0
    st["median_confidence"]=float(np.median(confs)) if confs else 0
    bc = Counter()
    for c in claims:
        bl = c.get("classification",{}).get("keyword_baseline",{})
        bc[bl.get("veracity","?")] += 1
    st["baseline_veracity_distribution"]=dict(bc)
    with open(OUTPUT_DIR/"descriptive_stats.json","w") as f: json.dump(st,f,indent=2)
    return st


def fig1(stats, plt):
    """Veracity distribution — horizontal bars, Set3 pastel."""
    vals = [stats["veracity_distribution"].get(k,0) for k in VKEYS]
    total = sum(vals)
    fig, ax = plt.subplots(figsize=(10,4.5))
    bars = ax.barh(VLABELS, vals, color=VCOLS, edgecolor="white", linewidth=1, height=0.55)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width()+40, bar.get_y()+bar.get_height()/2,
                f"{val}  ({val/total*100:.1f}%)", va="center", fontsize=11, fontweight="bold", color="#333")
    ax.set_xlabel("Number of Claims", fontsize=14, fontweight="bold")
    ax.set_title("Veracity Distribution (LLM + RAG)", fontsize=16, fontweight="bold", pad=18)
    ax.invert_yaxis(); ax.set_xlim(0, max(vals)*1.25)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR/"fig1_veracity_distribution.png"); plt.close()
    logger.info("  fig1")


def fig2(stats, plt):
    """LLM+RAG vs Baseline — paired horizontal."""
    llm_vals = [stats["veracity_distribution"].get(k,0) for k in VKEYS]
    bl_vals  = [stats["baseline_veracity_distribution"].get(k,0) for k in VKEYS]
    y = np.arange(len(VLABELS)); h = 0.35
    fig, ax = plt.subplots(figsize=(10,5))
    ax.barh(y-h/2, llm_vals, h, color=VCOLS, edgecolor="white", linewidth=0.8)
    ax.barh(y+h/2, bl_vals,  h, color="#d9d9d9", edgecolor="white", linewidth=0.8)
    for i,(lv,bv) in enumerate(zip(llm_vals,bl_vals)):
        ax.text(lv+30, i-h/2, f"{lv}", va="center", fontsize=10, fontweight="bold", color="#333")
        if bv>0: ax.text(bv+30, i+h/2, f"{bv}", va="center", fontsize=10, color="#888")
    ax.set_yticks(y); ax.set_yticklabels(VLABELS, fontsize=12)
    ax.set_xlabel("Number of Claims", fontsize=14, fontweight="bold")
    ax.set_title("LLM+RAG (colored) vs. Keyword Baseline (gray)", fontsize=15, fontweight="bold", pad=18)
    ax.invert_yaxis()
    plt.tight_layout(); plt.savefig(OUTPUT_DIR/"fig2_llm_vs_baseline.png"); plt.close()
    logger.info("  fig2")


def fig3(stats, plt):
    """Risk tier — vertical bars."""
    vals = [stats["risk_distribution"].get(k,0) for k in RKEYS]; total=sum(vals)
    fig, ax = plt.subplots(figsize=(7,5.5))
    bars = ax.bar(RLABELS, vals, color=RCOLS, edgecolor="white", linewidth=1, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                f"{val}\n({val/total*100:.1f}%)", ha="center", fontsize=11, fontweight="bold", color="#333")
    ax.set_ylabel("Number of Claims", fontsize=14, fontweight="bold")
    ax.set_title("Risk Tier Distribution", fontsize=16, fontweight="bold", pad=18)
    ax.set_ylim(0, max(vals)*1.18)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR/"fig3_risk_distribution.png"); plt.close()
    logger.info("  fig3")


def fig4(claims, plt):
    """Platform comparison — side by side % bars."""
    pv = defaultdict(lambda: Counter())
    for c in claims:
        p = c.get("source_platform","?")
        v = c.get("classification",{}).get("llm_grounded",{}).get("veracity","?")
        pv[p][v] += 1
    rt = sum(pv["reddit"].values()); yt = sum(pv["youtube"].values())
    rp = [pv["reddit"].get(vk,0)/rt*100 for vk in VKEYS]
    yp = [pv["youtube"].get(vk,0)/yt*100 for vk in VKEYS]
    x = np.arange(len(VLABELS)); w = 0.35
    fig, ax = plt.subplots(figsize=(10,5.5))
    b1 = ax.bar(x-w/2, rp, w, color=S3[4], edgecolor="white", linewidth=0.5)  # blue pastel
    b2 = ax.bar(x+w/2, yp, w, color=S3[5], edgecolor="white", linewidth=0.5)  # orange pastel
    for bar, val in zip(b1, rp):
        if val>2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}%", ha="center", fontsize=9, color="#333")
    for bar, val in zip(b2, yp):
        if val>2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}%", ha="center", fontsize=9, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(VLABELS, fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=14, fontweight="bold")
    ax.set_title(f"Veracity by Platform — Reddit (blue, n={rt}) vs. YouTube (orange, n={yt})",
                 fontsize=14, fontweight="bold", pad=18)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR/"fig4_platform_comparison.png"); plt.close()
    logger.info("  fig4")


def fig5(claims, plt):
    """Category × Veracity heatmap."""
    cats = ["supplement_efficacy","drug_misuse","disease_cure","weight_loss","detox","mental_health","anti_aging","other"]
    clabs = ["Supplement Efficacy","Drug Misuse","Disease Cure","Weight Loss","Detox","Mental Health","Anti-Aging","Other"]
    matrix = np.zeros((len(cats), len(VKEYS)))
    for c in claims:
        ca = c.get("risk_category","other"); v = c.get("classification",{}).get("llm_grounded",{}).get("veracity","?")
        if ca in cats and v in VKEYS: matrix[cats.index(ca)][VKEYS.index(v)] += 1
    rs = matrix.sum(axis=1, keepdims=True); rs[rs==0]=1; pct = matrix/rs*100
    fig, ax = plt.subplots(figsize=(10,6))
    im = ax.imshow(pct, cmap="YlOrBr", aspect="auto", vmin=0, vmax=75)
    ax.set_xticks(range(len(VLABELS))); ax.set_xticklabels(VLABELS, fontsize=11)
    ax.set_yticks(range(len(clabs))); ax.set_yticklabels(clabs, fontsize=11)
    for i in range(len(cats)):
        for j in range(len(VKEYS)):
            val=int(matrix[i][j]); p=pct[i][j]
            if val>0:
                col = "white" if p>45 else "#333"
                ax.text(j, i, f"{val}\n({p:.0f}%)", ha="center", va="center", fontsize=8, color=col, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02); cb.set_label("Row %", fontsize=11)
    ax.set_title("Claim Category vs. Veracity", fontsize=16, fontweight="bold", pad=18)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR/"fig5_category_heatmap.png"); plt.close()
    logger.info("  fig5")


def fig6(claims, plt):
    """Confidence box plot — Set3 pastel fills."""
    cb = defaultdict(list)
    for c in claims:
        cl = c.get("classification",{}).get("llm_grounded",{})
        v = cl.get("veracity","?"); cf = cl.get("confidence",0)
        if v in VKEYS and cf>0: cb[v].append(cf)
    data = [cb.get(vk,[0]) for vk in VKEYS]
    fig, ax = plt.subplots(figsize=(9,5))
    bp = ax.boxplot(data, labels=VLABELS, patch_artist=True, widths=0.45,
                     medianprops=dict(color="#222", linewidth=2),
                     flierprops=dict(marker="o", markersize=3, markerfacecolor="#999", alpha=0.4))
    for patch, col in zip(bp["boxes"], VCOLS):
        patch.set_facecolor(col); patch.set_alpha(0.8)
    ax.set_ylabel("Confidence Score", fontsize=14, fontweight="bold")
    ax.set_title("Classification Confidence by Veracity", fontsize=16, fontweight="bold", pad=18)
    ax.set_ylim(0, 1.05)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR/"fig6_confidence_boxplot.png"); plt.close()
    logger.info("  fig6")


def fig7(stats, plt):
    """Category distribution — horizontal sorted, pastel Set3."""
    cats = stats["by_risk_category"]
    sc = sorted(cats.items(), key=lambda x:x[1], reverse=True)
    names = [c[0].replace("_"," ").title() for c in sc]; vals = [c[1] for c in sc]
    cols = [S3[i % len(S3)] for i in range(len(vals))]
    fig, ax = plt.subplots(figsize=(9,5))
    bars = ax.barh(names[::-1], vals[::-1], color=cols[::-1], edgecolor="white", linewidth=0.6, height=0.6)
    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width()+30, bar.get_y()+bar.get_height()/2,
                f"{val}", va="center", fontsize=11, fontweight="bold", color="#444")
    ax.set_xlabel("Number of Claims", fontsize=14, fontweight="bold")
    ax.set_title("Health Claims by Category", fontsize=16, fontweight="bold", pad=18)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR/"fig7_category_distribution.png"); plt.close()
    logger.info("  fig7")


def generate_examples(claims):
    for label in VKEYS:
        m = [c for c in claims if c.get("classification",{}).get("llm_grounded",{}).get("veracity")==label]
        m.sort(key=lambda c: c.get("classification",{}).get("llm_grounded",{}).get("confidence",0), reverse=True)
        fp = OUTPUT_DIR/f"examples_{label}.txt"
        with open(fp,"w",encoding="utf-8") as f:
            f.write(f"{'='*80}\nVERIFICATION EXAMPLES: {label} ({len(m)} total)\n{'='*80}\n\n")
            for i, c in enumerate(m[:20],1):
                cl = c.get("classification",{}).get("llm_grounded",{})
                ev = c.get("evidence",{})
                f.write(f"--- Example {i} ---\n")
                f.write(f"Platform:       {c.get('source_platform','?')}\n")
                f.write(f"Product:        {c.get('product','N/A')}\n")
                f.write(f"Claimed Effect: {c.get('claimed_effect','N/A')}\n")
                f.write(f"Target:         {c.get('target_condition','N/A')}\n")
                f.write(f"Strength:       {c.get('claim_strength','N/A')}\n")
                f.write(f"Category:       {c.get('risk_category','N/A')}\n")
                f.write(f"Verbatim:       \"{c.get('verbatim_quote','')[:200]}\"\n")
                f.write(f"Confidence:     {cl.get('confidence',0)}\n")
                f.write(f"Risk Tier:      {cl.get('risk_tier','N/A')}\n")
                f.write(f"Reasoning:      {cl.get('reasoning','')[:300]}\n")
                f.write(f"Key Evidence:   {cl.get('key_evidence','')[:300]}\n")
                f.write(f"Recommendation: {cl.get('recommendation','')[:200]}\n")
                f.write(f"PubMed hits:    {len(ev.get('pubmed_articles',[]))}\n")
                f.write(f"FDA events:     {len(ev.get('fda_events',[]))}\n")
                f.write(f"NIH ref:        {'Yes' if ev.get('nih_reference') else 'No'}\n")
                f.write(f"Source URL:     {c.get('source_url','N/A')}\n\n")
        logger.info(f"  examples_{label}.txt — {min(20,len(m))} examples")


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Publication Figures (Set3 Pastel) & Examples")
    logger.info("="*60)
    claims = load_claims(INPUT_PATH)
    logger.info(f"Loaded {len(claims)} claims")
    stats = compute_stats(claims)
    plt = setup()
    fig1(stats, plt); fig2(stats, plt); fig3(stats, plt)
    fig4(claims, plt); fig5(claims, plt); fig6(claims, plt)
    fig7(stats, plt); generate_examples(claims)
    logger.info("\nDone.")
