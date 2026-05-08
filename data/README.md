# Data directory (local only)

This folder is **not** part of the public Git repository. Generated JSONL, figures, and exports can include **third-party content** (e.g. Reddit, YouTube) and may be restricted by **platform terms of use**, **copyright**, or **privacy** rules in your jurisdiction.

Run the pipeline scripts from the repository root to create the expected layout:

- `00_raw/` — collected posts  
- `01_claims/` — extracted claims  
- `02_evidence/` — claims with retrieved evidence  
- `03_classified/` — classified claims  
- `04_evaluation/results/` — stats, figures, example text files  

Do not commit raw or derived social text here unless you have the right to redistribute it.
