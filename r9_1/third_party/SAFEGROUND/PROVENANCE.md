# Vendored SafeGround uncertainty code (unmodified)

- Source: https://github.com/UCSB-AI/SAFEGROUND, branch `master`,
  commit `5e8fca7ef091bc6751cad9703ca430e775aa4433` (2026-06-19).
- Files: `code/{heatmap,regions,margin,entropy,concentration,combined,uncertainty}.py`,
  fetched from `https://raw.githubusercontent.com/UCSB-AI/SAFEGROUND/<commit>/code/<file>`
  on 2026-10-16; SHA-256 in `SHA256SUMS`. License: MIT (`LICENSE`).
- Only the empty `__init__.py` files were added so that `combined.py`'s
  `from SAFEGROUND.code.margin import ...` resolves; no line of the vendored files was edited.
- Paper: arXiv:2602.02419v2. Known paper/code differences: patch size 14 (paper) vs 28
  (code default); activation threshold 0.3 (paper) vs 0.0 (code default); single-region
  edge values (IE 0.5, CD 0.1) and empty-region value (1.0) exist only in code.
