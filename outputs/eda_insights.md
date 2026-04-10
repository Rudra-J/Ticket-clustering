# EDA Insights - Tobi-Bueck/customer-support-tickets
**Generated:** 2026-04-10
**Dataset:** 61,765 rows x 19 columns (via HuggingFace Hub)

## Critical Insights (must act on)

- **Unlabeled tickets:** 21.3% of tickets have no `type` label.
  `pd.get_dummies()` on `type` will silently drop nulls, producing a skewed
  one-hot block. Downstream `encode_metadata` must fill nulls before encoding
  (e.g. `df['type'].fillna('Unknown')`) to avoid losing 13k rows of signal.

- **Subject nulls:** 8.6% of subjects are null. The pipeline
  already uses `fillna('')` before concatenation, so these become body-only
  tickets. No code change needed, but cluster quality for these rows is lower.

- **Queue cardinality:** 52 unique queues → one-hot would add 52
  sparse columns. Frequency encoding is correctly set in config.yaml.

- **Bilingual corpus:** 45.8% English, 54.2% German.
  all-mpnet-base-v2 is multilingual and handles both. Domain keyword allowlist
  in config.yaml is English-only — consider adding German equivalents for key
  ITSM terms (e.g. "Sicherheit" for security, "Netzwerk" for network).

## Important Insights (should consider)

- **Priority distribution:** BALANCED — max share 37.8%.
  No skew — priority contributes clean ordinal signal.

- **Body length:** median 57 words, max 281 words.
  All tickets fit within 512 tokens — no truncation concern.

- **Short texts:** 0.1% of tickets have <5 combined words (typically
  null-subject tickets where body is also short). These produce low-quality
  embeddings and may cluster as noise under HDBSCAN.

- **Tags not used as features:** tag_1 through tag_8 are sparse (tag_6/7/8
  <10% fill) and not wired into the pipeline. They are valuable for post-hoc
  cluster labelling in `evaluation.py` business_interpret().

## Minor Notes (awareness only)

- `version` column: 53.7% null — not in pipeline, no action.
- `answer` column: not used in clustering — it's the agent's reply, not the issue.
- Duplicate rows: 0 — none found.
- `body` nulls: only 2 rows — negligible.

## Config Recommendations

- `clustering.kmeans.k`: start range 4–12
  based on 4 known ticket types (and 52 queues suggesting finer clusters).
- `clustering.hdbscan.min_cluster_size`: 308
  (~0.5% of 61,765 rows — lower than 1% to allow tighter clusters).
- `fusion.alpha`: if priority proves skewed after full run, lower from 0.7 → 0.6
  to reduce operational feature weight relative to semantic embeddings.
