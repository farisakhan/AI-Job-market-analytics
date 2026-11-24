# AI-Driven Job Market Analytics and Explainable Skill Trend Prediction using Transformer-Based Temporal Learning and Knowledge Graphs


> **Status note.** Sections 6–7 contain results from a *synthetic* dataset used to validate the pipeline. Replace them with results from your real data (`python src/run_pipeline.py --data ...`) before submission.

## 1. Abstract
Job portals show current vacancies but not which skills will matter next, how skills relate, or why a recommendation was made. We present an end-to-end framework that (i) unifies multi-source job postings, (ii) extracts skills, roles and certifications into a dynamic skill knowledge graph, (iii) forecasts skill demand with a Hybrid Graph Transformer Skill Prediction model (HGTSP), (iv) explains predictions with SHAP/LIME-style attributions and attention, and (v) produces personalised, explainable course, certification, career-path and company recommendations.

## 2. Research gap
Existing systems (a) list only current vacancies, (b) do not predict future skills, (c) ignore relations among skills, companies and roles, (d) lack explainability, and (e) offer no personalised guidance.

## 3. Contributions
1. Distributed job-data collection framework and unified dataset.
2. Dynamic skill knowledge graph with graph-based skill mining.
3. HGTSP: Transformer temporal encoder + graph attention over the skill graph.
4. Explainability layer (global/local attributions, temporal & graph attention).
5. Hybrid explainable career recommender.

## 4. System architecture (8 phases)
Collection → Cleaning → BERT/SBERT embedding & extraction → Knowledge-graph generation → Transformer temporal learning → Attention-based skill ranking → Explainable AI → Career recommendation. (Figure: `outputs/figures`; module/code map in README.)

## 5. Methodology
**5.1 Data collection.** BeautifulSoup for static pages, Selenium for JS-rendered pages, Scrapy for crawling, Glassdoor partner API; per-source records mapped to one schema (`job_id, source, title, company, location, description, date_posted, salary`). Scraping respects robots.txt, delays and each site's Terms of Service; public datasets are the preferred source.
**5.2 Cleaning.** HTML/URL stripping, date parsing, cross-source near-duplicate removal, title normalisation, salary outlier handling.
**5.3 Extraction.** Catalogue matcher with aliases (+ spaCy EntityRuler) and Sentence-Transformer embeddings (TF-IDF+SVD fallback) for skill vectors. Monthly demand = postings mentioning the skill per 1000 postings.
**5.4 Knowledge graph.** Nodes: Skill, Role, Company, Certification, Category. Edges: REQUIRES(role→skill, share of postings), HIRES_FOR(company→role), CO_OCCURS(skill–skill, PMI), VALIDATES(cert→skill). Graph features: PageRank, weighted degree, clustering, Louvain community. Exported to Neo4j.
**5.5 Forecasting.** Direct 3-month-ahead forecast from a 12-month window, normalised per skill; time-based split (last 6 months held out). Baselines: Naive, Seasonal-Naive, Random Forest, XGBoost, LSTM, Prophet, TFT. Metrics: MAE, RMSE, MAPE (denominator floored at 1 post/1000).
**5.6 HGTSP.** Per-skill Transformer encoder over tokens [own demand, graph-neighbour demand, month sin/cos] with attention pooling → gated fusion with skill embedding and graph features → graph-attention layer over the KG (Veličković et al., 2018 style) → regression head. Temporal attention α and graph attention β are retained for explanation. Implementation: `src/hgtsp_torch.py`.
**5.7 Explainability.** Permutation importance and SHAP (global), LIME (local; own implementation of the algorithm when the library is absent), grouped by feature family; HGTSP attention weights.
**5.8 Recommendation.** `score(role) = 0.45·cosine(user, role) + 0.30·future-alignment + 0.25·graph-proximity (Personalised PageRank)`. Skill gaps ranked with softmax attention over [role importance, forecast growth, graph proximity]; each item maps to a course/certification and carries a textual justification.

## 6. Experimental results (synthetic data: 29,440 postings, 70 months, 55 skills, 4 sources)
Held-out last 6 months, 3-month-ahead, units = postings per 1000.

| Model | MAE | RMSE | MAPE % |
|---|---|---|---|
| Naive (last value) | 17.15 | 23.00 | 28.86 |
| Seasonal Naive | 26.48 | 46.83 | 31.53 |
| Random Forest | 15.23 | 22.31 | 22.51 |
| XGBoost (HistGB fallback) | **14.78** | **20.93** | 23.67 |
| HGTSP-lite (graph-augmented GBM)* | 17.16 | 26.97 | 25.03 |
| Ablation: same model, no graph features | 14.98 | 21.65 | 23.51 |

\*The sandbox had no PyTorch, so the transformer/GAT model could not be run; "HGTSP-lite" is a gradient-boosting model fed with graph features and is **not** the proposed architecture.

**Honest reading.** (1) Tree models beat naive baselines by ~14% MAE. (2) Graph features did **not** help here and slightly hurt — expected, because the synthetic generator evolves each skill's demand independently, so the graph carries no extra predictive signal. This experiment therefore neither supports nor refutes the value of the graph; it must be tested on real data, where related skills (e.g. LLM/LangChain/Generative AI) move together. (3) Single-month demand for low-volume skills is very noisy, so the system flags such skills `reliable=False` and reports a smoothed 24-month trend (`trend_pct_per_year`) next to the model forecast. (4) The pipeline recovers the trends built into the generator (Prompt Engineering +77 %/yr, LangChain +74 %, Generative AI +56 %, LLM +42 % up; Blockchain −27 %, Hadoop −15 % down) — a sanity check of extraction and aggregation, not evidence about the real market.

## 7. Explainability & recommendation (examples)
Global importance by family (permutation, ΔMAE): own recent demand is the only family with positive importance; neighbour and graph features are ≤ 0 on synthetic data. Local explanations in `outputs/local_explanations.json`. Example recommendation (student with Python, SQL, ML, Pandas, Git): top role Data Scientist (0.421 = skill-match 0.615, future-alignment 0.012, graph-proximity 0.562); top gaps Scikit-learn, Deep Learning, Generative AI, NLP, each with course, certification and justification (`outputs/recommendations.json`).

## 8. Limitations
Synthetic evaluation only; keyword extraction misses implicit skills; scraped postings are biased toward online/tech roles; monthly postings ≠ true labour demand; graph and forecast built on one country/period; recommendations do not model personal constraints (time, cost).

## 9. Future work
Run HGTSP, LSTM, Prophet and TFT on real data with multiple seeds and significance tests; ablations (no GAT / no temporal attention / no neighbour tokens); longer horizons; salary forecasting; user study of explanation usefulness; live Neo4j + Power BI deployment.

## 10. References (verify details before citing)
Vaswani et al. 2017, *Attention Is All You Need*; Lim et al. 2021, *Temporal Fusion Transformers* (Int. J. Forecasting); Veličković et al. 2018, *Graph Attention Networks*; Lundberg & Lee 2017, *SHAP*; Ribeiro et al. 2016, *LIME*; Chen & Guestrin 2016, *XGBoost*; Devlin et al. 2019, *BERT*; Reimers & Gurevych 2019, *Sentence-BERT*; Taylor & Letham 2018, *Prophet*.
