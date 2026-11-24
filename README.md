# AI-Driven Job Market Analytics & Explainable Skill Trend Prediction (HGTSP)
Transformer-based temporal learning + Skill Knowledge Graph + Explainable career recommendation.

## Run
```bash
pip install -r requirements.txt            # core libs are enough for the demo; torch/shap/lime/spacy/prophet are optional
python src/run_pipeline.py                  # synthetic demo data (pipeline test)
python src/run_pipeline.py --data data/raw/linkedin_clean.csv \
   --colmap '{"job_title":"title","company_name":"company","job_description":"description","listed_time":"date_posted"}'
```
Required columns after mapping: `title, company, location, description, date_posted` (salary optional).
`date_posted` must be a real date (convert epoch/“3 days ago” first). Need >= ~24 months of history for forecasting.

## Layout
| Phase | File |
|---|---|
| 1 Collection | `src/collect.py` (BeautifulSoup, Selenium, Scrapy template, Glassdoor API, unify) |
| 2 Cleaning | `src/clean.py` |
| 3 Extraction | `src/extract.py` (regex+aliases, spaCy EntityRuler, SBERT) |
| 4 Knowledge graph | `src/kg.py` (NetworkX, Neo4j CSV/Cypher/driver) |
| 5-6 Forecasting | `src/forecast.py`, `lstm_model.py`, `prophet_model.py`, `tft_model.py`, `hgtsp_torch.py` |
| 7-8 XAI + recommender | `src/explain.py`, `src/recommend.py`, `outputs/powerbi/*` |

## Outputs (`outputs/`)
`model_comparison.csv`, `skill_forecast.csv`, `global_importance.csv`, `local_explanations.json`, `recommendations.json`,
`figures/*.png`, `neo4j/{nodes,edges}.csv + load_graph.cypher`, `powerbi/*.csv` (import all into Power BI Desktop; relate on `skill`).

