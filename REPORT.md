# AI-Driven Job Market Analytics and Explainable Skill Trend Prediction using Transformer-Based Temporal Learning and Knowledge Graphs

## 1. Abstract

This project develops an end-to-end framework for job-market analytics, skill extraction, knowledge-graph construction, temporal skill-demand prediction and explainable career recommendations. The system processes job postings, extracts skills and roles, builds relationships between skills, companies and roles, and exports the resulting graph for Neo4j.

The real Indian LinkedIn dataset is used for data cleaning, skill extraction, knowledge-graph construction and current-demand-based career recommendations. Because the real dataset contains only 12 months of observations, forecasting and model comparison are evaluated separately using a 70-month synthetic dataset. Synthetic evaluation validates pipeline functionality; it is not evidence of actual Indian job-market trends.

## 2. Research gap

Existing job-search systems primarily describe current vacancies. They often do not explain relationships between skills, roles and companies, predict future skill demand, or provide transparent recommendations based on a user's existing skills.

This project combines job-posting analytics, skill relationships, temporal modelling and explainable career recommendations in one pipeline.

## 3. Project contributions

1. A unified schema for job postings from different sources, preserving source and country metadata.
2. Job-title normalization and skill/certification extraction.
3. A skill knowledge graph connecting skills, roles, companies and categories.
4. Neo4j-compatible graph exports.
5. Temporal skill-demand forecasting and model benchmarking.
6. Global and local explainability outputs.
7. Career recommendations that can operate using current skill demand when forecast history is insufficient.

## 4. System architecture

The pipeline consists of the following phases:

1. Job-data loading and source normalization.
2. Data cleaning and job-role normalization.
3. Skill and certification extraction.
4. Skill knowledge-graph construction.
5. Temporal skill-demand forecasting and model comparison.
6. Skill ranking and prediction.
7. Explainability.
8. Career and skill-gap recommendations.

The real-data pipeline can complete graph construction, descriptive visualizations and recommendations even when forecasting is skipped.

## 5. Methodology

### 5.1 Data loading and schema

The real dataset is a LinkedIn job-postings dataset focused on India. The unified schema includes:

`job_id, source, country, title, company, location, description, date_posted, salary`

The loader preserves the `source` field and adds `country = India` when it is absent from the supplied India dataset. The original raw CSV is not modified.

### 5.2 Data cleaning and preprocessing

The cleaning process includes:

- HTML entity decoding and removal of HTML tags.
- URL and whitespace cleanup.
- Missing-value handling.
- Conversion of posting dates into date values.
- Removal of records without usable posting dates.
- Removal of very short job descriptions.
- Near-duplicate removal based on title, company, description and posting month.
- Job-title normalization into standard role categories.
- Salary conversion and removal of implausible salary values.

### 5.3 Skill and certification extraction

The main skill extraction method uses a catalogue of skills and aliases with regular-expression matching. An optional spaCy EntityRuler can supplement the dictionary-based matcher when the required spaCy model is available.

Certification extraction uses explicit pattern matching to avoid accidental substring matches. No certification matches were observed in the current real dataset using the strict matching rules.

The skill catalogue contains 55 skills. The retained real-data analysis contains 16 skills after applying the minimum-posting-frequency threshold.

The current pipeline uses TF-IDF with Truncated SVD as a lightweight skill-embedding fallback. The resulting embedding matrix contains 55 skill vectors with 32 dimensions. This run does not use BERT or Sentence Transformers embeddings.

Monthly skill demand is calculated as the number of postings mentioning a skill per 1,000 job postings.

### 5.4 Knowledge graph construction

The knowledge graph represents job-market relationships using typed nodes and edges.

Node types include:

- Skill
- Role
- Company
- Category
- Certification, when an observed certification is available

Relationship types include:

- `REQUIRES`: connects a role to a required skill.
- `HIRES_FOR`: connects a company to a role.
- `CO_OCCURS`: connects skills that appear together.
- `IN_CATEGORY`: connects a skill to its category.
- `VALIDATES`: connects an observed certification to a skill.

Graph analysis uses measures such as PageRank, weighted degree, clustering and community information.

### 5.5 Neo4j export

The graph is exported into three files:

- `outputs/neo4j/nodes.csv`
- `outputs/neo4j/edges.csv`
- `outputs/neo4j/load_graph.cypher`

The CSV files contain the graph's node and relationship records. The Cypher file provides the statements needed to load the exported graph into Neo4j.

### 5.6 Temporal forecasting and model comparison

Forecasting predicts skill demand three months ahead using a 12-month lookback window. The last six months are reserved for testing. The pipeline requires at least 21 months of historical observations for the configured lookback, forecast horizon and test window.

The evaluated models include:

- Naive baseline
- Seasonal Naive baseline
- Random Forest
- XGBoost
- LSTM
- Prophet
- HGTSP (proposed PyTorch model)
- Ablation model without graph features

Evaluation metrics are Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE).

### 5.7 Explainability

Global feature importance is calculated using permutation importance and grouped by feature family. Local explanations use a weighted linear surrogate in the LIME style, with the LIME library used when available.

For the synthetic run, HGTSP PyTorch completed successfully. The explainability pipeline used a separate HistGradientBoosting surrogate trained on the graph-augmented training features because the current explainer expects a compatible scikit-learn estimator.

Therefore, the reported global and local explanations describe the surrogate model, not the HGTSP PyTorch model directly. SHAP calculations were also executed on up to 300 test samples.

### 5.8 Career recommendation

The career recommender combines:

- User-to-role skill matching.
- Demand alignment.
- Skill-graph proximity using Personalised PageRank.

When forecasting is unavailable, the recommender uses current skill-demand percentiles instead of future forecast growth. The output includes ranked roles, skill gaps, suggested courses/certifications, companies and textual explanations.

The real-data run uses this current-demand mode. Its recommendations must not be described as forecast-based recommendations.

## 6. Data and dataset selection

### 6.1 Main real dataset

The primary dataset is the LinkedIn job-postings dataset focused on India.

| Attribute | Result |
|---|---:|
| Original postings loaded | 949 |
| Postings after cleaning | 781 |
| Historical coverage | March 2023 – February 2024 |
| Months represented | 12 |
| Source | LinkedIn |
| Country | India |
| Retained skills | 16 |

The 781 cleaned records are used for preprocessing, skill extraction, knowledge-graph construction and current-demand-based career recommendations.

The `source` and `country` fields are preserved in the cleaned real-data output.

### 6.2 Additional datasets investigated

Additional public datasets were examined to determine whether they could extend the Indian historical series.

| Dataset | Observed coverage / issue | Decision |
|---|---|---|
| Naukri historical datasets | Older records from 2015–2017 and 2019–2020; these periods are separated from the 2023–2024 LinkedIn series by a large gap | Not merged into the forecasting series |
| LinkedIn Job Postings 2023–2024, Kaggle (`arshkon`) | The supplied archive covered approximately December 2023–April 2024 and was overwhelmingly US-focused | Not used for Indian forecasting |
| Indian Tech Jobs 2026 archive | Posting dates were largely relative and descriptions were truncated | Not used for temporal forecasting |

No additional free dataset was verified to contain full descriptions, reliable original posting dates, India-focused records and continuous monthly coverage that would safely extend the main series.

The Naukri datasets were retained as separately inspected data and were not combined with the 2023–2024 LinkedIn series. Combining disconnected periods and treating the intervening gap as continuous would distort a temporal model's interpretation of demand.

The US-focused LinkedIn dataset was also not mixed into the Indian forecasting series. Country and source differences must be retained and analysed separately.

## 7. Experimental results

### 7.1 Real-data execution

The real-data pipeline completed cleaning, skill extraction, graph construction, Neo4j export, descriptive visualizations and career recommendations.

| Component | Result |
|---|---:|
| Cleaned postings | 781 |
| Historical months | 12 |
| Retained skills | 16 |
| Knowledge-graph nodes | 648 |
| Knowledge-graph edges | 1,285 |
| Average extracted skills per posting | 2.31 |
| Skill embedding matrix | 55 × 32 |

The run summary records:

- `forecast_status`: `SKIPPED`
- `xai_status`: `SKIPPED`
- `recommendations_status`: `OK (current-demand mode, no forecast)`

The recommendations were generated for three demonstration personas:

| Demonstration user | Top recommended role |
|---|---|
| Final-year CSE student (Python, SQL, ML basics) | Data Analyst |
| Web developer (JS, React, jQuery, PHP) | Frontend Developer |
| Data analyst (Excel, SQL, Tableau) | Credit Analyst |

The real-data run generated descriptive skill-trend, knowledge-graph and skill-demand figures. Forecast-dependent figures and explanations are not claimed as real-data outputs.

### 7.2 Synthetic forecasting validation

A separate synthetic dataset was generated for pipeline validation.

| Attribute | Result |
|---|---:|
| Synthetic postings | 29,440 |
| Historical coverage | 70 months |
| Synthetic date range | January 2021 – October 2026 |
| Skills | 55 |
| Sources | LinkedIn, Glassdoor, Indeed and Naukri |
| Knowledge-graph nodes | 105 |
| Knowledge-graph edges | 2,505 |

The synthetic held-out evaluation results are:

| Model | MAE | RMSE | MAPE (%) |
|---|---:|---:|---:|
| Naive (last value) | 17.2217 | 23.0276 | 28.9679 |
| Seasonal Naive | 26.5599 | 46.8532 | 31.6242 |
| Random Forest | 15.2176 | 22.5198 | 22.3810 |
| XGBoost | 15.8674 | 22.6801 | 24.4598 |
| LSTM | 15.4606 | 22.6808 | 22.3268 |
| Prophet | 15.9076 | 21.9606 | 23.2298 |
| HGTSP (proposed, torch) | 16.8896 | 28.7079 | 21.1197 |
| Ablation: no graph | 15.0300 | 21.4611 | 24.9380 |

The ablation model without graph features achieved the lowest MAE and RMSE. HGTSP achieved the lowest MAPE. These results do not show that HGTSP is the best model on every metric.

The synthetic evaluation validates that the forecasting and model-comparison pipeline executes on a sufficiently long dataset. It does not demonstrate predictive performance on real Indian job-market data.

### 7.3 Synthetic explainability results

Because the HGTSP PyTorch model was used for forecasting, a separate HistGradientBoosting surrogate was trained for model explanation.

Permutation importance by feature family was:

| Feature family | Importance |
|---|---:|
| Own recent demand (lags) | 0.0079 |
| Seasonality | 0.0002 |
| KG structure (PageRank, degree, clustering, community) | -0.0030 |
| Momentum / volatility | -0.0035 |
| Graph neighbours' demand | -0.0157 |

Negative permutation-importance values indicate that permuting a feature family did not reduce the surrogate's validation performance in this run. They should not be interpreted as causal negative effects.

The synthetic run generated global importance, importance by family and local explanation files. These explanations belong to the HistGradientBoosting surrogate, not directly to HGTSP PyTorch.

### 7.4 Generated outputs

Real-data outputs are stored separately under `outputs/real_data/`:

- `outputs/real_data/run_summary.json`
- `outputs/real_data/recommendations.json`
- `outputs/real_data/figures/`
- `outputs/real_data/neo4j/nodes.csv`
- `outputs/real_data/neo4j/edges.csv`
- `outputs/real_data/neo4j/load_graph.cypher`
- `outputs/real_data/powerbi/`

Synthetic validation outputs are stored under `outputs/`:

- `outputs/run_summary.json`
- `outputs/model_comparison.csv`
- `outputs/skill_forecast.csv`
- `outputs/global_importance.csv`
- `outputs/importance_by_family.csv`
- `outputs/local_explanations.json`
- `outputs/recommendations.json`
- `outputs/figures/` (six figures)
- `outputs/neo4j/` (nodes, edges and Cypher export)

Record-level job-posting exports are not included in the GitHub submission. Real and synthetic results must be interpreted separately.

## 8. Limitations

1. **Short real history:** The real Indian dataset contains 12 months, while the configured forecasting pipeline requires at least 21 months. Real-data forecasting and forecast-dependent explainability were therefore skipped.
2. **Synthetic validation:** Forecasting, model comparison and surrogate explainability were validated on a synthetic 70-month dataset. These results demonstrate pipeline execution, not real-market predictive accuracy.
3. **Small real sample:** The cleaned real dataset contains 781 postings and 16 retained skills, limiting the breadth of analysis.
4. **Posting-date concentration:** Real postings are concentrated on a relatively small number of dates, which can make monthly demand estimates noisy.
5. **Extraction method:** The primary skill extractor uses dictionary/alias matching and regular expressions; implicit or context-dependent skills may be missed.
6. **Embedding fallback:** The current embedding run uses TF-IDF + SVD rather than BERT or Sentence Transformers.
7. **Source and geography:** The real forecasting series is India-focused. Other country datasets were not mixed into it.
8. **Demand proxy:** Job postings mentioning a skill are a proxy for observed demand, not a complete measure of labour-market demand.
9. **Certification evidence:** No strict certification matches were found in the real dataset, so certification relationships should not be inferred from this run.

## 9. Conclusion and future work

The real-data pipeline successfully performs job-data cleaning, skill extraction, knowledge-graph construction, Neo4j export, descriptive analysis and current-demand-based career recommendations.

The separate synthetic validation demonstrates that the temporal forecasting, model comparison and explainability stages can execute when sufficient historical observations are available. Synthetic results must remain clearly distinguished from findings based on the real Indian dataset.

Future work includes collecting a longer, legally reusable and India-focused dataset with full descriptions and reliable posting dates; evaluating forecasting on real continuous history; comparing extraction methods under a common evaluation protocol; and conducting user studies to assess recommendation usefulness.

## 10. References

- Vaswani et al. (2017), *Attention Is All You Need*.
- Lim et al. (2021), *Temporal Fusion Transformers*.
- Veličković et al. (2018), *Graph Attention Networks*.
- Lundberg and Lee (2017), *A Unified Approach to Interpreting Model Predictions*.
- Ribeiro et al. (2016), *Why Should I Trust You? Explaining the Predictions of Any Classifier*.
- Chen and Guestrin (2016), *XGBoost: A Scalable Tree Boosting System*.
- Reimers and Gurevych (2019), *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*.
- Taylor and Letham (2018), *Forecasting at Scale*.
