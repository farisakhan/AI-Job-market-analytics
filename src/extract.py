"""Phase 3 - Skill / role / certification extraction + embeddings.

Strategy (hybrid):
 1. Dictionary/regex matcher with aliases (fast, precise, always available).
 2. spaCy EntityRuler / NER (optional) to catch skills phrased in context.
 3. Sentence-Transformers / BERT embeddings (optional) to (a) normalise unseen skill phrases to the
    nearest catalogue skill and (b) provide semantic skill vectors for the graph + HGTSP.
Everything degrades gracefully if the optional libraries are missing.
"""
import re
import numpy as np, pandas as pd
from skills_catalog import SKILLS, CERTIFICATIONS, ROLES

def _build_patterns():
    pats = {}
    for name, (_, aliases) in SKILLS.items():
        for term in [name] + aliases:
            esc = re.escape(term.lower())
            # short/ambiguous tokens need strict boundaries; 'R','Go' only match case-sensitively below
            pats[term] = (name, re.compile(rf"(?<![\w+#.]){esc}(?![\w+#])", re.I))
    return pats
_PATS = _build_patterns()
_CASE_SENSITIVE = {"R": re.compile(r"(?<![\w])R(?![\w+#])"), "Go": re.compile(r"(?<![\w])Go(?![\w+#])")}

def extract_skills_dict(text):
    found = set()
    for term, (name, rx) in _PATS.items():
        if name in _CASE_SENSITIVE:
            if _CASE_SENSITIVE[name].search(text): found.add(name)
        elif rx.search(text): found.add(name)
    return sorted(found)

def extract_certs(text):
    low = text.lower()
    return [c for c in CERTIFICATIONS if c.lower().split("(")[0].strip() in low]

def get_spacy_extractor():
    """Optional: spaCy EntityRuler with the skill catalogue (pip install spacy; python -m spacy download en_core_web_sm)."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns([{"label": "SKILL", "pattern": t, "id": n} for t, (n, _) in _PATS.items()])
        def f(text):
            return sorted({e.ent_id_ or e.text for e in nlp(text).ents if e.label_ == "SKILL"})
        return f
    except Exception as e:
        print("[extract] spaCy unavailable -> dictionary matcher only:", type(e).__name__)
        return None

def get_embedder(name="sentence-transformers/all-MiniLM-L6-v2"):
    """Sentence-Transformers (BERT-family) embeddings. Falls back to a TF-IDF+SVD embedding."""
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(name)
        return lambda texts: m.encode(list(texts), normalize_embeddings=True, show_progress_bar=False), "sbert"
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        state = {}
        def f(texts):
            texts = list(texts)
            if "v" not in state:
                state["v"] = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(texts + [s for s in SKILLS])
                X = state["v"].transform(texts)
                state["s"] = TruncatedSVD(min(32, max(2, X.shape[1] - 1)), random_state=0).fit(X)
            Z = state["s"].transform(state["v"].transform(texts))
            return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        return f, "tfidf-svd"

def extract_all(df):
    spacy_fn = get_spacy_extractor()
    skills = []
    for t in df["description"]:
        s = set(extract_skills_dict(t))
        if spacy_fn: s |= set(spacy_fn(t))
        skills.append(sorted(s))
    df = df.copy()
    df["skills"] = skills
    df["certs"] = df["description"].map(extract_certs)
    print(f"[extract] avg skills/posting = {df['skills'].map(len).mean():.2f}")
    return df

def skill_embeddings(df=None):
    """Embedding for each catalogue skill from its name + category + sample context."""
    emb, kind = get_embedder()
    names = list(SKILLS)
    texts = [f"{n} {SKILLS[n][0]} skill {' '.join(SKILLS[n][1])}" for n in names]
    E = emb(texts)
    print(f"[extract] skill embeddings: {kind}, shape {E.shape}")
    return dict(zip(names, E))

def monthly_skill_counts(df):
    """Long -> wide monthly demand matrix (months x skills), also normalised share per 1000 postings."""
    ex = df[["month", "skills"]].explode("skills").dropna()
    cnt = ex.groupby(["month", "skills"]).size().unstack(fill_value=0)
    tot = df.groupby("month").size().reindex(cnt.index)
    full = pd.date_range(cnt.index.min(), cnt.index.max(), freq="MS")
    cnt = cnt.reindex(full, fill_value=0); tot = tot.reindex(full).ffill()
    share = cnt.div(tot, axis=0) * 1000
    return cnt, share, tot
