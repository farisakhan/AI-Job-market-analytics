"""Phase 6 + 8 - Attention-style skill ranking and Explainable Hybrid Career Recommendation.

role_score = a*cosine(user, role) + b*future_alignment(role) + c*graph_proximity(user -> role)
 - cosine           : content-based match between user skill vector and role skill vector (KG REQUIRES weights)
 - future_alignment : share of the role's demand-weighted skill profile that is forecast to GROW
 - graph_proximity  : Personalised PageRank from the user's skills on the skill co-occurrence graph
Skill gaps are ranked with a softmax attention over [role importance, forecast growth, graph proximity].
Every recommendation carries a human-readable explanation with the numeric contribution of each term.
"""
import numpy as np, pandas as pd, networkx as nx
from skills_catalog import COURSES, SKILLS

def _softmax(x, tau=1.0):
    z = (np.array(x) - np.max(x)) / tau; e = np.exp(z); return e / e.sum()

class CareerRecommender:
    def __init__(self, G, H, forecast_df, weights=(0.45, 0.30, 0.25), demand_only=False):
        """demand_only=True: no forecast is available (too little history). The 'growth' signal is then the
        skill's CURRENT demand percentile (0-100) instead of forecast growth, and texts say so."""
        self.demand_only = demand_only
        self.G, self.H, self.fc = G, H, forecast_df.set_index("skill")
        self.w = weights
        self.roles = [n[1] for n in G if n[0] == "Role" and n[1] != "Other"]
        self.skills = sorted(self.fc.index)
        self.sidx = {s: i for i, s in enumerate(self.skills)}
        self.R = pd.DataFrame(0.0, index=self.roles, columns=self.skills)
        for r in self.roles:
            for _, s, d in G.edges(("Role", r), data=True):
                if d["rel"] == "REQUIRES" and s[1] in self.R.columns: self.R.loc[r, s[1]] = d["weight"]
        if demand_only:
            self.growth = (self.fc["current_level"].rank(pct=True) * 100).reindex(self.skills).fillna(0)
            self.grow_thr, self.align_label = 50, "demand-alignment"
        else:
            self.growth = self.fc["growth_pct"].clip(-50, 150).reindex(self.skills).fillna(0)
            self.grow_thr, self.align_label = 10, "future-alignment"

    def _ppr(self, user_skills):
        pers = {s: 1.0 for s in user_skills if s in self.H}
        if not pers: return pd.Series(0.0, index=self.skills)
        pr = nx.pagerank(self.H, alpha=0.7, personalization=pers, weight="weight")
        return pd.Series({s: pr.get(s, 0) for s in self.skills})

    def recommend(self, user_skills, top_roles=3, top_skills=6, target_role=None):
        user_skills = [s for s in user_skills if s in self.sidx]
        u = pd.Series(0.0, index=self.skills); u[user_skills] = 1.0
        ppr = self._ppr(user_skills); ppr_n = ppr / (ppr.max() + 1e-9)
        a, b, c = self.w
        rows = []
        for r in self.roles:
            v = self.R.loc[r]
            cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
            grow = float((v * (self.growth > self.grow_thr)).sum() / (v.sum() + 1e-9))               # share of role weight in growing skills
            prox = float((v * ppr_n).sum() / (v.sum() + 1e-9))
            rows.append(dict(role=r, score=a * cos + b * grow + c * prox, cosine=cos, future_alignment=grow, graph_proximity=prox))
        rk = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
        role = target_role or rk.role.iloc[0]
        v = self.R.loc[role]
        gaps = [s for s in self.skills if v[s] >= 0.15 and u[s] == 0]
        feat = np.array([[v[s], self.growth[s] / 100, ppr_n[s]] for s in gaps]) if gaps else np.zeros((0, 3))
        att = _softmax(feat @ np.array([2.0, 1.5, 1.0]), 0.5) if gaps else []
        gap_df = pd.DataFrame({"skill": gaps, "role_importance": feat[:, 0] if gaps else [], "forecast_growth_pct": [self.growth[s] for s in gaps],
                               "graph_proximity": feat[:, 2] if gaps else [], "attention": att}).sort_values("attention", ascending=False).head(top_skills)
        recs = []
        for _, g in gap_df.iterrows():
            course, cert = COURSES.get(g.skill, (f"Search: '{g.skill} fundamentals' course", None))
            top_demand_pct = max(1, 100 - g.forecast_growth_pct)
            relation = "closely related" if g.graph_proximity >= 0.2 else "weakly related"
            demand_txt = (f"currently in the top {top_demand_pct:.0f}% most-demanded skills (no forecast: history too short)"
                          if self.demand_only else f"forecast demand {g.forecast_growth_pct:+.0f}% in the next 3 months")
            why = (f"Required by {role} in {g.role_importance*100:.0f}% of postings; {demand_txt}; "
                   f"{relation} to your skills (proximity {g.graph_proximity:.2f}).")
            recs.append(dict(skill=g.skill, course=course, certification=cert, attention=round(float(g.attention), 3), why=why))
        comps = sorted([(x[1], d["weight"]) for _, x, d in self.G.edges(("Role", role), data=True)
                        if d["rel"] == "HIRES_FOR"], key=lambda t: -t[1])[:5]
        expl = []
        for _, r in rk.head(top_roles).iterrows():
            expl.append(f"{r.role}: score {r.score:.3f} = {a}x skill-match {r.cosine:.2f} + {b}x {self.align_label} {r.future_alignment:.2f} + {c}x graph-proximity {r.graph_proximity:.2f}")
        return dict(user_skills=user_skills, role_ranking=rk.head(top_roles), target_role=role, skill_gaps=recs,
                    companies=[c for c, _ in comps], explanations=expl)
