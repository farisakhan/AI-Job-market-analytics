"""Phase 4 - Dynamic Skill Knowledge Graph (NetworkX in-memory + Neo4j export/loader).

Nodes : Skill, Role, Company, Certification, Category
Edges : (Role)-[REQUIRES {weight}]->(Skill)   (Company)-[HIRES_FOR {weight}]->(Role)
        (Skill)-[CO_OCCURS {pmi,count}]-(Skill)  (Skill)-[IN_CATEGORY]->(Category)
        (Certification)-[VALIDATES]->(Skill)
'Dynamic' = the graph is rebuilt per time-slice (monthly/quarterly) so edge weights evolve.
"""
import itertools, math
from collections import Counter
import numpy as np, pandas as pd, networkx as nx
from skills_catalog import SKILLS, COURSES

def build_graph(df, min_cooc=5):
    G = nx.Graph()
    N = len(df)
    sk_cnt = Counter(); pair_cnt = Counter(); role_sk = Counter(); comp_role = Counter(); role_cnt = Counter()
    for _, r in df[["role", "company", "skills"]].iterrows():
        s = r["skills"]; role_cnt[r["role"]] += 1; comp_role[(r["company"], r["role"])] += 1
        for a in s:
            sk_cnt[a] += 1; role_sk[(r["role"], a)] += 1
        for a, b in itertools.combinations(sorted(s), 2): pair_cnt[(a, b)] += 1
    for s, c in sk_cnt.items():
        G.add_node(("Skill", s), kind="Skill", name=s, freq=c, category=SKILLS[s][0])
        G.add_node(("Category", SKILLS[s][0]), kind="Category", name=SKILLS[s][0])
        G.add_edge(("Skill", s), ("Category", SKILLS[s][0]), rel="IN_CATEGORY", weight=1.0)
    for (a, b), c in pair_cnt.items():
        if c < min_cooc: continue
        pmi = math.log((c / N) / ((sk_cnt[a] / N) * (sk_cnt[b] / N)))
        G.add_edge(("Skill", a), ("Skill", b), rel="CO_OCCURS", count=c, pmi=pmi, weight=max(pmi, 0.0) + 1e-3)
    for (role, s), c in role_sk.items():
        G.add_node(("Role", role), kind="Role", name=role, freq=role_cnt[role])
        G.add_edge(("Role", role), ("Skill", s), rel="REQUIRES", weight=c / role_cnt[role])
    for (comp, role), c in comp_role.items():
        G.add_node(("Company", comp), kind="Company", name=comp)
        if ("Role", role) in G: G.add_edge(("Company", comp), ("Role", role), rel="HIRES_FOR", weight=c)
    # Add only certifications actually found in job descriptions
    observed_certs = set()

    if "certs" in df.columns:
        for cert_list in df["certs"]:
            if isinstance(cert_list, (list, tuple, set)):
                observed_certs.update(cert_list)

    for s, (course, cert) in COURSES.items():
        if cert and cert in observed_certs and ("Skill", s) in G:
            G.add_node(("Certification", cert), kind="Certification", name=cert)
            G.add_edge(("Certification", cert), ("Skill", s), rel="VALIDATES", weight=1.0)
    print(f"[kg] graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def skill_graph(G):
    """Skill-only projection (CO_OCCURS) used for graph features."""
    H = nx.Graph()
    for u, v, d in G.edges(data=True):
        if d.get("rel") == "CO_OCCURS": H.add_edge(u[1], v[1], weight=d["weight"], pmi=d["pmi"])
    return H

def graph_features(H, skills):
    pr = nx.pagerank(H, weight="weight") if H.number_of_edges() else {}
    deg = dict(H.degree(weight="weight"))
    try:
        comms = nx.community.louvain_communities(H, weight="weight", seed=42)
    except Exception:
        comms = list(nx.community.greedy_modularity_communities(H, weight="weight"))
    cid = {n: i for i, c in enumerate(comms) for n in c}
    clus = nx.clustering(H, weight="weight")
    f = pd.DataFrame(index=skills)
    f["pagerank"] = [pr.get(s, 0) for s in skills]
    f["wdegree"] = [deg.get(s, 0) for s in skills]
    f["clustering"] = [clus.get(s, 0) for s in skills]
    f["community"] = [cid.get(s, -1) for s in skills]
    return f

def neighbor_matrix(H, skills, topk=8):
    """Row-normalised adjacency (top-k PMI neighbours) -> used for graph message passing."""
    idx = {s: i for i, s in enumerate(skills)}; A = np.zeros((len(skills), len(skills)))
    for s in skills:
        if s not in H: continue
        nb = sorted(H[s].items(), key=lambda kv: -kv[1]["weight"])[:topk]
        for t, d in nb:
            if t in idx: A[idx[s], idx[t]] = d["weight"]
    rs = A.sum(1, keepdims=True); rs[rs == 0] = 1
    return A / rs

def export_neo4j(G, outdir):
    """CSV + Cypher for Neo4j (use `LOAD CSV` or neo4j-admin import)."""
    nodes = [dict(id=f"{k}:{n}", label=k, name=n, **{x: v for x, v in d.items() if x in ("freq", "category")})
             for (k, n), d in G.nodes(data=True)]
    edges = [dict(src=f"{u[0]}:{u[1]}", dst=f"{v[0]}:{v[1]}", rel=d["rel"], weight=round(float(d.get("weight", 1)), 4))
             for u, v, d in G.edges(data=True)]
    pd.DataFrame(nodes).to_csv(outdir / "nodes.csv", index=False)
    pd.DataFrame(edges).to_csv(outdir / "edges.csv", index=False)
    (outdir / "load_graph.cypher").write_text("""// Copy nodes.csv / edges.csv into Neo4j import dir, then:
CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE;
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS r
MERGE (n:Entity {id:r.id}) SET n.name=r.name, n.label=r.label, n.freq=toInteger(r.freq), n.category=r.category;
LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS r
MATCH (a:Entity {id:r.src}), (b:Entity {id:r.dst})
CALL apoc.create.relationship(a, r.rel, {weight: toFloat(r.weight)}, b) YIELD rel RETURN count(rel);
// Example queries
// Emerging neighbours of Python:  MATCH (:Entity {name:'Python'})-[r:CO_OCCURS]-(m) RETURN m.name, r.weight ORDER BY r.weight DESC LIMIT 10;
// Skills required by a role:      MATCH (r:Entity {name:'Data Scientist'})-[q:REQUIRES]->(s) RETURN s.name, q.weight ORDER BY q.weight DESC;
""")
    print(f"[kg] exported {len(nodes)} nodes / {len(edges)} edges -> {outdir}")

def push_to_neo4j(G, uri="bolt://localhost:7687", user="neo4j", password="password"):
    """Live load with the official driver (optional)."""
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver(uri, auth=(user, password))
    with drv.session() as s:
        for (k, n), d in G.nodes(data=True):
            s.run(f"MERGE (x:{k} {{name:$n}}) SET x.freq=$f", n=n, f=int(d.get("freq", 0)))
        for u, v, d in G.edges(data=True):
            s.run(f"MATCH (a:{u[0]} {{name:$a}}), (b:{v[0]} {{name:$b}}) MERGE (a)-[r:{d['rel']}]->(b) SET r.weight=$w",
                  a=u[1], b=v[1], w=float(d.get("weight", 1)))
    drv.close()
