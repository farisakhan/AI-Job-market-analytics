// Copy nodes.csv / edges.csv into Neo4j import dir, then:
CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE;
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS r
MERGE (n:Entity {id:r.id}) SET n.name=r.name, n.label=r.label, n.freq=toInteger(r.freq), n.category=r.category;
LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS r
MATCH (a:Entity {id:r.src}), (b:Entity {id:r.dst})
CALL apoc.create.relationship(a, r.rel, {weight: toFloat(r.weight)}, b) YIELD rel RETURN count(rel);
// Example queries
// Emerging neighbours of Python:  MATCH (:Entity {name:'Python'})-[r:CO_OCCURS]-(m) RETURN m.name, r.weight ORDER BY r.weight DESC LIMIT 10;
// Skills required by a role:      MATCH (r:Entity {name:'Data Scientist'})-[q:REQUIRES]->(s) RETURN s.name, q.weight ORDER BY q.weight DESC;
