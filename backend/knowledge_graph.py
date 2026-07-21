"""
Knowledge Graph — Step 10.

Takes the entities produced by Step 9 (entity_extractor.extract_entities)
and turns them into a graph of nodes + edges, so the frontend can render
relationships between equipment, tickets, statuses, dates, etc.

Flow:
    text
      -> extract_entities(text)          (Step 9)
      -> [ {"entity": ..., "type": ...}, ... ]
      -> build_knowledge_graph(entities)  (this module)
      -> {"nodes": [...], "edges": [...]}

Kept independent of entity_extractor.py / rag.py: this module only knows
how to turn a flat list of {"entity", "type"} dicts into a graph. It
doesn't care where that list came from.

Grouping rule
-------------
Real documents (e.g. Breakdown_History) contain *multiple* equipment
records back to back — one Ticket_ID/Asset_ID/Date/Status block per
pump, repeated. So instead of connecting every entity in the whole text
to the *first* equipment found, entities are grouped into segments: each
"Equipment" entity starts a new group, and every entity that follows
belongs to that group until the next "Equipment" entity appears. This
matches the simple single-record example exactly, and stays correct when
a chunk contains several records.
"""

EQUIPMENT_TYPE = "Equipment"

def build_knowledge_graph(entities: list[dict]) -> dict:
    """
    Converts a list of {"entity": str, "type": str} dicts into:

        {
          "nodes": [{"id": ..., "label": ..., "type": ...}, ...],
          "edges": [{"source": ..., "target": ..., "relation": ...}, ...]
        }

    Nodes: one per unique entity string (dedup on id, first type seen wins).
    Edges: each non-Equipment entity is connected to the nearest preceding
    Equipment entity in the list, with relation = the entity's type.
    Entities appearing before any Equipment is seen are skipped (there is
    nothing to connect them to yet).

        build_knowledge_graph([
            {"entity": "Pump", "type": "Equipment"},
            {"entity": "EQ-2009", "type": "Equipment_ID"},
            {"entity": "BRK-8012", "type": "Ticket_ID"},
        ])
        -> {
            "nodes": [
                {"id": "Pump", "label": "Pump", "type": "Equipment"},
                {"id": "EQ-2009", "label": "EQ-2009", "type": "Equipment_ID"},
                {"id": "BRK-8012", "label": "BRK-8012", "type": "Ticket_ID"},
            ],
            "edges": [
                {"source": "Pump", "target": "EQ-2009", "relation": "Equipment_ID"},
                {"source": "Pump", "target": "BRK-8012", "relation": "Ticket_ID"},
            ],
        }
    """
    graph = {"nodes": [], "edges": []}

    if not entities:
        return graph

    # --- Nodes: one per unique entity id -----------------------------
    added_ids = set()
    for item in entities:
        node_id = item["entity"]
        if node_id not in added_ids:
            graph["nodes"].append({
                "id": node_id,
                "label": node_id,
                "type": item["type"],
            })
            added_ids.add(node_id)

    # --- Edges: segment by Equipment, link the rest of each group ----
    current_equipment = None
    for item in entities:
        if item["type"] == EQUIPMENT_TYPE:
            current_equipment = item["entity"]
            continue

        if current_equipment is None:
            # No equipment seen yet — nothing to anchor this entity to.
            continue

        if item["entity"] == current_equipment:
            continue

        graph["edges"].append({
            "source": current_equipment,
            "target": item["entity"],
            "relation": item["type"],
        })

    return graph
