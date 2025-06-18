# ingest_graph.py

import pandas as pd
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode, Relation

def main():
    # --- 1. Load & normalize your data (as before) ---
    df = pd.read_excel("DF1.xlsx", sheet_name="Sheet1")
    df["ingredients_list"] = (
        df["Item Description"]
          .str.replace(r"(?i)^Ingredients:\s*", "", regex=True)
          .str.split(",")
          .apply(lambda L: [i.strip() for i in L if i.strip()])
    )
    exploded = df.explode("ingredients_list")[["item_name", "ingredients_list"]]
    exploded = exploded.rename(columns={"ingredients_list": "ingredient"}).dropna()

    # Unique IDs for each node type
    dishes     = df["item_name"].dropna().unique().tolist()
    ingredients= exploded["ingredient"].unique().tolist()
    events     = df["Event Type"].dropna().unique().tolist()
    meals      = df["Meal Type"].dropna().unique().tolist()
    categories = df["Category"].dropna().unique().tolist()
    menus      = df["menu_id"].dropna().unique().tolist()

    # --- 2. Connect to Neo4j Desktop ---
    store = Neo4jPropertyGraphStore(
        url="bolt://127.0.0.1:7687",       # or neo4j://127.0.0.1:7687
        username="neo4j",
        password="Ashfaq8790",
        encrypted=False,
        refresh_schema=True,
        create_indexes=True
    )

    # --- 3. Build EntityNode list ---
    nodes = []

    # Dish nodes
    for dish in dishes:
        nodes.append(EntityNode(
            id=dish,
            label="Dish",
            name=dish,
            properties={"name": dish},
        ))

    # Ingredient nodes
    for ing in ingredients:
        nodes.append(EntityNode(
            id=ing,
            label="Ingredient",
            name=ing,
            properties={"name": ing},
        ))

    # EventType, MealType, Category, Menu
    for ev in events:
        nodes.append(EntityNode(id=ev, label="EventType",name=ev, properties={"name": ev}))
    for meal in meals:
        nodes.append(EntityNode(id=meal, label="MealType",name=meal, properties={"name": meal}))
    for cat in categories:
        nodes.append(EntityNode(id=cat, label="Category",name=cat ,properties={"name": cat}))
    for menu_id in menus:
        nodes.append(EntityNode(id=menu_id, label="Menu",name=menu_id, properties={"name": menu_id}))

    # Upsert all nodes in one call
    store.upsert_nodes(nodes)

    # --- 4. Build Relation list ---
    relations = []

    # Dish → Ingredient
    for dish, ing in exploded.itertuples(index=False):
        relations.append(Relation(
            source_id=dish,
            target_id=ing,
            label="CONTAINS"
        ))

    # Dish → EventType
    for dish, ev in df[["item_name","Event Type"]].drop_duplicates().itertuples(index=False):
        relations.append(Relation(source_id=dish, target_id=ev, label="SUITABLE_FOR"))

    # Dish → MealType
    for dish, meal in df[["item_name","Meal Type"]].drop_duplicates().itertuples(index=False):
        relations.append(Relation(source_id=dish, target_id=meal, label="SERVED_DURING"))

    # Dish → Category
    for dish, cat in df[["item_name","Category"]].drop_duplicates().itertuples(index=False):
        relations.append(Relation(source_id=dish, target_id=cat, label="BELONGS_TO"))

    # Menu → Dish
    for menu_id, dish in df[["menu_id","item_name"]].drop_duplicates().itertuples(index=False):
        relations.append(Relation(source_id=menu_id, target_id=dish, label="CONTAINS"))

    # (Optional) COMPLEMENTS edges if you have them
    # for dish, comp_list in df[["item_name","Complements"]].dropna().itertuples(index=False):
    #     for comp in comp_list.split(","):
    #         relations.append(Relation(source_id=dish, target_id=comp.strip(), label="COMPLEMENTS"))

    # Upsert all relations in one call
    # --- NEW SECTION: Dish → Dish CO-OCCURS relationships ---
    print("Building CO-OCCURS relationships between dishes...")

   # Import defaultdict for frequency tracking
    from collections import defaultdict

   # Track how often each pair of dishes appears together across all menus
    co_occurs_frequency = defaultdict(int)

   # Analyze each menu to find dishes that historically appear together
    for menu_id in df["menu_id"].dropna().unique():
      # Get all dishes that appeared in this specific menu
      dishes_in_menu = df[df["menu_id"] == menu_id]["item_name"].dropna().unique().tolist()
      
      # Create co-occurrence pairs for every combination of dishes in this menu
      # This captures the historical knowledge of what combinations actually worked
      for i, dish1 in enumerate(dishes_in_menu):
         for j, dish2 in enumerate(dishes_in_menu):
               # Skip self-relationships (a dish doesn't co-occur with itself)
               if i != j:
                  # Sort the pair to ensure consistency (prevents A→B and B→A being counted separately)
                  pair = tuple(sorted([dish1, dish2]))
                  co_occurs_frequency[pair] += 1

    print(f"Analyzed {len(df['menu_id'].dropna().unique())} menus")
    print(f"Found {len(co_occurs_frequency)} unique dish pairs that co-occur")

   # Create bidirectional CO-OCCURS relationships with frequency information
   # We only include pairs that appear together multiple times to ensure genuine patterns
    for (dish1, dish2), frequency in co_occurs_frequency.items():
      # Filter for meaningful co-occurrences (appeared together at least twice)
      # This eliminates random one-time pairings and focuses on proven combinations
      if frequency >= 2:
         # Determine relationship strength based on frequency
         strength = "high" if frequency >= 5 else "medium" if frequency >= 3 else "low"
         
         # Create relationship from dish1 to dish2
         relations.append(Relation(
               source_id=dish1,
               target_id=dish2,
               label="CO_OCCURS",
               properties={
                  "relationship_type": "historical_pairing",
                  "frequency": frequency,
                  "strength": strength,
                  "source": "menu_analysis"
               }
         ))
         
         # Create relationship from dish2 to dish1 (bidirectional)
         # This makes querying easier since we can find co-occurrences from either direction
         relations.append(Relation(
               source_id=dish2,
               target_id=dish1,
               label="CO_OCCURS",
               properties={
                  "relationship_type": "historical_pairing",
                  "frequency": frequency,
                  "strength": strength,
                  "source": "menu_analysis"
               }
         ))

   # Calculate and display statistics about the co-occurrence patterns we discovered
    total_co_occurs = sum(2 for freq in co_occurs_frequency.values() if freq >= 2)
    high_strength = sum(2 for freq in co_occurs_frequency.values() if freq >= 5)
    medium_strength = sum(2 for freq in co_occurs_frequency.values() if 3 <= freq < 5)

    print(f"Created {total_co_occurs} CO-OCCURS relationships")
    print(f"  - {high_strength} high-strength relationships (5+ co-occurrences)")
    print(f"  - {medium_strength} medium-strength relationships (3-4 co-occurrences)")
    print(f"  - {total_co_occurs - high_strength - medium_strength} low-strength relationships (2 co-occurrences)")

# --- END NEW SECTION ---
    store.upsert_relations(relations)

    print("Graph ingestion complete — check Neo4j Desktop for your nodes & relationships.")

    store.close()

if __name__ == "__main__":
    main()
