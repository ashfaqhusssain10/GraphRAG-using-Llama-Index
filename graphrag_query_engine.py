# graphrag_query_engine.py
# graphrag_query_engine.py

#some Bugs related to the Items Extraction or naming issue between the neo4j and the json file 
"""
GraphRAG Query Engine for Template Integration
Main engine that connects template requirements with GraphRAG system
"""
import streamlit as st
import re
import json
import time
import os                    # <-- ADD THIS
import pickle                # <-- ADD THIS
from datetime import datetime # <-- ADD THIS
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import logging
import random
import threading 
import shutil
from llama_index.core.llms import ChatMessage
# Import your existing GraphRAG components
try:
    from ex1 import (
        FoodServiceGraphRAGStore, 
        FoodServiceGraphRAGQueryEngine,
        Neo4jGraphRAGAdapter,
        setup_graphrag_core_system,
        setup_complete_community_graphrag_system
    )
except ImportError as e:
    logging.error(f"Could not import GraphRAG components: {e}")
    raise

from template_bridge import TemplateBridge, create_template_bridge
from config import (
    GRAPHRAG_CONFIG, 
    TEMPLATE_CONFIG, 
    INSIGHT_CONFIG,
    ERROR_CONFIG,
    get_category_mapping,
    get_cypher_filter
)
# ADD THIS AFTER IMPORTS - AROUND LINE 20
# Global cache to prevent double initialization
_GRAPHRAG_SYSTEM_CACHE = {
    'adapter': None,
    'query_engine': None,
    'initialized': False
} 
_CACHED_LOCK = threading.RLock()
_SYSTEM_INITIALIZATION_LOCK = threading.RLock()
_SYSTEM_INITIALIZATION_IN_PROGRESS = False 
_GRAPHRAG_SYSTEM_CACHE = {
    'initialized': False,
    'query_engine': None,
    'communities_data': None,
    'execution_id': None
}

# Add this to graphrag_query_engine.py - REPLACE the existing get_cached_graphrag_system function
def get_cache_directory():
    """Ensure consistent cache location"""
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, "graphrag_cache")
    print(f"📁 Using cache directory: {cache_dir}")
    return cache_dir

# REPLACE the pickle-based cache functions in graphrag_query_engine.py with these JSON-based versions:

def save_communities_safely(adapter, cache_dir="graphrag_cache"):
    """
    RELIABLE JSON CACHING: No more pickle issues
    """
    with _CACHED_LOCK:
        if cache_dir is None:
            cache_dir = get_cache_directory()
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create process-level lock file
        lock_file = os.path.join(cache_dir, "save_operation.lock")
        cache_file = os.path.join(cache_dir, "communities_data.json")  # Changed from .pkl to .json
        metadata_file = os.path.join(cache_dir, "metadata.json")
        temp_cache_file = cache_file + ".tmp"
        temp_metadata_file = metadata_file + ".tmp"
        
        try:
            # Create lock file
            with open(lock_file, 'w') as f:
                f.write(str(os.getpid()))
            
            # Extract data in JSON-serializable format (NO PICKLE!)
            community_summaries = adapter.graphrag_store.get_community_summaries()
            entity_communities = dict(adapter.graphrag_store.entity_communities)
            
            # Convert to JSON-safe format
            communities_data = {
                'community_summaries': {str(k): str(v) for k, v in community_summaries.items()},
                'entity_communities': {str(k): list(v) if hasattr(v, '__iter__') else [v] for k, v in entity_communities.items()},
                'cache_version': '3.0_JSON',
                'created_at': datetime.now().isoformat()
            }
            
            print(f"📊 Preparing to cache:")
            print(f"   Communities: {len(communities_data['community_summaries'])}")
            print(f"   Entities: {len(communities_data['entity_communities'])}")
            
            # Atomic save to JSON (much more reliable than pickle)
            with open(temp_cache_file, 'w', encoding='utf-8') as f:
                json.dump(communities_data, f, indent=2, ensure_ascii=False)
            
            # Save metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'community_count': len(communities_data['community_summaries']),
                'entity_count': len(communities_data['entity_communities']),
                'cache_version': '3.0_JSON',
                'format': 'JSON'
            }
            
            with open(temp_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Atomic move to final locations
            if os.path.exists(cache_file):
                os.remove(cache_file)
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
                
            os.rename(temp_cache_file, cache_file)
            os.rename(temp_metadata_file, metadata_file)
            
            print(f"✅ Successfully cached {metadata['community_count']} communities (JSON format)")
            print(f"📁 Cache location: {cache_file}")
            return True
            
        except Exception as e:
            print(f"❌ JSON caching failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Cleanup temp files on failure
            for temp_file in [temp_cache_file, temp_metadata_file]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
            return False
            
        finally:
            # Always remove lock file
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except:
                    pass

def load_communities_safely(cache_dir="graphrag_cache"):
    """
    RELIABLE JSON LOADING: No more pickle deserialization issues
    """
    with _CACHED_LOCK:
        # Support both JSON and legacy pickle formats
        json_cache_file = os.path.join(cache_dir, "communities_data.json")
        pickle_cache_file = os.path.join(cache_dir, "communities_data.pkl")
        metadata_file = os.path.join(cache_dir, "metadata.json")
        
        # Prefer JSON format
        cache_file = json_cache_file if os.path.exists(json_cache_file) else pickle_cache_file
        is_json_format = cache_file.endswith('.json')
        
        # Add process-level lock file to prevent race conditions
        lock_file = os.path.join(cache_dir, "cache.lock")
        
        if os.path.exists(lock_file):
            print("🔒 Cache operation in progress, waiting...")
            while os.path.exists(lock_file):
                time.sleep(0.1)
                
    print(f"🔍 CACHE DIAGNOSTIC:")
    print(f"   📁 Looking for cache in: {os.path.abspath(cache_dir)}")
    print(f"   📄 JSON cache exists: {os.path.exists(json_cache_file)}")
    print(f"   📄 Pickle cache exists: {os.path.exists(pickle_cache_file)}")
    print(f"   📄 Metadata file exists: {os.path.exists(metadata_file)}")
    print(f"   🎯 Using format: {'JSON' if is_json_format else 'PICKLE'}")
    
    if os.path.exists(cache_file):
        print(f"   📊 Cache file size: {os.path.getsize(cache_file)} bytes")
    
    if not os.path.exists(cache_file) or not os.path.exists(metadata_file):
        print("💡 No valid cache found")
        return None
    
    try:
        # Load metadata first
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Check cache age (24 hours)
        cache_time = datetime.fromisoformat(metadata['created_at'])
        cache_age_hours = (datetime.now() - cache_time).total_seconds() / 3600
        
        if cache_age_hours > 240:
            print(f"🕒 Cache expired ({cache_age_hours:.1f}h old)")
            return None
        
        print(f"⏰ Cache age: {cache_age_hours:.1f} hours (valid)")
        
        # Load community data based on format
        if is_json_format:
            print(f"📥 Loading JSON cache...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                communities_data = json.load(f)
            
            # Convert back from JSON-safe format
            communities_data['community_summaries'] = {
                int(k) if k.isdigit() else k: v 
                for k, v in communities_data['community_summaries'].items()
            }
            communities_data['entity_communities'] = {
                k: set(v) if isinstance(v, list) else v 
                for k, v in communities_data['entity_communities'].items()
            }
            
            print(f"✅ JSON cache loaded successfully")
        else:
            print(f"📥 Loading legacy pickle cache...")
            with open(cache_file, 'rb') as f:
                communities_data = pickle.load(f)
            print(f"⚠️ Loaded pickle cache (consider upgrading to JSON)")
        
        print(f"🚀 Loaded {metadata['community_count']} cached communities")
        return communities_data
        
    except Exception as e:
        print(f"❌ Cache loading failed: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        if is_json_format:
            print(f"💡 JSON parsing error - cache may be corrupted")
        else:
            print(f"💡 Pickle deserialization error - try clearing cache")
        return None
@st.cache_resource(ttl=604800, show_spinner=False)
def get_cached_graphrag_system():
    """
    ROBUST JSON CACHE SYSTEM: No more pickle failures
    """
    global _SYSTEM_INITIALIZATION_IN_PROGRESS
    
    with _SYSTEM_INITIALIZATION_LOCK:
        if _SYSTEM_INITIALIZATION_IN_PROGRESS:
            print("🚫 System initialization already in progress, waiting...")
            while _SYSTEM_INITIALIZATION_IN_PROGRESS:
                time.sleep(0.1)
            
            if (_GRAPHRAG_SYSTEM_CACHE['initialized'] and 
                _GRAPHRAG_SYSTEM_CACHE['adapter'] is not None):
                print("⚡ Using system built by concurrent initialization")
                return _GRAPHRAG_SYSTEM_CACHE['adapter'], _GRAPHRAG_SYSTEM_CACHE['query_engine']
        
        execution_id = str(int(time.time() * 1000))
        print(f"🔧 GraphRAG system initialization - ID: {execution_id}")
        
        # Check if we already have a valid system in global cache
        if (_GRAPHRAG_SYSTEM_CACHE['initialized'] and 
            _GRAPHRAG_SYSTEM_CACHE['adapter'] is not None):
            print(f"⚡ Using existing global cache - ID: {execution_id}")
            return _GRAPHRAG_SYSTEM_CACHE['adapter'], _GRAPHRAG_SYSTEM_CACHE['query_engine']
        
        # Mark initialization as in progress
        _SYSTEM_INITIALIZATION_IN_PROGRESS = True
        
        try:
            print(f"🏗️ Building GraphRAG system - ID: {execution_id}")
            
            # Try to load cached communities (JSON format preferred)
            cached_communities = load_communities_safely()
            
            # Build fresh system objects
            #
            adapter = setup_graphrag_core_system()
            #adapter = setup_complete_community_graphrag_system()
            if not adapter:
                raise Exception("Failed to setup GraphRAG core system")

            # 🔥 FIX: Ensure the graph is loaded from Neo4j BEFORE attempting to use it
            print(f"🔄 Loading graph data from Neo4j into adapter.graphrag_store.graph - ID: {execution_id}")
            adapter.sync_from_neo4j()
            print(f"✅ Graph data loaded. Nodes: {len(adapter.graphrag_store.graph.nodes)}")

            query_engine = None # Initialize query_engine to None
            
            # IMPROVED CACHE RESTORATION with JSON reliability
            cache_restored_successfully = False
            
            if cached_communities:
                print(f"📥 Attempting cache restoration - ID: {execution_id}")
                print(f"   🔍 Cache format: {cached_communities.get('cache_version', 'Unknown')}")
                print(f"   🔍 Communities to restore: {len(cached_communities['community_summaries'])}")
                print(f"   🔍 Entities to restore: {len(cached_communities['entity_communities'])}")
                
                try:
                    # Validate cache data structure first
                    if ('community_summaries' not in cached_communities or 
                        'entity_communities' not in cached_communities):
                        raise ValueError("Cache missing required data structures")
                    
                    # Ensure adapter has required attributes
                    if not hasattr(adapter.graphrag_store, 'community_summaries'):
                        adapter.graphrag_store.community_summaries = {}
                    if not hasattr(adapter.graphrag_store, 'entity_communities'):
                        adapter.graphrag_store.entity_communities = {}
                    
                    # Restore with type safety
                    print(f"   📋 Restoring community summaries...")
                    adapter.graphrag_store.community_summaries = dict(cached_communities['community_summaries'])
                    
                    print(f"   📋 Restoring entity communities...")
                    adapter.graphrag_store.entity_communities = dict(cached_communities['entity_communities'])
                    
                    # Validate restoration success
                    restored_communities = len(adapter.graphrag_store.community_summaries)
                    restored_entities = len(adapter.graphrag_store.entity_communities)
                    
                    if restored_communities > 0 and restored_entities > 0:
                        cache_restored_successfully = True
                        print(f"   ✅ JSON CACHE RESTORATION SUCCESSFUL!")
                        print(f"      📊 Restored: {restored_communities} communities, {restored_entities} entities")
                        print(f"      🎯 Cache hit ratio: 100%")
                    else:
                        print(f"   ❌ Cache restoration failed: Empty data after restoration")
                        print(f"      📊 Communities: {restored_communities}, Entities: {restored_entities}")
                        
                except Exception as e:
                    print(f"   ❌ Cache restoration FAILED: {e}")
                    print(f"      🔍 Exception type: {type(e).__name__}")
                    cache_restored_successfully = False
            else:
                print(f"   💡 No cached communities available - first run or cache expired")
            
            # Only rebuild if cache restoration failed
            if not cache_restored_successfully:
                print(f"🔨 Building communities from scratch - ID: {execution_id}")
                success = adapter.build_communities_from_neo4j()
                
                if success:
                    # Save the new communities in JSON format
                    print(f"💾 Saving communities to JSON cache...")
                    save_communities_safely(adapter)
                    print(f"✅ Communities cached for future use")
                else:
                    print(f"❌ Community building failed - ID: {execution_id}")
            else:
                print(f"⚡ Using cached communities, SKIPPING rebuild - ID: {execution_id}")
                print(f"   🚀 Performance boost: ~5-10 minutes saved")
            
            # Create query engine now that communities are loaded/built
            print("\n5️⃣ Creating community-powered query engine...")
            query_engine = adapter.create_query_engine(similarity_top_k=5)
            print("✅ Query engine created.")

            # Update global cache
            _GRAPHRAG_SYSTEM_CACHE.update({
                'adapter': adapter,
                'query_engine': query_engine,
                'initialized': True
            })
            
            final_communities = len(adapter.graphrag_store.community_summaries)
            print(f"🎯 GraphRAG system ready with {final_communities} communities - ID: {execution_id}")
            return adapter, query_engine
            
        finally:
            # Always clear the initialization flag
            _SYSTEM_INITIALIZATION_IN_PROGRESS = False



# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphRAGTemplateEngine:
    """
    Main engine that integrates GraphRAG with template-based recommendations
    """
    
    def __init__(self, template_file: str = None):
        """
        Initialize the GraphRAG Template Engine
        
        Args:
            template_file: Optional custom template file path
        """
        self.template_bridge = create_template_bridge(template_file)
        self.graphrag_adapter = None
        self.graphrag_query_engine = None
        self.insight_generator = InsightGenerator()
        
        # Initialize GraphRAG system
        self._initialize_graphrag_system()
        
        logger.info("GraphRAGTemplateEngine initialized successfully")
    
    def _initialize_graphrag_system(self):
        """Initialize GraphRAG system using global cache"""
        import time
        execution_id = str(int(time.time() * 1000))
        logger.info(f"🆔 GraphRAG initialization started - Execution ID: {execution_id}")
        try:
            logger.info("🔍 Checking for cached GraphRAG system...")
            
            # FIXED: Use cached system instead of rebuilding
            adapter, query_engine = get_cached_graphrag_system()
            
            if adapter and query_engine:
                self.graphrag_adapter = adapter
                self.graphrag_query_engine = query_engine
                logger.info("✅ GraphRAG system ready - Execution ID: {execution_id}")
            else:
                raise Exception("Failed to get GraphRAG system")
                
        except Exception as e:
            logger.error(f"GraphRAG initialization failed: {e}")
            raise
    
    def recommend_with_graphrag(self, event_type: str, budget_per_head: int) -> Dict[str, Any]:
        """
        Generate template-based recommendation using GraphRAG intelligence
        
        Args:
            event_type: "Traditional", "Party", or "Premium"
            budget_per_head: Budget amount per person
        
        Returns:
            Complete recommendation with GraphRAG suggestions and insights
        """
        logger.info(f"Generating recommendation for {event_type} event with ₹{budget_per_head} budget")
        
        try:
            # Step 1: Find appropriate template
            template = self.template_bridge.find_template_by_budget(budget_per_head, event_type)
            if not template:
                return self._handle_no_template_found(event_type, budget_per_head)
            
            # Step 2: Extract template requirements
            requirements = self.template_bridge.extract_requirements(template)
            
            # Step 3: Build GraphRAG queries
            query_specs = self.template_bridge.build_graphrag_queries(requirements, event_type)
            
            # Step 4: Execute GraphRAG queries
            graphrag_suggestions = self._execute_graphrag_queries(query_specs, event_type)
            
            # Step 5: Map suggestions to template structure
            filled_template = self._fill_template_with_suggestions(template, requirements, graphrag_suggestions)
            
            # Step 6: Generate insights
            insights = self.insight_generator.generate_template_insights(
                filled_template, graphrag_suggestions, event_type
            )
            
            # Step 7: Create final recommendation
            recommendation = {
                "template_id": template["id"],
                "template_name": template["name"],
                "budget_range": template["pricing"]["budget"],
                "event_type": event_type,
                "items": filled_template["items"],
                "insights": insights,
                "graphrag_metadata": {
                    "queries_executed": len(query_specs),
                    "suggestions_found": len(graphrag_suggestions),
                    "fallbacks_used": self._count_fallbacks(graphrag_suggestions)
                }
            }
            
            logger.info(f"Generated recommendation: {template['name']} with {len(filled_template['items'])} items")
            return recommendation
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return self._handle_recommendation_error(event_type, budget_per_head, str(e))
    
    def _execute_graphrag_queries(self, query_specs: List[Dict[str, Any]], event_type: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Execute GraphRAG queries for each category requirement
        
        Args:
            query_specs: List of query specifications from template bridge
            event_type: Event type for context
        
        Returns:
            Dictionary mapping categories to suggested items with insights
        """
        suggestions = {}
        
        for query_spec in query_specs:
            category = query_spec["category"]
            count_needed = query_spec["count"]
            budget_allocation = query_spec.get("budget_allocation")
            
            logger.info(f"Querying GraphRAG for {count_needed} {category} items")
            
            try:
                # Execute category-specific query
                category_suggestions = self._query_category_items(query_spec, event_type, budget_allocation)
                
                # Ensure we have exact count needed
                if len(category_suggestions) < count_needed:
                    logger.warning(f"Insufficient {category} suggestions: got {len(category_suggestions)}, needed {count_needed}")
                    category_suggestions = self._handle_insufficient_suggestions(query_spec, category_suggestions, event_type)
                elif len(category_suggestions) > count_needed:
                    # Take top suggestions based on relevance
                    category_suggestions = category_suggestions[:count_needed]
                
                suggestions[category] = category_suggestions
                
            except Exception as e:
                logger.error(f"Query failed for category {category}: {e}")
                suggestions[category] = self._handle_query_failure(query_spec, event_type)
        
        return suggestions
    
    def _query_category_items(self, query_spec: Dict[str, Any], event_type: str, budget_allocation: int = None) -> List[Dict[str, Any]]:
        """
        Query GraphRAG for specific category items
        
        Args:
            query_spec: Query specification with category, count, filters
            event_type: Event type for context
        
        Returns:
            List of suggested items with insights
        """
        category = query_spec["category"]
        count = query_spec["count"]
        cypher_filter = query_spec["cypher_filter"]
        if budget_allocation:
            print(f"\n🔍 DIAGNOSTIC: Analyzing baseline selection for {category}")
            self._diagnose_baseline_selection(category, budget_allocation, event_type)
            baseline_candidates = self._diagnose_baseline_selection(category, budget_allocation, event_type)
        
            print(f"🎯 LLM will likely choose from these top candidates:")
            for item_name, analysis in baseline_candidates[:3]:
                print(f"   • {item_name}: ₹{analysis['price']} (distance: ₹{analysis['budget_distance']})")    
        # Retrieve relevant community summaries for the category
        relevant_summaries_text = self._get_relevant_community_summaries_for_category(
            category, event_type
        )
        
        # Build GraphRAG query string
        query_text = self._build_category_query_text(category, event_type, count, budget_allocation)
        
        # Execute query using existing GraphRAG system
        try:
            messages = [ChatMessage(role="user", content=query_text)]
            raw_response_obj = self.graphrag_query_engine.llm.chat(messages)
            raw_response = str(raw_response_obj)

            logger.info(f"LLM Full Response for {category} (Analysis + JSON):\n\n{raw_response}\n")
            # Extract specific items from GraphRAG response
             # 🔥 ADD THIS POST-LLM VALIDATION:
            if budget_allocation:
                # selected_item = self._extract_baseline_item_from_response(raw_response)
                # if selected_item:
                #     self._validate_llm_selection(selected_item, baseline_candidates, budget_allocation)
                pass 
            extracted_items = self._extract_items_from_response(raw_response, category, count)
                        # --- GUARDRAIL: Enforce the exact count ---
            if len(extracted_items) > count:
                logger.warning(
                    f"LLM provided {len(extracted_items)} items for {category}, "
                    f"but only {count} were requested. Truncating to the required count."
                )
                extracted_items = extracted_items[:count]
            # Enhance with co-occurrence data
            enhanced_items = self._enhance_with_co_occurrence_data(extracted_items, event_type)
            
            return enhanced_items
            
        except Exception as e:
            logger.error(f"GraphRAG query execution failed for {category}: {e}")
            raise




    def _get_relevant_community_summaries_for_category(self, category: str, event_type: str) -> str:
        """
        Retrieves and formats relevant community summaries for a given category and event type.
        It finds Dish entities that belong to the specified category and collects their community summaries,
        along with their associated ingredients.
        """
        relevant_communities_ids = set()
        
        # Get the category mapping from config.py
        category_map = get_category_mapping(category)
        if not category_map:
            logger.warning(f"Could not determine category mapping for category: '{category}'. Skipping community summaries.")
            return ""

        config_labels = category_map.get("graphrag_labels", [])
        
        target_category_node_name = None
        # Try to find a specific Category node name from config_labels (excluding "Dish")
        for label in config_labels:
            if label != "Dish": # "Dish" is a label for items, not a category node name
                # Attempt to retrieve the node by its ID. The get method returns a list of nodes.
                # The label in config_labels *should* be the ID of the Category node (e.g., "Starters")
                potential_category_nodes = self.graphrag_adapter.graphrag_store.get(ids=[label])
                
                # Check if any node was found and if its *actual* label attribute is 'Category'
                if potential_category_nodes and potential_category_nodes[0].label == "Category":
                    target_category_node_name = potential_category_nodes[0].id
                    logger.debug(f"Found Category node: ID='{target_category_node_name}', Label='Category'")
                    break
        
        if not target_category_node_name:
            logger.info(f"No 'Category' node found in graph for labels: {config_labels}. Skipping category-specific community summaries.")
            return ""

        # Find Dish nodes linked to this category node
        # We need to query for relationships where the target is a Dish and source is the category
        # Since get_triplets doesn't support 'relation_names', we will iterate and filter manually.
        relevant_dish_names = set()
        
        # Iterate through all relationships in the in-memory graph
        for relation_id, relation_obj in self.graphrag_adapter.graphrag_store.graph.relations.items():
            # Check if the relationship is 'BELONGS_TO'
            if relation_obj.label == "BELONGS_TO":
                # Check if the source of the relationship is our target category node
                if relation_obj.target_id == target_category_node_name:
                    # Get the target node object from the graph store
                    source_node = self.graphrag_adapter.graphrag_store.get(ids=[relation_obj.source_id])
                    
                    # Check if the target node was found and if its actual label is 'Dish'
                    if source_node and source_node[0].label == "Dish":
                        relevant_dish_names.add(source_node[0].name)

        logger.debug(f"DEBUG_COMMUNITY_SUMMARIES: Found {len(relevant_dish_names)} relevant dishes for category '{category}'")

        # Now, collect community summaries for these relevant dishes
        community_summaries_text = []
        for dish_name in relevant_dish_names:
            entity_communities = self.graphrag_adapter.graphrag_store.get_entity_communities(dish_name)
            for community_id in entity_communities:
                if community_id not in relevant_communities_ids:
                    summary = self.graphrag_adapter.graphrag_store.get_community_summaries().get(community_id)
                    if summary:
                        community_summaries_text.append(f"Community Summary for '{dish_name}' (ID: {community_id}):\n{summary}")
                        relevant_communities_ids.add(community_id)
        
        # Finally, gather ingredients for these dishes to enrich context
        ingredient_details = []
        for dish_name in relevant_dish_names:
            # Query for ingredients linked to this dish
            # We will iterate through all relations and filter for 'CONTAINS'
            ingredients = []
            for relation_id, relation_obj in self.graphrag_adapter.graphrag_store.graph.relations.items():
                if relation_obj.label == "CONTAINS" and relation_obj.source_id == dish_name:
                    # Get the target node object
                    target_node = self.graphrag_adapter.graphrag_store.get(ids=[relation_obj.target_id])
                    if target_node and target_node[0].label == "Ingredient":
                        ingredients.append(target_node[0].name)
            if ingredients:
                ingredient_details.append(f"Ingredients for '{dish_name}': {', '.join(ingredients)}")

        # Combine all collected insights
        all_insights = []
        if community_summaries_text:
            all_insights.append("--- Community Insights ---\n" + "\n\n".join(community_summaries_text))
        if ingredient_details:
            all_insights.append("--- Ingredient Details ---\n" + "\n".join(ingredient_details))
        
        if not all_insights:
            logger.info(f"No community summaries or ingredient details found for category: '{category}'.")
            return ""

        return "\n\n" + "\n\n".join(all_insights)

    def _load_pricing_inventory(self) -> Dict[str, Any]:
        """Loads and caches the pricing and inventory data from the JSON file."""
        if hasattr(self, '_pricing_inventory') and self._pricing_inventory:
            return self._pricing_inventory
        try:
            with open("items_price_uom.json", 'r', encoding='utf-8') as f:
                pricing_data = json.load(f)
                self._pricing_inventory = {item["item_name"]: item for item in pricing_data}
                return self._pricing_inventory
        except Exception as e:
            logger.error(f"Failed to load pricing inventory: {e}")
            return {}

    def _filter_inventory_by_category(self, inventory: Dict[str, Any], category: str) -> Dict[str, Any]:
        category_map = get_category_mapping(category)
        target_categories = category_map.get("graphrag_labels") if category_map else []  # ✅ CORRECT
        if not target_categories:
            return {}
        return {
            name: data for name, data in inventory.items() 
            if data.get("category") in target_categories
        }
    def _estimate_category_budget(self, category: str) -> int:
        """Estimates a budget for a category if not provided."""
        if 'starter' in category:
            return 60
        if 'biryani' in category:
            return 150
        if 'dessert' in category:
            return 50
        return 80
    # REPLACE _extract_items_from_graphrag_response with this simpler version:
    def _extract_items_from_response(self, response: str, category: str, count: int) -> List[Dict[str, Any]]:
        """
        Extracts structured item data from the LLM's JSON response.
        This version is enhanced to handle a list of objects with name, quantity, and UOM,
        and safely ignores any preceding analysis text.
        """
        inventory = self._load_pricing_inventory()
        found_items = []
        found_names = set()

        try:
            # Clean the response to find the JSON blob, ignoring any text before it.
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                logger.error(f"No JSON object found in LLM response for {category}.")
                return []
            
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            # Expects a list of objects with name, quantity, uom
            item_objects = data.get("baseline", [])
            
            for item_obj in item_objects:
                if len(found_items) >= count:
                    break
                
                item_name_raw = item_obj.get("name")
                quantity = item_obj.get("quantity")
                uom = item_obj.get("uom")

                if not all([item_name_raw, quantity is not None, uom]):
                    logger.warning(f"LLM returned incomplete item object, skipping: {item_obj}")
                    continue

                clean_item_name = re.sub(r'\(.*\)', '', item_name_raw).strip()

                # Find the full item details from our inventory for validation
                matched_item = None
                for inv_name, inv_data in inventory.items():
                    if clean_item_name.lower() in inv_name.lower():
                        matched_item = inv_data
                        break
                
                if matched_item and matched_item["item_name"].lower() not in found_names:
                    # The LLM now provides the quantity, so we use it directly.
                    found_items.append({
                        "name": matched_item["item_name"],
                        "category": matched_item["category"],
                        # This combined quantity field is used by the frontend.
                        "quantity": f"{quantity}{uom}",
                        "source": "llm_json_extraction",
                        "match_confidence": 1.0
                    })
                    found_names.add(matched_item["item_name"].lower())
                else:
                    logger.warning(f"LLM returned item '{item_name_raw}' which was not found in inventory after cleaning.")

        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from LLM response for {category}. Response: {response[:500]}...")
            return []
        except Exception as e:
            logger.error(f"Error parsing LLM response for {category}: {e}")
            return []

        if len(found_items) < count:
            logger.warning(f"LLM returned {len(found_items)}/{count} valid items for {category}.")

        return found_items[:count]

    # def _extract_baseline_section(self, response: str, category: str) -> List[Dict[str, Any]]:
    #     """Extract items specifically from baseline sections"""
        
    #     # Target patterns for baseline sections
    #     baseline_patterns = [
    #         r'(?i)baseline[^:]*:\s*-?\s*([^\n(₹]+)',  # "Baseline Selection: - Item"
    #         r'(?i)baseline[^:]*:\s*([A-Z][a-zA-Z\s]+(?:Biryani|Kebab|Tikka|Curry|Rice|Naan|Paneer|Chicken))',  # Direct baseline mentions
    #         r'-\s*([A-Z][a-zA-Z\s]+(?:Biryani|Kebab|Tikka|Curry|Rice|Naan|Paneer|Chicken))',  # Bullet point items
    #     ]
        
    #     inventory = self._load_pricing_inventory()
    #     category_items = self._filter_inventory_by_category(inventory, category)
        
    #     extracted_items = []
    #     found_names = set()
        
    #     for pattern in baseline_patterns:
    #         matches = re.findall(pattern, response)
    #         for match in matches:
    #             clean_name = self._clean_extracted_name(match)
    #             matched_item = self._intelligent_fuzzy_match(clean_name, category_items, category)
                
    #             if matched_item and matched_item["item_name"].lower() not in found_names:
    #                 extracted_items.append({
    #                     "name": matched_item["item_name"],
    #                     "category": matched_item["category"],
    #                     "source": "baseline_section",
    #                     "match_confidence": self._calculate_match_confidence(clean_name, matched_item["item_name"])
    #                 })
    #                 found_names.add(matched_item["item_name"].lower())
        
    #     return extracted_items

    def _handle_no_template_found(self, event_type: str, budget: int) -> Dict[str, Any]:
        """Handles cases where no suitable template is found."""
        logger.warning(f"No template found for {event_type} at ₹{budget}")
        return {
            "error": "No matching template found",
            "message": f"We could not find a pre-defined template for a {event_type} event with a budget of ₹{budget}. Try adjusting the budget."
        }
    
    def _handle_recommendation_error(self, event_type: str, budget: int, error_msg: str) -> Dict[str, Any]:
        """Handles generic errors during the recommendation process."""
        logger.error(f"Recommendation error for {event_type} at ₹{budget}: {error_msg}")
        return {
            "error": "Recommendation Generation Failed",
            "message": f"An unexpected error occurred: {error_msg}. Please check the logs."
        }
    def _handle_insufficient_suggestions(self, query_spec: Dict[str, Any], current_suggestions: List[Dict], event_type: str) -> List[Dict[str, Any]]:
            """Handles cases where GraphRAG returns fewer items than needed."""
            category = query_spec["category"]
            count_needed = query_spec["count"]
            
            logger.warning(f"Insufficient suggestions for {category}. Needed {count_needed}, got {len(current_suggestions)}. Using fallbacks.")
            
            fallbacks = {
                "starter": ["Chicken 65", "Paneer Tikka", "Veg Manchurian"],
                "main_biryani": ["Chicken Dum Biryani", "Veg Biryani"],
                "dessert": ["Gulab Jamun", "Double Ka Meetha"]
            }
            
            inventory = self._load_pricing_inventory()
            potential_fallbacks = fallbacks.get(category, [])
            
            for item_name in potential_fallbacks:
                if len(current_suggestions) >= count_needed:
                    break
                if not any(d['name'] == item_name for d in current_suggestions):
                    if item_name in inventory:
                        item_data = inventory[item_name]
                        current_suggestions.append({
                            "name": item_data["item_name"],
                            "category": item_data["category"],
                            "source": "fallback",
                            "insight": "Popular fallback selection"
                        })
            
            return current_suggestions

    def _handle_query_failure(self, query_spec: Dict[str, Any], event_type: str) -> List[Dict[str, Any]]:
        """Handles failures in a specific GraphRAG category query by returning fallbacks."""
        logger.error(f"Query failed for {query_spec['category']}. Providing fallback items.")
        return self._handle_insufficient_suggestions(query_spec, [], event_type)
        

# PREMIUM PROMPT ARCHITECTURE v3.0 - SURGICAL REPLACEMENT
# REPLACE ENTIRE METHOD WITH THIS ENGINEERED VERSION:

    def _build_category_query_text(self, category: str, event_type: str, count: int, budget_allocation: int = None,community_insights: str = "") -> str:
        """
        Premium-optimized GraphRAG query builder with inventory awareness
        Engineering Version: 3.1 - JSON Output Focus
        """
        
        # Load pricing inventory for prompt integration
        inventory_data = self._load_pricing_inventory()
        category_inventory = self._filter_inventory_by_category(inventory_data, category)
        
        # Calculate budget flexibility range
        base_budget = budget_allocation or self._estimate_category_budget(category)
        budget_range = {
            'min': base_budget - 50,
            'baseline': base_budget, 
            'max': base_budget + 50
        }

        # New JSON output instruction
        # New JSON output instruction with reasoning prompt
        # json_output_instruction = (
        #     '\n\n**Output Format:**'
        #     '\n1. **Analysis:** First, provide a brief analysis. If the inventory UOM (like "Pcs") conflicts with a general weight (like "50g"), explain how you determined the correct quantity (e.g., "The template suggested 50g, but Chicken Lollipop is sold by the piece. A standard serving is 2 Pcs, so I have chosen that quantity.").'
        #     '\n2. **JSON Output:** After your analysis, provide a single, valid JSON object and nothing else. '
        #     'The JSON object must have one key: "baseline". '
        #     'The value for "baseline" must be a list of objects, each containing the item\'s "name", your calculated "quantity", and its "uom" (Unit of Measurement, e.g., "g", "kg", "Pcs", "ml").'
        #     '\n\nExample Response:'
        #     '\nAnalysis: The template requested a starter around 80g. I selected \'Chicken 65\'. A standard portion is 120g, which fits the budget and is a better serving size.'
        #     '\n{"baseline": [{"name": "Chicken 65", "quantity": 120, "uom": "g"}]}'
        # )
        # New, more explicit JSON output instruction
        json_output_instruction = (
            #f'\n\n**Analysis:** First, provide a brief analysis explaining your choices, especially how you handled any unit of measurement (UOM) conflicts and calculated quantities.'
            f'\n\n**Analysis:** First, provide a detailed analysis explaining your decision-making process for item selection, focusing on how you used the provided inventory and `cmp_peak_price` to stay within the budget range. Describe any trade-offs or assumptions made regarding quantity and UOM to meet the budget. Clearly state the total estimated cost of your selected baseline items and compare it to the given budget range.'
            f'\n\n**JSON Output:** After your analysis, provide a single, valid JSON object and nothing else. This object MUST adhere to these rules:'
            f'\n1. It must contain one key: "baseline".'
            f'\n2. The "baseline" value must be a list of item objects.'
            f'\n\n**CRITICAL RULES:**'
            f'\n1. **Use Peak Price:** You MUST use the `cmp_peak_price` from the inventory for all cost calculations.'
            f'\n2. **Strict Item Count:** The "baseline" array MUST contain exactly {count} item(s).'
            f'\n3. **JSON Schema:** Each object in the list must contain "name" (string), "quantity" (number), and "uom" (string).'
            f'\n\n**Example for count={count}**: {{"baseline": [{{"name": "Example Item", "quantity": 100, "uom": "g"}}]}}'
        )
         # Add community insights to the prompt if available
        community_context = ""
        if community_insights:
            community_context = f"\n\n{community_insights}"
            community_context += "\nWhen selecting items, consider the themes, representative items, and pairing wisdom identified in these community insights to make more contextually relevant and high-quality recommendations, especially for 'luxury' and 'impressive presentation' aspects."
        
        
        premium_queries = {
            "starter": f"""
    You are a luxury catering optimization consultant for {event_type.lower()} events.

    STARTER CATEGORY PREMIUM OPTIMIZATION:
    Budget Flexibility: ₹{budget_range['min']}-{budget_range['max']} (baseline ₹{budget_range['baseline']})
    Required: {count} starter items

    AVAILABLE INVENTORY WITH EXACT PRICING:
    {json.dumps(category_inventory, indent=2)}
    {community_context}

    PREMIUM OPTIMIZATION METHODOLOGY:
    1. Select baseline {count}-item combination around ₹{budget_range['baseline']}
    2. Engineer ONE strategic 2-item premium upgrade (target ₹{budget_range['max']-20}-{budget_range['max']} range)
    3. Prioritize: Premium proteins, gourmet preparation, impressive presentation

    OUTPUT FORMAT:
    **Baseline Selection ({count} items):**
    - [List exact item names from inventory]
    - Total: ₹XXX

    **Premium Upgrade Option:**
    - Replace: [Item A] + [Item B] → [Premium Item C] + [Premium Item D]
    - Investment: +₹XX for [specific luxury benefit]
    - Impact: [One line premium value justification]
    - Result: [One line party experience elevation]

    Focus on items that create memorable {event_type.lower()} experiences through luxury elevation.
    {json_output_instruction}
    """,

            "main_biryani": f"""
    You are optimizing premium main courses for ₹{budget_range['baseline']} {event_type.lower()} event budget.

    MAIN COURSE PREMIUM ENGINEERING:
    Budget Flexibility: ₹{budget_range['min']}-{budget_range['max']}
    Required: {count} biryani dish with luxury focus

    BIRYANI INVENTORY WITH PRICING:
    {json.dumps(category_inventory, indent=2)}
    {community_context}

    PREMIUM SELECTION CRITERIA:
    1. Identify baseline biryani around ₹{budget_range['baseline']}
    2. Show premium upgrade path (+₹30-50 investment)
    3. Focus on: Luxury ingredients, complex preparation, premium presentation

    OUTPUT FORMAT:
    **Baseline Selection:**
    - Biryani: [Exact item name] (₹XX)

    **Premium Upgrade Path:**
    - Replace: [Current Item] → [Premium Item]
    - Investment: +₹XX for [luxury upgrade type]
    - Value: [Premium ingredient/preparation benefit]
    - Experience: [Party impact enhancement]

    Select biryani that maximizes {event_type.lower()} event luxury within budget flexibility.
    {json_output_instruction}
    """,

            "main_rice": f"""
    Premium rice dish optimization for {event_type.lower()} events.

    RICE CATEGORY LUXURY FOCUS:
    Budget Range: ₹{budget_range['min']}-{budget_range['max']}
    Required: {count} flavored rice/pulav with premium positioning

    RICE INVENTORY:
    {json.dumps(category_inventory, indent=2)}
    {community_context}
    OPTIMIZATION APPROACH:
    1. Select baseline rice dish around ₹{budget_range['baseline']}
    2. Identify premium alternative (+₹20-40 upgrade)
    3. Emphasize: Exotic ingredients, aromatic preparation, visual appeal

    OUTPUT FORMAT:
    **Baseline Choice:** [Rice dish name] (₹XX)
    **Premium Option:** [Luxury rice dish] (+₹XX for [upgrade benefit])
    **Justification:** [Two lines explaining premium value]

    Choose rice that complements luxury {event_type.lower()} experience.
    {json_output_instruction}
    """,

            "side_bread": f"""
    Premium bread selection for {event_type.lower()} luxury dining.

    BREAD OPTIMIZATION:
    Budget: ₹{budget_range['min']}-{budget_range['max']}
    Required: {count} bread item with premium focus

    BREAD INVENTORY:
    {json.dumps(category_inventory, indent=2)}
    {community_context}
    SELECTION METHODOLOGY:
    1. Select baseline bread around ₹{budget_range['baseline']}
    2. Premium upgrade option (+₹15-30)
    3. Focus: Artisanal preparation, texture variety, visual presentation

    OUTPUT: [Bread name] (₹XX) with optional upgrade to [Premium bread] (+₹XX for [benefit])
    {json_output_instruction}
    """,

            "side_curry": f"""
    Luxury curry optimization for {event_type.lower()} events.

    CURRY PREMIUM SELECTION:
    Budget Range: ₹{budget_range['min']}-{budget_range['max']}
    Required: {count} curry with gourmet positioning

    CURRY INVENTORY:
    {json.dumps(category_inventory, indent=2)}
    {community_context}
    METHODOLOGY:
    1. Select baseline curry ₹{budget_range['baseline']}
    2. Show premium upgrade (+₹25-45)
    3. Prioritize: Complex spice profiles, premium ingredients, rich preparation

    OUTPUT: Baseline + Premium upgrade option with 2-line justification.
    {json_output_instruction}
    """,

            "side_accompaniment": f"""
    Premium accompaniment curation for {event_type.lower()} events.

    ACCOMPANIMENT LUXURY FOCUS:
    Budget: ₹{budget_range['min']}-{budget_range['max']}
    Required: {count} accompaniment items

    INVENTORY:
    {json.dumps(category_inventory, indent=2)}
    {community_context}
    Select accompaniments that elevate the dining experience with premium positioning.
    {json_output_instruction}
    """,

            "dessert": f"""
    Luxury dessert curation for memorable {event_type.lower()} event endings.

    DESSERT PREMIUM OPTIMIZATION:
    Budget Flexibility: ₹{budget_range['min']}-{budget_range['max']}
    Required: {count} dessert items for spectacular conclusion

    LUXURY DESSERT INVENTORY:
    {json.dumps(category_inventory, indent=2)}
    {community_context}
    PREMIUM CURATION METHODOLOGY:
    1. Select baseline {count}-dessert combination around ₹{budget_range['baseline']}
    2. Engineer ONE strategic 2-item luxury upgrade (target ₹{budget_range['max']-20}-{budget_range['max']})
    3. Focus: Presentation elegance, exotic ingredients, memorable experience

    OUTPUT FORMAT:
    **Baseline Selection:**
    - [Dessert items with pricing]
    - Total: ₹XXX

    **Luxury Upgrade Option:**
    - Replace: [Current Desserts] → [Premium Desserts]
    - Investment: +₹XX for [luxury enhancement]
    - Elegance: [Presentation/ingredient upgrade]
    - Memory: [Party ending experience enhancement]

    Create dessert combinations that leave lasting impressions.
    {json_output_instruction}
    """
        }
        
        query = premium_queries.get(category, f"Recommend {count} premium {category} items for {event_type.lower()} events with luxury focus and budget flexibility ₹{budget_range['min']}-{budget_range['max']}\n{json_output_instruction}")
        
        logger.debug(f"Built premium query for {category}: {query[:100]}...")
        return query

    def _intelligent_fuzzy_match(self, extracted_name: str, known_items: Dict[str, Any], category: str) -> Optional[Dict[str, Any]]:
        """
        INTELLIGENT FUZZY MATCHING: Word overlap with culinary intelligence
        """
        best_match = None
        best_score = 0.0
        
        extracted_words = set(extracted_name.lower().split())
        
        # Culinary keywords get bonus points
        culinary_keywords = {
            'biryani', 'rice', 'curry', 'chicken', 'paneer', 'dal', 'naan', 'roti', 
            'tikka', 'masala', 'fry', 'dum', 'butter', 'garlic', 'mint', 'jeera',
            'paan', 'jamun', 'gulab', 'cake', 'halwa', 'kesari'
        }
        
        for known_name, item_data in known_items.items():
            if not self._is_category_appropriate(item_data["category"], category):
                continue
            
            known_words = set(known_name.split())
            
            # Calculate base similarity
            intersection = extracted_words.intersection(known_words)
            union = extracted_words.union(known_words)
            
            if not union:
                continue
            
            jaccard_score = len(intersection) / len(union)
            
            # Bonus for substring containment
            substring_bonus = 0.3 if (extracted_name.lower() in known_name or known_name in extracted_name.lower()) else 0
            
            # Bonus for culinary keyword matches
            keyword_bonus = 0.2 * len(intersection.intersection(culinary_keywords))
            
            # Penalty for excessive length mismatch
            length_penalty = 0.1 if abs(len(extracted_words) - len(known_words)) > 2 else 0
            
            total_score = jaccard_score + substring_bonus + keyword_bonus - length_penalty
            
            if total_score > best_score and total_score > 0.6:
                best_score = total_score
                best_match = item_data
        
        return best_match
    # In graphrag_query_engine.py, inside GraphRAGTemplateEngine class
    # Add this after your existing _build_category_query_text method

    def _diagnose_baseline_selection(self, category: str, budget: int, event_type: str):
        """DIAGNOSTIC: Trace how baseline items are selected"""
        
        inventory = self._load_pricing_inventory()
        category_items = self._filter_inventory_by_category(inventory, category)
        
        # Analyze what's available in the budget range
        budget_analysis = {}
        for item_name, item_data in category_items.items():
            price = item_data['cmp_base_price']
            distance_from_budget = abs(price - budget)
            
            budget_analysis[item_name] = {
                'price': price,
                'budget_distance': distance_from_budget,
                'within_baseline_range': (budget - 50) <= price <= (budget + 50),
                'is_budget_optimal': distance_from_budget <= 20
            }
        
        # Sort by budget proximity
        sorted_options = sorted(budget_analysis.items(), 
                            key=lambda x: x[1]['budget_distance'])
        
        print(f"🎯 BASELINE SELECTION ANALYSIS for {category} (₹{budget} budget):")
        print(f"📊 Total items in category: {len(category_items)}")
        print(f"📊 Items within baseline range: {sum(1 for x in budget_analysis.values() if x['within_baseline_range'])}")
        
        print(f"\n🏆 TOP 5 BUDGET-OPTIMAL CANDIDATES:")
        for item_name, analysis in sorted_options[:5]:
            status = "✅ OPTIMAL" if analysis['is_budget_optimal'] else "⚠️  DISTANT"
            print(f"   {status} {item_name}: ₹{analysis['price']} (distance: ₹{analysis['budget_distance']})")
        
        return sorted_options

    def _is_item_match(self, extracted_name: str, known_name: str, category: str) -> bool:
        """Check if extracted name matches a known item name"""
        # Simple similarity check
        extracted_words = set(extracted_name.split())
        known_words = set(known_name.split())
        
        # Calculate word overlap
        overlap = len(extracted_words.intersection(known_words))
        min_words = min(len(extracted_words), len(known_words))
        
        return min_words > 0 and overlap / min_words >= 0.5

    def _is_category_appropriate(self, item_category: str, query_category: str) -> bool:
        """Check if item category is appropriate for query category"""
        category_mappings = {
            "starter": ["Starters", "Snacks"],
            "main_biryani": ["Main Course"],
            "main_rice": ["Main Course"],
            "side_bread": ["Main Course"],
            "side_curry": ["Main Course"],
            "side_accompaniment": ["Sides & Accompaniments"],
            "dessert": ["Desserts", "Sweets"]
        }
        
        appropriate_categories = category_mappings.get(query_category, [])
        return item_category in appropriate_categories

    def _calculate_match_confidence(self, extracted: str, known: str) -> float:
        """Calculate confidence score for item name match"""
        extracted_words = set(extracted.lower().split())
        known_words = set(known.lower().split())
        
        if not extracted_words or not known_words:
            return 0.0
        
        intersection = extracted_words.intersection(known_words)
        union = extracted_words.union(known_words)
        
        return len(intersection) / len(union) if union else 0.0

    def _enhance_with_co_occurrence_data(self, items: List[Dict[str, Any]], event_type: str) -> List[Dict[str, Any]]:
        """Enhance item suggestions with co-occurrence insights"""
        enhanced_items = []
        
        for item in items:
            enhanced_item = item.copy()
            
            # Generate insight based on item and event type
            insight = self._generate_item_insight(item["name"], event_type)
            enhanced_item["insight"] = insight
            enhanced_item["co_occurrence_score"] = 0.8  # Default score
            
            enhanced_items.append(enhanced_item)
        
        return enhanced_items

    def _generate_item_insight(self, item_name: str, event_type: str) -> str:
        """Generate condensed insight for an item"""
        insight_templates = {
            "Traditional": {
                "default": f"Classic choice for traditional events",
                "biryani": f"Traditional centerpiece with proven success",
                "curry": f"Authentic flavor profile for traditional settings"
            },
            "Party": {
                "default": f"Popular party selection",
                "starter": f"Engaging party appetizer",
                "fusion": f"Modern party favorite"
            }
        }
        
        event_templates = insight_templates.get(event_type, insight_templates["Traditional"])
        
        # Simple keyword-based insight selection
        item_lower = item_name.lower()
        if "biryani" in item_lower:
            return event_templates.get("biryani", event_templates["default"])
        elif "curry" in item_lower:
            return event_templates.get("curry", event_templates["default"])
        elif any(word in item_lower for word in ["65", "tikka", "manchurian"]):
            return event_templates.get("starter", event_templates["default"])
        else:
            return event_templates["default"]

    def _fill_template_with_suggestions(self, template: Dict[str, Any], requirements: Dict[str, Any], suggestions: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Fill template structure with GraphRAG suggestions"""
        filled_template = template.copy()
        filled_items = []
        
        # Process each category in original template order
        for original_item in template["items"]:
            category = original_item["category"]
            
            # Map to GraphRAG category
            graphrag_category = self.template_bridge.category_mapper.map_to_graphrag_category(category, original_item)
            
            # Get suggestion for this item slot
            category_suggestions = suggestions.get(graphrag_category, [])
            
            if category_suggestions:
                # Take next available suggestion
                suggestion = category_suggestions.pop(0)
                
                filled_item = {
                    "category": original_item["category"],
                    "name": suggestion["name"],
                    "weight": original_item.get("weight"),
                    #"quantity": original_item.get("quantity"),
                    # Use the quantity from the LLM's suggestion, not the template
                    "quantity": suggestion.get("quantity", original_item.get("quantity")),
                    "insight": suggestion.get("insight", ""),
                    "graphrag_metadata": {
                        "suggested_category": graphrag_category,
                        "source": suggestion.get("source", "graphrag"),
                        "confidence": suggestion.get("match_confidence", 0.0)
                    }
                }
            else:
                # Fallback to placeholder
                filled_item = original_item.copy()
                filled_item["insight"] = "Item selection pending"
                filled_item["graphrag_metadata"] = {"source": "placeholder"}
            
            filled_items.append(filled_item)
        
        filled_template["items"] = filled_items
        return filled_template

    def _count_fallbacks(self, suggestions: Dict[str, List[Dict[str, Any]]]) -> int:
        """Count how many fallback suggestions were used"""
        fallback_count = 0
        for category_suggestions in suggestions.values():
            for suggestion in category_suggestions:
                if suggestion.get("source") in ["fallback", "community_default"]:
                    fallback_count += 1
        return fallback_count
    
    def run_diagnostics(self):
        """
        Run diagnostics on the current GraphRAG system
        """
        print(f"\n🔧 Running diagnostics for GraphRAGTemplateEngine...")
        
        # Check if GraphRAG system is initialized
        if not self.graphrag_adapter or not self.graphrag_query_engine:
            print("❌ GraphRAG system not properly initialized")
            return False
        
        # Run the external diagnostic function
        return diagnose_community_system(self.graphrag_adapter, self.graphrag_query_engine)


class InsightGenerator:
    """
    Generates condensed insights for template recommendations
    """
    
    def generate_template_insights(self, filled_template: Dict[str, Any], suggestions: Dict[str, List[Dict[str, Any]]], event_type: str) -> Dict[str, Any]:
        """
        Generate comprehensive insights for template recommendation
        
        Args:
            filled_template: Template filled with specific items
            suggestions: Original GraphRAG suggestions
            event_type: Event type context
        
        Returns:
            Insight dictionary with individual and overall insights
        """
        insights = {
            "individual_insights": {},
            "overall_insight": "",
            "success_indicators": {},
            "recommendation_strength": ""
        }
        
        # Generate individual item insights
        for item in filled_template["items"]:
            if "insight" in item and item["insight"]:
                insights["individual_insights"][item["name"]] = item["insight"]
        
        # Generate overall insight
        insights["overall_insight"] = self._generate_overall_insight(filled_template, event_type)
        
        # Calculate success indicators
        insights["success_indicators"] = self._calculate_success_indicators(suggestions)
        
        # Determine recommendation strength
        insights["recommendation_strength"] = self._determine_recommendation_strength(suggestions)
        
        return insights
    
    def _generate_overall_insight(self, filled_template: Dict[str, Any], event_type: str) -> str:
        """Generate overall combination insight"""
        template_name = filled_template["name"]
        event_lower = event_type.lower()
        
        # Template-specific insights
        if "starter" in template_name.lower():
            if "party" in template_name.lower():
                return f"Variety-focused {event_lower} combination with engagement-driven starter selection. Proven success pattern."
            else:
                return f"Balanced {event_lower} approach with starter variety and traditional foundation."
        elif "biryani" in template_name.lower():
            return f"Biryani-centered {event_lower} experience with complementary accompaniments. High satisfaction pattern."
        else:
            return f"Well-structured {event_lower} combination based on successful menu patterns."
    
    def _calculate_success_indicators(self, suggestions: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Calculate success indicators from suggestions"""
        total_suggestions = sum(len(items) for items in suggestions.values())
        fallback_count = 0
        high_confidence_count = 0
        
        for category_suggestions in suggestions.values():
            for suggestion in category_suggestions:
                if suggestion.get("source") in ["fallback", "community_default"]:
                    fallback_count += 1
                if suggestion.get("match_confidence", 0) > 0.8:
                    high_confidence_count += 1
        
        success_rate = ((total_suggestions - fallback_count) / total_suggestions * 100) if total_suggestions > 0 else 0
        confidence_rate = (high_confidence_count / total_suggestions * 100) if total_suggestions > 0 else 0
        
        return {
            "estimated_success_rate": f"{success_rate:.0f}%",
            "high_confidence_items": f"{confidence_rate:.0f}%",
            "fallback_items_used": fallback_count
        }
    
    def _determine_recommendation_strength(self, suggestions: Dict[str, List[Dict[str, Any]]]) -> str:
        """Determine overall recommendation strength"""
        total_items = sum(len(items) for items in suggestions.values())
        fallback_items = sum(1 for items in suggestions.values() 
                           for item in items 
                           if item.get("source") in ["fallback", "community_default"])
        
        fallback_ratio = fallback_items / total_items if total_items > 0 else 1
        
        if fallback_ratio <= 0.2:
            return "Strong"
        elif fallback_ratio <= 0.5:
            return "Good"
        else:
            return "Fair"


# ============================================================================
# MAIN INTERFACE FUNCTIONS
# ============================================================================

def create_graphrag_template_engine(template_file: str = None, run_diagnostics: bool= False) -> GraphRAGTemplateEngine:
    """
    Factory function to create GraphRAG Template Engine
    
    Args:
        template_file: Optional custom template file path
    
    Returns:
        Configured GraphRAGTemplateEngine instance
    """
    engine = GraphRAGTemplateEngine(template_file)
    
    if run_diagnostics:
        print("🔍 Running post-creation diagnostics...")
        engine.run_diagnostics()
    return engine

def reset_graphrag_cache():
    """
    Reset both in-memory and on-disk cache for a clean rebuild.
    """
    global _GRAPHRAG_SYSTEM_CACHE
    
    # 1. Reset in-memory cache
    _GRAPHRAG_SYSTEM_CACHE = {'adapter': None, 'query_engine': None, 'initialized': False}
    logger.info("🔄 In-memory GraphRAG cache has been reset.")

    # 2. Clear on-disk cache
    try:
        cache_dir = get_cache_directory()
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            logger.info(f"🗑️ On-disk cache directory '{cache_dir}' has been deleted.")
        else:
            logger.info("ℹ️ On-disk cache directory not found, nothing to delete.")
    except Exception as e:
        logger.error(f"❌ Failed to delete on-disk cache: {e}")
# ============================================================================
# TESTING
# ============================================================================
# Add this to graphrag_query_engine.py for debugging

def diagnose_community_system(adapter, query_engine):
    """
    COMPREHENSIVE DIAGNOSTICS: Identify exactly what's wrong
    """
    print("\n🔬 COMMUNITY SYSTEM DIAGNOSTICS")
    print("=" * 60)
    
    # 1. Check Neo4j connection
    try:
        test_query = "MATCH (n) RETURN count(n) as node_count LIMIT 1"
        result = adapter.neo4j_store.structured_query(test_query)
        node_count = result[0]['node_count'] if result else 0
        print(f"✅ Neo4j Connection: {node_count} nodes available")
    except Exception as e:
        print(f"❌ Neo4j Connection: FAILED - {e}")
        return False
    
    # 2. Check community summaries
    summaries = adapter.graphrag_store.get_community_summaries()
    print(f"📊 Community Summaries: {len(summaries)} found")
    
    if len(summaries) == 0:
        print("⚠️  NO COMMUNITIES DETECTED - This is the root cause!")
        
        # Check raw graph data
        try:
            nodes = len(adapter.graphrag_store.graph.nodes)
            relations = len(adapter.graphrag_store.graph.relations)
            print(f"📈 Raw Graph Data: {nodes} nodes, {relations} relations")
            
            if nodes == 0:
                print("❌ CRITICAL: No nodes in graph - Neo4j sync failed")
            elif relations == 0:
                print("❌ CRITICAL: No relations in graph - relationships not synced")
            else:
                print("💡 Graph data exists but community detection failed")
                
        except Exception as e:
            print(f"❌ Graph data check failed: {e}")
        
        return False
    
    # 3. Sample community content
    print("\n🔍 COMMUNITY CONTENT SAMPLE:")
    for i, (comm_id, summary) in enumerate(list(summaries.items())[:3]):
        print(f"\nCommunity {comm_id}:")
        print(f"  Summary length: {len(summary)} characters")
        print(f"  Preview: {summary[:100]}...")
    
    # 4. Check entity mappings
    entity_communities = adapter.graphrag_store.entity_communities
    print(f"\n🏷️  Entity Mappings: {len(entity_communities)} entities mapped")
    
    if len(entity_communities) == 0:
        print("❌ CRITICAL: No entity-community mappings")
        return False
    
    # Sample entity mappings
    print("\n📋 SAMPLE ENTITY MAPPINGS:")
    for i, (entity, communities) in enumerate(list(entity_communities.items())[:5]):
        print(f"  {entity} → Communities: {communities}")
    
    # 5. Test query functionality
    print("\n🎯 TESTING QUERY FUNCTIONALITY:")
    test_queries = [
        ("biryani", "Party"),
        ("starter", "Traditional"), 
        ("dessert", "Party")
    ]
    
    for category, event_type in test_queries:
        try:
            print(f"\n  Testing: {category} for {event_type} events...")
            
            # Build test query
            query_text = f"What are good {category} options for {event_type.lower()} events?"
            
            # Execute query
            response = query_engine.query(query_text)
            
            if response and len(response) > 50:
                print(f"    ✅ Response generated ({len(response)} chars)")
                print(f"    Preview: {response[:100]}...")
            else:
                print(f"    ❌ Poor response: {response[:50]}...")
                
        except Exception as e:
            print(f"    ❌ Query failed: {e}")
    
    print(f"\n🎯 DIAGNOSIS COMPLETE")
    return True

def test_item_extraction():
    """
    TEST ITEM EXTRACTION: Check if extraction patterns work
    """
    print("\n🧪 TESTING ITEM EXTRACTION PATTERNS")
    print("=" * 50)
    
    # Sample GraphRAG responses to test against
    test_responses = [
        "I recommend **Chicken 65**, **Paneer Tikka**, and **Veg Spring Rolls** for party events.",
        "For traditional events, consider: Chicken Biryani, Mutton Curry, and Dal Tadka.",
        "The best starters are: 1. Chicken Manchurian 2. Cheese Balls 3. Fish Fry",
        "Excellent dessert choices include Gulab Jamun, Double Ka Meetha, and Tiramisu.",
    ]
    
    # Load pricing data for validation
    try:
        with open("items_price_uom.json", 'r', encoding='utf-8') as f:
            pricing_data = json.load(f)
            known_items = {item["item_name"].lower(): item for item in pricing_data}
        print(f"📚 Loaded {len(known_items)} known items for validation")
    except Exception as e:
        print(f"❌ Could not load pricing data: {e}")
        return
    
    # Test each response
    for i, response in enumerate(test_responses, 1):
        print(f"\n🔍 Test {i}: {response[:50]}...")
        
        # Test extraction (you'll need to import the actual extraction function)
        try:
            # This would call your actual extraction function
            # extracted = _extract_items_from_graphrag_response(response, "starter", 3)
            
            # For now, let's do basic pattern matching
            import re
            patterns = [
                r'\*\*([^*]+?)\*\*',  # Bold items
                r'(?:recommend|suggest|include|consider)[^:]*:\s*([^.]+)',  # Recommendation phrases
                r'\d+\.\s*([^\n\d]+)',  # Numbered lists
            ]
            
            found_items = []
            for pattern in patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                for match in matches:
                    clean_match = re.sub(r'[^\w\s]', '', match.strip())
                    if clean_match.lower() in known_items:
                        found_items.append(clean_match)
            
            print(f"    ✅ Extracted {len(found_items)} valid items: {found_items}")
            
        except Exception as e:
            print(f"    ❌ Extraction failed: {e}")

# Add this function call in your main app to run diagnostics
def run_full_diagnostics():
    """Run complete system diagnostics"""
    try:
        adapter, query_engine = get_cached_graphrag_system()
        
        # Run diagnostics
        system_ok = diagnose_community_system(adapter, query_engine)
        
        if system_ok:
            test_item_extraction()
            print("\n🎉 System appears to be working correctly!")
        else:
            print("\n🚨 CRITICAL ISSUES FOUND - Community system needs repair")
            
        return system_ok
        
    except Exception as e:
        print(f"❌ Diagnostics failed: {e}")
        import traceback
        traceback.print_exc()
        return False
if __name__ == "__main__":
    print("Testing GraphRAG Template Engine...")

    # ==> STEP 1: Uncomment the line below to clear your cache.
    #reset_graphrag_cache()
    # ==> STEP 2: Run the script once with the line above uncommented.
    # ==> STEP 3: Comment the line out again for normal operation.
    
    # Set to True if you need to run diagnostics, otherwise False.
    run_diagnostics_flag = False
    
    try: 
        # Create engine
        engine = create_graphrag_template_engine(run_diagnostics=run_diagnostics_flag)
        print("✓ GraphRAG Template Engine created successfully")
        
        # Test recommendation
        test_event = "Party"
        test_budget = 300
        
        print(f"\nTesting recommendation for {test_event} event with ₹{test_budget} budget...")
        
        recommendation = engine.recommend_with_graphrag(test_event, test_budget)
        
        if "error" not in recommendation:
            print(f"✓ Generated recommendation: {recommendation['template_name']}")
            print(f"✓ Items: {len(recommendation['items'])}")
            print(f"✓ Insights: {recommendation['insights']['recommendation_strength']} strength")
            
            # Display sample items
            print("\nSample items:")
            for i, item in enumerate(recommendation['items'][:3]):
                print(f"  {i+1}. {item['name']} ({item['category']}) - {item.get('insight', 'No insight')}")
        else:
            print(f"✗ Recommendation failed: {recommendation['error']}")
            
    except Exception as e:
        print(f"✗ Testing failed: {e}")
        import traceback
        traceback.print_exc()