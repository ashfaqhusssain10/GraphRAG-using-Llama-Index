import re
import networkx as nx
from graspologic.partition import hierarchical_leiden
from collections import defaultdict
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode, Relation, LabelledNode
from llama_index.core.llms import ChatMessage
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import google.generativeai as genai
import torch
from typing import Dict, List, Any
from dotenv import load_dotenv
load_dotenv()
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import logging
logger = logging.getLogger(__name__)

class FoodServiceGraphRAGStore(SimplePropertyGraphStore):
    """
    A specialized GraphRAG store designed for food service data that implements
    community detection to identify natural groupings of ingredients, dishes, and
    culinary concepts. This approach discovers hidden patterns in your menu data
    and creates expert-level summaries for each thematic cluster.
    
    Think of this as having a team of food scientists analyze your entire menu
    system and create detailed reports about different culinary themes, which
    can then be used to provide much richer answers to complex food-related queries.
    """
    
    def __init__(self, llm=None, max_cluster_size=64,min_cluster_size=3):
        """
        Initialize the custom GraphRAG store with community detection capabilities.
        
        Args:
            llm: The language model to use for generating community summaries.
                 If None, will use the model from Settings.llm
            max_cluster_size: Maximum size for communities. Smaller values create
                            more focused, specialized communities. Larger values
                            create broader, more general communities.
        """
        super().__init__()
        
        # Store community summaries - this is where our "expert reports" live
        self.community_summaries = {}
        
        # Store community membership information for quick lookup
        self.entity_communities = defaultdict(set)
        
        # Configuration for community detection algorithm
        self.max_cluster_size = max_cluster_size
        self.min_cluster_size=min_cluster_size
        # Use provided LLM or fall back to global settings
        self.llm = llm if llm is not None else Settings.llm
        
        if self.llm is None:
            raise ValueError("No LLM provided and Settings.llm is not configured")
    
    def generate_community_summary(self, community_relationships: List[str]) -> str:
        """
        Generate a comprehensive summary for a community using our local LLM.
        
        This method takes the raw relationship data from a community and asks
        our language model to analyze it like a food expert would, identifying
        the common themes, culinary principles, and practical insights that
        define this particular grouping.
        
        Args:
            community_relationships: List of relationship descriptions in the format
                                   "entity1 -> entity2 -> relation -> description"
        
        Returns:
            A comprehensive summary explaining what makes this community cohesive
        """
        
        # Prepare the context for our food domain expert prompt
        relationships_text = "\n".join(community_relationships)
        
        # Create a specialized prompt that encourages the LLM to think like
        # a culinary expert analyzing ingredient and dish relationships
        system_prompt = (
    "You are a culinary expert and food service consultant analyzing relationships "
    "between dishes, ingredients, menu categories, and dining contexts. You are "
    "provided with a set of relationships from a food service knowledge graph that "
    "includes both categorical relationships (what dishes contain, what categories "
    "they belong to) and historical co-occurrence patterns (what dishes have actually "
    "been served successfully together in real operations).\n\n"
    
    "Your task is to create a comprehensive summary that explains:\n"
    "1. What culinary theme or concept unifies these relationships\n"
    "2. The practical implications for menu planning and dish pairing\n"
    "3. Key ingredients, techniques, or dining contexts that define this group\n"
    "4. Historical pairing patterns and what they reveal about successful combinations\n"
    "5. How these elements work together to create cohesive food experiences\n"
    "6. Specific recommendations for menu planners based on proven co-occurrence data\n\n"
    "7. A list of the all representative items (dishes or ingredients) from this community.\n"
    "Pay special attention to co-occurrence relationships - these represent actual "
    "historical evidence of successful dish pairings from real menu service. When you "
    "see dishes that have co-occurred multiple times, emphasize these as validated "
    "combinations that have proven successful in practice.\n\n"
    
    "Focus on actionable insights that would help a chef, menu planner, or "
    "food service manager understand both the theoretical relationships and the "
    "practical pairing wisdom embedded in this data."
)
        
        user_prompt = f"Analyze these food service relationships:\n\n{relationships_text}"
        
        # Use our local LLM to generate the expert summary
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ]
        
        try:
            response = self.llm.chat(messages)
            # Clean up the response by removing any model artifacts
            clean_summary = re.sub(r"^assistant:\s*", "", str(response)).strip()
            return clean_summary
        except Exception as e:
            # Fallback to a basic summary if LLM generation fails
            return f"Community containing {len(community_relationships)} relationships between food service entities."
    
    def build_communities(self):
        """
        The main community detection method that transforms your food service graph
        into a collection of thematically coherent communities with expert summaries.
        
        This process works in several stages:
        1. Convert our property graph into NetworkX format for analysis
        2. Apply hierarchical Leiden clustering to identify natural groupings
        3. Extract relationship information for each community
        4. Generate expert summaries for each community using our LLM
        
        Think of this as hiring a team of food scientists to study your entire
        menu system and write detailed reports about different culinary themes.
        """
        
        print("🔍 Analyzing graph structure for community detection...")
        
        # Step 1: Convert our property graph to NetworkX format
        # NetworkX provides the mathematical tools for community detection
        nx_graph = self._create_networkx_graph()
        
        if len(nx_graph.nodes()) == 0:
            print("⚠️  Warning: Graph is empty, no communities to build")
            return
        
        print(f"📊 Graph contains {len(nx_graph.nodes())} nodes and {len(nx_graph.edges())} edges")
        
        # Step 2: Apply hierarchical Leiden clustering
        # This algorithm is particularly good at finding communities in complex networks
        # like food service systems where relationships exist at multiple scales
        print("🧮 Running hierarchical Leiden clustering algorithm...")
        
        try:
            community_clusters = hierarchical_leiden(
                nx_graph, 
                max_cluster_size=self.max_cluster_size
            )
            
            print(f"✓ Identified {len(set(cluster.cluster for cluster in community_clusters))} communities")
            
        except Exception as e:
            print(f"❌ Community detection failed: {e}")
            return
        
        # Step 3: Organize community information
        # We need to understand which entities belong to which communities
        # and what relationships define each community
        print("📋 Organizing community information...")
        entity_communities, community_relationships = self._organize_community_data(
            nx_graph, community_clusters
        )
        # NEW: Filter communities based on size criteria
        print("🔍 Filtering communities by size criteria...")
        filtered_communities = self._filter_communities_by_size(
            entity_communities, community_relationships
        )
        entity_communities, community_relationships = filtered_communities

        print(f"✓ Retained {len(community_relationships)} communities after size filtering")
        
        # Store entity community memberships for quick lookup during queries
        self.entity_communities = entity_communities
        
        # Step 4: Generate expert summaries for each community
        # This is where we transform raw relationship data into actionable insights
        print("📝 Generating expert summaries for each community...")
        
        for community_id, relationships in community_relationships.items():
            if len(relationships) > 0:  # Only summarize non-empty communities
                print(f"  Analyzing community {community_id} ({len(relationships)} relationships)...")
                
                try:
                    summary = self.generate_community_summary(relationships)
                    self.community_summaries[community_id] = summary
                    if "biryani" in summary.lower() or "dessert" in summary.lower():  # <-- ADD THIS
                     print(f"    🎯 RELEVANT SUMMARY: {summary[:200]}...")
                    print(f"    ✓ Generated summary ({len(summary)} characters)")
                    
                except Exception as e:
                    print(f"    ❌ Failed to generate summary for community {community_id}: {e}")
                    # Store a basic fallback summary
                    self.community_summaries[community_id] = f"Community with {len(relationships)} food service relationships"
        
        print(f"🎉 Community building complete! Generated {len(self.community_summaries)} expert summaries")
    
    def _create_networkx_graph(self) -> nx.Graph:
        """
        Convert our LlamaIndex property graph into a NetworkX graph suitable for analysis.
        
        This conversion preserves the essential relationship structure while adding
        the metadata needed for community detection algorithms. We focus on the
        most meaningful relationships for food service analysis.
        
        Returns:
            A NetworkX graph ready for community detection
        """
        
        nx_graph = nx.Graph()
        
        # Add all nodes from our property graph
        # Each node represents a dish, ingredient, category, or other food service entity
        for node_id, node in self.graph.nodes.items():
            nx_graph.add_node(node_id, **node.properties)
        
        # Add edges representing meaningful relationships
        # We include relationship metadata to help with community analysis
        for relation_id, relation in self.graph.relations.items():
            
            # Extract relationship description for community analysis
            # This helps the algorithm understand what makes entities related
            relationship_desc = relation.properties.get('relationship_description', 
                                                       f"{relation.label} relationship")
            
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relationship_desc
            )
        
        return nx_graph
    
    def _organize_community_data(self, nx_graph: nx.Graph, clusters) -> tuple:
        """
        Process the raw clustering results into organized data structures.
        
        This method transforms the mathematical output of the clustering algorithm
        into practical data structures that we can use for query processing and
        summary generation.
        
        Args:
            nx_graph: The NetworkX graph used for clustering
            clusters: Raw clustering results from hierarchical_leiden
        
        Returns:
            Tuple of (entity_communities, community_relationships) where:
            - entity_communities maps each entity to its community IDs
            - community_relationships maps each community to its relationship descriptions
        """
        
        entity_communities = defaultdict(set)
        community_relationships = defaultdict(list)
        
        # Process each cluster assignment
        for cluster_info in clusters:
            node = cluster_info.node
            community_id = cluster_info.cluster
            
            # Track which communities each entity belongs to
            entity_communities[node].add(community_id)
            
            # Collect relationship information for each community
            # This helps us understand what defines each community thematically
            for neighbor in nx_graph.neighbors(node):
                edge_data = nx_graph.get_edge_data(node, neighbor)
                
                if edge_data:
                    # Create a descriptive relationship string
                    relationship_desc = (
                        f"{node} -> {neighbor} -> "
                        f"{edge_data.get('relationship', 'related')} -> "
                        f"{edge_data.get('description', 'food service relationship')}"
                    )
                    
                    community_relationships[community_id].append(relationship_desc)
        
        # Convert defaultdicts to regular dicts and remove duplicates
        entity_communities = {k: list(v) for k, v in entity_communities.items()}
        community_relationships = {
            k: list(set(v)) for k, v in community_relationships.items()
        }
        
        return entity_communities, community_relationships
    
    ###
    def _filter_communities_by_size(self, entity_communities, community_relationships):
        """
        Filter communities to keep only those within our desired size range.
        
        This method removes communities that are too small to provide meaningful
        insights or too large to maintain thematic coherence. It's like curating
        a collection - we want groups that are substantial enough to be useful
        but focused enough to be comprehensible.
        
        Args:
            entity_communities: Mapping of entities to their community memberships
            community_relationships: Mapping of communities to their relationships
            
        Returns:
            Tuple of filtered (entity_communities, community_relationships)
        """
        
        filtered_entity_communities = {}
        filtered_community_relationships = {}
        
        communities_removed_too_small = 0
        #communities_removed_too_large = 0
        
        for community_id, relationships in community_relationships.items():
            community_size = len(relationships)
            
            # Check if community meets our size criteria
            if community_size < self.min_cluster_size:
                communities_removed_too_small += 1
                continue  # Skip this community - too small
                
            # if community_size > self.max_cluster_size:
            #     communities_removed_too_large += 1
            #     continue  # Skip this community - too large
                
            # Community meets criteria - keep it
            filtered_community_relationships[community_id] = relationships
        
        # Update entity mappings to only include entities from retained communities
        for entity, community_list in entity_communities.items():
            filtered_list = [cid for cid in community_list 
                            if cid in filtered_community_relationships]
            if filtered_list:  # Only keep entities that belong to at least one retained community
                filtered_entity_communities[entity] = filtered_list
        
        # Report filtering statistics
        total_original = len(community_relationships)
        total_retained = len(filtered_community_relationships)
        
        print(f"  📊 Community filtering results:")
        print(f"    • Original communities: {total_original}")
        print(f"    • Retained communities: {total_retained}")
        print(f"    • Removed (too small < {self.min_cluster_size}): {communities_removed_too_small}")
       # print(f"    • Removed (too large > {self.max_cluster_size}): {communities_removed_too_large}")
        
        return filtered_entity_communities, filtered_community_relationships
    
    
    def get_community_summaries(self) -> Dict[int, str]:
        """
        Access the generated community summaries.
        
        Returns:
            Dictionary mapping community IDs to their expert-generated summaries
        """
        return self.community_summaries
    
    def get_entity_communities(self, entity: str) -> List[int]:
        """
        Find which communities a specific entity belongs to.
        
        This is useful for query processing - when someone asks about a specific
        ingredient or dish, we can quickly identify which community summaries
        are most relevant.
        
        Args:
            entity: The entity name to look up
        
        Returns:
            List of community IDs that this entity belongs to
        """
        return list(self.entity_communities.get(entity, []))

class FoodServiceGraphRAGQueryEngine:
    """
    A specialized query engine that leverages community summaries to provide
    comprehensive, contextually rich answers about food service operations.
    
    Unlike traditional retrieval that searches for specific facts, this engine
    uses the community summaries as expert knowledge bases, combining insights
    from multiple relevant communities to construct nuanced responses.
    
    Think of this as having access to a team of food service consultants who
    have each specialized in different aspects of your operation (ingredients,
    menu planning, dietary restrictions, etc.) and can collaborate to answer
    complex questions about your food service system.
    """
    
    def __init__(self, graph_store: FoodServiceGraphRAGStore, llm=None, similarity_top_k=5):
        """
        Initialize the community-powered query engine.
        
        Args:
            graph_store: The GraphRAG store containing community summaries
            llm: Language model for final answer synthesis
            similarity_top_k: Number of most relevant entities to consider for each query
        """
        self.graph_store = graph_store
        self.llm = llm if llm is not None else Settings.llm
        self.similarity_top_k = similarity_top_k
        
        if self.llm is None:
            raise ValueError("No LLM provided and Settings.llm is not configured")
    
    def query(self, query_str: str) -> str:
        """
        Process a query using community-based analysis to provide comprehensive answers.
        
        This method works by:
        1. Identifying entities in the query that exist in our graph
        2. Finding which communities these entities belong to
        3. Gathering the expert summaries for relevant communities
        4. Synthesizing these insights into a comprehensive answer
        
        Args:
            query_str: The natural language question about your food service operation
        
        Returns:
            A comprehensive answer leveraging community insights
        """
        
        print(f"🔍 Processing query: '{query_str}'")
        
        # Step 1: Identify relevant entities from the query
        relevant_entities = self._extract_entities_from_query(query_str)
        print(f"📋 Identified entities: {relevant_entities}")
        
        if not relevant_entities:
            print("⚠️  No relevant entities found in query")
            return "I couldn't identify any specific food service entities in your query. Please try asking about specific dishes, ingredients, categories, or menu items."
        
        # Step 2: Find communities associated with these entities
        relevant_communities = self._get_relevant_communities(relevant_entities)
        print(f"🏘️  Found {len(relevant_communities)} relevant communities")
        
        if not relevant_communities:
            print("⚠️  No community summaries found for identified entities")
            return "I found the entities you mentioned, but they don't appear to be part of any analyzed communities yet."
        
        # Step 3: Gather community summaries
        community_summaries = self._gather_community_summaries(relevant_communities)
        
        # Step 4: Generate comprehensive answer using community insights
        final_answer = self._synthesize_answer(query_str, community_summaries, relevant_entities)
        
        return final_answer
    def _detect_pairing_query(self, query_str: str) -> bool:
     """
     Detect if the user is asking for pairing or recommendation advice.
     These queries should prioritize CO-OCCURS relationships in the analysis.
     """
     pairing_keywords = [
         'goes with', 'pairs with', 'goes well', 'complements', 'recommend', 
         'suggestion', 'what should i serve', 'what to pair', 'accompaniment',
         'side dish', 'together', 'combination', 'menu planning'
     ]
     
     query_lower = query_str.lower()
     return any(keyword in query_lower for keyword in pairing_keywords)

    
    def _extract_entities_from_query(self, query_str: str) -> List[str]:
        """
        Identify food service entities mentioned in the query.
        
        This method looks for entities that exist in our graph by checking
        against all known dishes, ingredients, categories, and other food service
        concepts. This approach ensures we're working with entities that have
        community associations.
        
        Args:
            query_str: The user's question
        
        Returns:
            List of entity names found in the query
        """
        
        query_lower = query_str.lower()
        found_entities = []
        is_pairing_query = self._detect_pairing_query(query_str)
        # Check all nodes in our graph to see if they're mentioned in the query
        for node_id, node in self.graph_store.graph.nodes.items():
            entity_name = node.properties.get('name', node_id)
            node_type = node.properties.get('node_type', 'unknown')
            # Look for the entity name in the query (case-insensitive)
            if entity_name.lower() in query_lower:
             if is_pairing_query and node_type == 'Dish':
              found_entities.insert(0, entity_name)
             else:
                found_entities.append(entity_name)
        
        # Also check for partial matches and common food service terms
        # This helps catch queries that use general terms like "dessert" or "breakfast"
        food_terms = {
            'dessert': ['dessert', 'sweet', 'cake', 'pastry'],
            'breakfast': ['breakfast', 'morning', 'brunch'],
            'lunch': ['lunch', 'midday'],
            'dinner': ['dinner', 'evening', 'supper'],
            'appetizer': ['appetizer', 'starter', 'app'],
            'main': ['main', 'entree', 'entrée'],
            'ingredient': ['ingredient', 'component']
        }
        
        for category, terms in food_terms.items():
            if any(term in query_lower for term in terms):
                # Look for entities in our graph that might match this category
                for node_id, node in self.graph_store.graph.nodes.items():
                    node_props = node.properties
                    if (category.lower() in str(node_props).lower() or 
                        any(term in str(node_props).lower() for term in terms)):
                        entity_name = node_props.get('name', node_id)
                        if entity_name not in found_entities:
                            found_entities.append(entity_name)
        
        return found_entities[:self.similarity_top_k]  # Limit to top matches
    
    def _get_relevant_communities(self, entities: List[str]) -> List[int]:
        """
        Find all communities that contain any of the relevant entities.
        
        This method identifies which expert summaries are most likely to contain
        insights relevant to the user's query.
        
        Args:
            entities: List of entity names to look up
        
        Returns:
            List of unique community IDs that contain these entities
        """
        
        relevant_communities = set()
        
        for entity in entities:
            entity_communities = self.graph_store.get_entity_communities(entity)
            relevant_communities.update(entity_communities)
        
        return list(relevant_communities)
    
    def _gather_community_summaries(self, community_ids: List[int]) -> Dict[int, str]:
        """
        Retrieve the expert summaries for relevant communities.
        
        Args:
            community_ids: List of community IDs to retrieve summaries for
        
        Returns:
            Dictionary mapping community IDs to their summaries
        """
        
        all_summaries = self.graph_store.get_community_summaries()
        relevant_summaries = {}
        
        for community_id in community_ids:
            if community_id in all_summaries:
                relevant_summaries[community_id] = all_summaries[community_id]
        
        return relevant_summaries
    
    def _synthesize_answer(self, query: str, community_summaries: Dict[int, str], entities: List[str]) -> str:
        """
        Combine insights from multiple community summaries to create a comprehensive answer.
        
        This is where the magic happens - we take the expert knowledge from relevant
        communities and synthesize it into a coherent response that addresses the
        user's specific question.
        
        Args:
            query: The original user question
            community_summaries: Expert summaries from relevant communities
            entities: The entities identified in the query
        
        Returns:
            A comprehensive, synthesized answer
        """
        
        if not community_summaries:
            return "I couldn't find relevant community information for your query."
        is_pairing_query = self._detect_pairing_query(query)
        
        # Prepare context for the synthesis
        summaries_text = "\n\n".join([
            f"Community {cid} Analysis:\n{summary}" 
            for cid, summary in community_summaries.items()
        ])
        
        entities_text = ", ".join(entities)
        if is_pairing_query:
        # Create a specialized prompt for food service answer synthesis
         system_prompt = (
             "You are a food service expert specializing in menu planning and dish pairing. "
             "The user is asking for pairing recommendations or menu planning advice. You have "
             "access to community analyses that include historical co-occurrence data - actual "
             "evidence of what dish combinations have been successfully served together.\n\n"
             
             "When you see mentions of dishes that have 'co-occurred' or 'appeared together' "
             "multiple times, treat these as high-confidence recommendations based on proven "
             "success in real food service operations. Emphasize these historical patterns "
             "and explain why they represent validated combinations.\n\n"
             
             "Provide specific, actionable pairing recommendations that prioritize combinations "
             "with strong historical evidence while also explaining the culinary principles "
             "that make these pairings successful."
         )
        else: 
         system_prompt = (
             "You are a food service expert tasked with answering questions about menu "
             "planning, ingredients, dishes, and culinary operations. You have access to "
             "detailed community analyses that provide expert insights about different "
             "aspects of the food service operation.\n\n"
             "Your goal is to synthesize information from these community analyses to "
             "provide a comprehensive, practical answer that directly addresses the user's "
             "question. Focus on actionable insights and specific details that would be "
             "valuable for food service professionals.\n\n"
             "If the community analyses don't directly answer the question, acknowledge "
             "this and provide the most relevant information available while suggesting "
             "areas where additional analysis might be helpful."
         )
        
        user_prompt = (
            f"Question: {query}\n\n"
            f"Relevant entities identified: {entities_text}\n\n"
            f"Community analyses:\n{summaries_text}\n\n"
            f"Please provide a comprehensive answer to the question based on these community insights."
        )
        
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ]
        
        try:
            response = self.llm.chat(messages)
            clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
            return clean_response
        except Exception as e:
            print(f"❌ Answer synthesis failed: {e}")
            # Provide a fallback response using the community summaries directly
            fallback = f"Based on the analysis of {len(community_summaries)} relevant communities:\n\n"
            for cid, summary in community_summaries.items():
                fallback += f"• {summary[:200]}...\n"
            return fallback
# ========================================================================
    # TEMPLATE INTEGRATION ENHANCEMENTS
    # Add these methods inside the FoodServiceGraphRAGQueryEngine class
    # ========================================================================
    
    def query_by_category(self, category: str, count: int, event_type: str, context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Query GraphRAG for specific category with exact count requirement
        Enhanced for template integration
        
        Args:
            category: GraphRAG category (e.g., "starter", "main_biryani", "side_bread")
            count: Exact number of items needed
            event_type: "Traditional", "Party", or "Premium"
            context: Additional context for better suggestions
        
        Returns:
            List of exactly 'count' items with insights and co-occurrence data
        """
        print(f"🔍 Querying GraphRAG for {count} {category} items for {event_type} events")
        
        # Build category-specific query
        query_text = self._build_enhanced_category_query(category, count, event_type, context)
        
        try:
            # Execute the main GraphRAG query
            raw_response = self.query(query_text)
            
            # Extract specific items from the response
            extracted_items = self._extract_items_from_graphrag_response(raw_response, category, count)
            
            # Enhance with co-occurrence data and insights
            enhanced_items = self._enhance_items_with_cooccurrence(extracted_items, event_type, category)
            
            # Ensure we have exactly the right count
            final_items = self._ensure_exact_count(enhanced_items, category, count, event_type)
            
            print(f"✅ Successfully found {len(final_items)} {category} items")
            return final_items
            
        except Exception as e:
            print(f"❌ Category query failed for {category}: {e}")
            # Return fallback items
            return self._get_fallback_items(category, count, event_type)

    def _build_enhanced_category_query(self, category: str, count: int, event_type: str, context: Optional[Dict] = None) -> str:
        """Build enhanced natural language queries optimized for specific categories"""
        base_queries = {
            "starter": f"What are the top {count} starter appetizer items that work exceptionally well for {event_type.lower()} events? Focus on items with strong historical co-occurrence patterns and proven success rates.",
            "main_biryani": f"Recommend the best {count} biryani dish(es) for {event_type.lower()} events based on community analysis and historical service success.Reurn the only dish names",
            "main_rice": f"Suggest {count} flavored rice or pulav dish that complements other menu items in {event_type.lower()} event settings.",
            "side_bread": f"What {count} bread item(s) have the strongest co-occurrence patterns with curry dishes in successful {event_type.lower()} event menus?",
            "side_curry": f"Recommend {count} curry dish that has proven compatibility with biryani and bread combinations in {event_type.lower()} contexts.",
            "side_accompaniment": f"What {count} accompaniment(s) like raita, salad, or pickle provide the best balance for {event_type.lower()} event meals based on historical data?",
            "dessert": f"List exactly {count} specific dessert names like 'Gulab Jamun' or 'Double Ka Meetha' that provide a good ending for {event_type.lower()} event meals. Return only the dessert names."
        }
        
        query = base_queries.get(category, f"Recommend {count} high-quality items from {category} category for {event_type.lower()} events")
        
        # Add context if provided
        if context:
            template_context = context.get("template_name", "")
            if template_context:
                query += f" This is for a {template_context} style menu."
        
        return query

    def _extract_items_from_graphrag_response(self, response: str, category: str, count: int) -> List[Dict[str, Any]]:
        """Extract specific item names from GraphRAG response using multiple strategies"""
        extracted_items = []
        
        # Load pricing data for item validation
        try:
            with open("items_price_uom.json", 'r', encoding='utf-8') as f:
                pricing_data = json.load(f)
                known_items = {item["item_name"].lower(): item for item in pricing_data}
        except Exception as e:
            print(f"⚠️ Could not load pricing data: {e}")
            known_items = {}
        
        response_lower = response.lower()
        found_names = set()
        
        # Strategy 1: Pattern-based extraction
        extraction_patterns = [
            r'"([^"]+)"',           # Quoted items
            r"'([^']+)'",           # Single quoted items  
            r'•\s*([^\n•]+)',       # Bullet points
            r'\d+\.\s*([^\n\d]+)',  # Numbered lists
            r'-\s*([^\n-]+)',       # Dash lists
            r'\*\s*([^\n*]+)',      # Asterisk lists
            r'(?:recommend|suggest|include)(?:s|ing)?\s+([^\n,.]+)', # Recommendation phrases
        ]
        
        for pattern in extraction_patterns:
            matches = re.findall(pattern, response_lower, re.IGNORECASE)
            for match in matches:
                clean_name = self._clean_extracted_name(match.strip())
                
                # Try to match with known items
                matched_item = self._find_best_item_match(clean_name, known_items, category)
                if matched_item and matched_item["item_name"].lower() not in found_names:
                    extracted_items.append({
                        "name": matched_item["item_name"],
                        "category": matched_item["category"],
                        "source": "pattern_extraction",
                        "match_confidence": self._calculate_match_confidence(clean_name, matched_item["item_name"].lower()),
                        "extraction_method": pattern
                    })
                    found_names.add(matched_item["item_name"].lower())
                    
                    if len(extracted_items) >= count:
                        break
            
            if len(extracted_items) >= count:
                break
        
        # Strategy 2: Direct name matching for remaining slots
        if len(extracted_items) < count:
            for known_name, item_data in known_items.items():
                if known_name in response_lower and known_name not in found_names:
                    if self._is_category_match(item_data["category"], category):
                        extracted_items.append({
                            "name": item_data["item_name"],
                            "category": item_data["category"], 
                            "source": "direct_match",
                            "match_confidence": 1.0,
                            "extraction_method": "direct_name_match"
                        })
                        found_names.add(known_name)
                        
                        if len(extracted_items) >= count:
                            break
        
        # Sort by confidence and return top items
        extracted_items.sort(key=lambda x: x["match_confidence"], reverse=True)
        return extracted_items[:count]

    def _clean_extracted_name(self, raw_name: str) -> str:
        """Clean extracted item names to improve matching"""
        clean_name = raw_name.strip()
        clean_name = re.sub(r'^(?:the|a|an)\s+', '', clean_name, flags=re.IGNORECASE)
        clean_name = re.sub(r'[^\w\s]', '', clean_name)
        clean_name = re.sub(r'\s+', ' ', clean_name)
        return clean_name.strip()

    def _find_best_item_match(self, extracted_name: str, known_items: Dict[str, Any], category: str) -> Optional[Dict[str, Any]]:
        """Find the best matching item from known items database"""
        best_match = None
        best_score = 0.0
        
        for known_name, item_data in known_items.items():
            if not self._is_category_match(item_data["category"], category):
                continue
            
            similarity = self._calculate_item_similarity(extracted_name, known_name)
            
            if similarity > best_score and similarity >= 0.6:
                best_score = similarity
                best_match = item_data
        
        return best_match

    def _calculate_item_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two item names"""
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard = len(intersection) / len(union)
        
        # Bonus for substring containment
        substring_bonus = 0.0
        if name1.lower() in name2.lower() or name2.lower() in name1.lower():
            substring_bonus = 0.2
        
        return min(1.0, jaccard + substring_bonus)

    def _is_category_match(self, item_category: str, query_category: str) -> bool:
        """Enhanced category matching for template integration"""
        category_mappings = {
            "starter": ["Starters", "Snacks"],
            "main_biryani": ["Main Course"],
            "main_rice": ["Main Course"], 
            "side_bread": ["Main Course"],
            "side_curry": ["Main Course"],
            "side_accompaniment": ["Sides & Accompaniments"],
            "dessert": ["Desserts", "Sweets"]
        }
        
        appropriate_categories = category_mappings.get(query_category, [query_category])
        return item_category in appropriate_categories

    def _enhance_items_with_cooccurrence(self, items: List[Dict[str, Any]], event_type: str, category: str) -> List[Dict[str, Any]]:
        """Enhance items with co-occurrence insights and success metrics"""
        enhanced_items = []
        
        for item in items:
            enhanced_item = item.copy()
            
            # Get co-occurrence insights from graph
            cooccurrence_data = self._get_item_cooccurrence_data(item["name"], event_type)
            
            # Generate condensed insight
            insight = self._generate_enhanced_insight(item["name"], category, event_type, cooccurrence_data)
            
            enhanced_item.update({
                "insight": insight,
                "cooccurrence_score": cooccurrence_data.get("score", 0.7),
                "cooccurrence_frequency": cooccurrence_data.get("frequency", 0),
                "success_indicators": cooccurrence_data.get("indicators", {})
            })
            
            enhanced_items.append(enhanced_item)
        
        return enhanced_items

    def _get_item_cooccurrence_data(self, item_name: str, event_type: str) -> Dict[str, Any]:
        """Query the graph database for actual co-occurrence data"""
        try:
            # Query Neo4j for co-occurrence relationships
            cooccurrence_query = """
            MATCH (item:Dish {name: $item_name})-[r:CO_OCCURS]-(other:Dish)
            WHERE r.frequency >= 2
            RETURN other.name as paired_item, r.frequency as frequency, r.strength as strength
            ORDER BY r.frequency DESC
            LIMIT 10
            """
            
            # Execute query through graph store if available
            if hasattr(self.graph_store, 'neo4j_store'):
                results = self.graph_store.neo4j_store.structured_query(
                    cooccurrence_query, 
                    param_map={"item_name": item_name}
                )
                
                # Process results
                paired_items = []
                total_frequency = 0
                high_strength_count = 0
                
                for result in results:
                    paired_items.append({
                        "item": result["paired_item"],
                        "frequency": result["frequency"], 
                        "strength": result["strength"]
                    })
                    total_frequency += result["frequency"]
                    if result["strength"] == "high":
                        high_strength_count += 1
                
                # Calculate co-occurrence score
                score = min(1.0, (total_frequency * 0.1) + (high_strength_count * 0.2))
                
                return {
                    "score": score,
                    "frequency": total_frequency,
                    "paired_items": paired_items,
                    "indicators": {
                        "total_pairings": len(paired_items),
                        "high_strength_pairings": high_strength_count,
                        "avg_frequency": total_frequency / len(paired_items) if paired_items else 0
                    }
                }
                
        except Exception as e:
            print(f"⚠️ Could not fetch co-occurrence data for {item_name}: {e}")
        
        # Fallback to estimated data
        return {
            "score": 0.7,
            "frequency": 3,
            "paired_items": [],
            "indicators": {"estimated": True}
        }

    def _generate_enhanced_insight(self, item_name: str, category: str, event_type: str, cooccurrence_data: Dict[str, Any]) -> str:
        """Generate condensed insights with co-occurrence information"""
        
        # Base insights by category and event type
        base_insights = {
            ("starter", "Traditional"): "Classic traditional starter choice",
            ("starter", "Party"): "Popular party engagement item",
            ("main_biryani", "Traditional"): "Traditional centerpiece with authentic appeal",
            ("main_biryani", "Party"): "Crowd-pleasing party centerpiece",
            ("side_bread", "Traditional"): "Traditional accompaniment for authentic meals",
            ("side_bread", "Party"): "Versatile party side for diverse tastes",
            ("side_curry", "Traditional"): "Authentic curry for traditional settings",
            ("side_curry", "Party"): "Flavorful curry complement for party menus",
            ("dessert", "Traditional"): "Traditional sweet conclusion",
            ("dessert", "Party"): "Satisfying party dessert favorite"
        }
        
        base_insight = base_insights.get((category, event_type), f"Quality {category} choice for {event_type.lower()} events")
        
        # Add co-occurrence enhancement
        frequency = cooccurrence_data.get("frequency", 0)
        
        if frequency >= 5:
            cooccurrence_text = f"High co-occurrence success (freq: {frequency})"
        elif frequency >= 3:
            cooccurrence_text = f"Proven pairing success (freq: {frequency})"
        elif frequency >= 2:
            cooccurrence_text = f"Validated combination (freq: {frequency})"
        else:
            cooccurrence_text = f"Community recommended choice"
        
        return f"{base_insight}. {cooccurrence_text}"

    def _ensure_exact_count(self, items: List[Dict[str, Any]], category: str, count: int, event_type: str) -> List[Dict[str, Any]]:
        """Ensure we return exactly the requested count of items"""
        if len(items) >= count:
            return items[:count]
        
        # Need more items - get fallbacks
        needed = count - len(items)
        fallback_items = self._get_fallback_items(category, needed, event_type)
        
        # Mark existing item names to avoid duplicates
        existing_names = {item["name"].lower() for item in items}
        
        # Add non-duplicate fallbacks
        for fallback in fallback_items:
            if fallback["name"].lower() not in existing_names:
                items.append(fallback)
                if len(items) >= count:
                    break
        
        return items[:count]

    def _get_fallback_items(self, category: str, count: int, event_type: str) -> List[Dict[str, Any]]:
        """Get fallback items when GraphRAG doesn't return enough suggestions"""
        
        # Curated fallback items for each category
        fallback_data = {
            "starter": [
                {"name": "Veg Samosa", "category": "Snacks", "reason": "Universally popular starter"},
                {"name": "Chicken 65", "category": "Starters", "reason": "Classic party favorite"},
                {"name": "Paneer Tikka", "category": "Starters", "reason": "Vegetarian crowd pleaser"},
                {"name": "Veg Spring Rolls", "category": "Starters", "reason": "Light and appealing"},
                {"name": "Cheese Corn Balls", "category": "Starters", "reason": "Modern party choice"}
            ],
            "main_biryani": [
                {"name": "Veg Biryani", "category": "Main Course", "reason": "Safe vegetarian choice"},
                {"name": "Chicken Dum Biryani", "category": "Main Course", "reason": "Traditional favorite"},
                {"name": "Mutton Biryani", "category": "Main Course", "reason": "Premium option"}
            ],
            "main_rice": [
                {"name": "Jeera Rice", "category": "Main Course", "reason": "Simple and versatile"},
                {"name": "Temple Style Pulihora", "category": "Main Course", "reason": "Traditional choice"}
            ],
            "side_bread": [
                {"name": "Chapati", "category": "Main Course", "reason": "Basic bread staple"},
                {"name": "Butter Naan", "category": "Main Course", "reason": "Popular bread choice"},
                {"name": "Rumali Roti", "category": "Main Course", "reason": "Light and flexible"}
            ],
            "side_curry": [
                {"name": "Dal Tadka", "category": "Main Course", "reason": "Universal dal choice"},
                {"name": "Mixed Vegetable Curry", "category": "Main Course", "reason": "Balanced vegetable option"},
                {"name": "Rajma Curry", "category": "Main Course", "reason": "Protein-rich option"}
            ],
            "side_accompaniment": [
                {"name": "Raita", "category": "Sides & Accompaniments", "reason": "Cooling accompaniment"},
                {"name": "Pickle", "category": "Sides & Accompaniments", "reason": "Traditional condiment"},
                {"name": "Salad", "category": "Sides & Accompaniments", "reason": "Fresh addition"}
            ],
            "dessert": [
                {"name": "Gulab Jamun", "category": "Sweets", "reason": "Classic Indian sweet"},
                {"name": "Double Ka Meetha", "category": "Sweets", "reason": "Traditional dessert"},
                {"name": "Tiramisu", "category": "Desserts", "reason": "Modern dessert choice"}
            ]
        }
        
        category_fallbacks = fallback_data.get(category, [])
        selected_fallbacks = []
        
        for i, fallback in enumerate(category_fallbacks[:count]):
            selected_fallbacks.append({
                "name": fallback["name"],
                "category": fallback["category"],
                "source": "community_fallback",
                "match_confidence": 0.8,
                "insight": f"Community backup: {fallback['reason']}",
                "cooccurrence_score": 0.6,
                "fallback_reason": fallback["reason"]
            })
        
        return selected_fallbacks

    def get_co_occurrence_insights(self, item_name: str, event_type: str) -> Dict[str, Any]:
        """Get detailed co-occurrence insights for a specific item - Public method"""
        return self._get_item_cooccurrence_data(item_name, event_type)

    def extract_items_from_response(self, response_text: str, category: str, count: int) -> List[Dict[str, Any]]:
        """Public method to extract items from GraphRAG response"""
        return self._extract_items_from_graphrag_response(response_text, category, count)

    def _calculate_match_confidence(self, extracted: str, known: str) -> float:
        """Calculate confidence score for item name matching"""
        if not extracted or not known:
            return 0.0
        
        extracted_words = set(extracted.lower().split())
        known_words = set(known.lower().split())
        
        if not extracted_words or not known_words:
            return 0.0
        
        intersection = extracted_words.intersection(known_words)
        union = extracted_words.union(known_words)
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Bonus for exact substring matches
        if extracted.lower() in known.lower() or known.lower() in extracted.lower():
            jaccard += 0.2
        
        return min(1.0, jaccard)

    # ========================================================================
    # END TEMPLATE INTEGRATION ENHANCEMENTS  
    # ========================================================================

class Neo4jGraphRAGAdapter:
    """
    An adapter that bridges the community detection capabilities with your existing
    Neo4j graph store. This allows you to leverage the powerful community analysis
    while continuing to use Neo4j as your primary data storage system.
    
    Think of this as a translator that helps the community detection algorithms
    work with your existing Neo4j infrastructure, combining the best of both worlds:
    the robustness of Neo4j for data storage and the analytical power of community
    detection for insight generation.
    """
    
    def __init__(self, neo4j_store, llm=None, min_cluster_size=3,max_cluster_size=64):
        """
        Initialize the adapter to work with your Neo4j store.
        
        Args:
            neo4j_store: Your existing Neo4jPropertyGraphStore
            llm: Language model for community summary generation
            max_cluster_size: Maximum size for detected communities
        """
        self.neo4j_store = neo4j_store
        self.llm = llm if llm is not None else Settings.llm
        self.max_cluster_size = max_cluster_size
        
        # Create our custom GraphRAG store for community analysis
        self.graphrag_store = FoodServiceGraphRAGStore(
            llm=self.llm, 
            max_cluster_size=max_cluster_size,
            min_cluster_size=min_cluster_size
        )
        
        # Track synchronization state
        self._last_sync_time = None

    def sync_from_neo4j(self):
        """
        Synchronizes the graph data from Neo4j to the in-memory GraphRAG store.
        This method is crucial for ensuring the in-memory graph used by GraphRAG
        is up-to-date with the Neo4j database. It transfers nodes and relationships,
        assigning appropriate labels and properties.
        """
        print("🔄 Synchronizing data from Neo4j to community analysis store...")

        nodes_to_upsert: List[EntityNode] = []
        relations_to_upsert: List[Relation] = []

        # Transfer nodes from Neo4j
        print("📊 Transferring nodes...")
        cypher_query = "MATCH (n) RETURN n, labels(n) as labels, elementId(n) as element_id"
        try:
            neo4j_nodes = self.neo4j_store.structured_query(cypher_query)

            for result in neo4j_nodes:
                node_data = result["n"]
                labels = result["labels"]
                element_id = result["element_id"]

                # Prioritize 'id' property from Neo4j node, then fallback to elementId
                node_id = node_data.get("id", element_id)
                
                # Derive primary label
                primary_label = "__Node__" # Default label
                if "Category" in labels:
                    primary_label = "Category"
                elif "Dish" in labels:
                    primary_label = "Dish"
                elif "EventType" in labels:
                    primary_label = "EventType"
                elif "MealType" in labels:
                    primary_label = "MealType"
                elif "Ingredient" in labels:
                    primary_label = "Ingredient"
                elif "Person" in labels:
                    primary_label = "Person"
                elif "Order" in labels:
                    primary_label = "Order"
                elif "Customer" in labels:
                    primary_label = "Customer"
                elif "Venue" in labels:
                    primary_label = "Venue"
                elif "__Entity__" in labels:
                    primary_label = "__Entity__"
                
                # Copy all properties from the Neo4j node data
                properties = dict(node_data)
                
                # Ensure 'name' is present in properties, or use node_id as fallback
                node_name = properties.get("name", node_id)
                properties["name"] = node_name # Ensure name property is explicitly set
                
                # Remove the 'id' key from properties if it's redundant with the derived node_id
                if "id" in properties and properties["id"] == node_id:
                    del properties["id"]

                # Create EntityNode object
                entity_node = EntityNode(
                    id=node_id,
                    label=primary_label,
                    name=node_name,
                    properties=properties
                )
                nodes_to_upsert.append(entity_node)

                #print(f"DEBUG_SYNC: Prepared node: ID='{node_id}', Labels={labels}, Derived_Primary_Label='{primary_label}'")

            # Add all collected nodes to the graph in one go
            self.graphrag_store.upsert_nodes(nodes_to_upsert)
            print(f"✅ Transferred {len(nodes_to_upsert)} nodes.")

        except Exception as e:
            logger.error(f"Error transferring nodes: {e}")
            raise # Re-raise the exception to propagate it

        # Transfer relationships from Neo4j
        print("🔗 Transferring relationships...")
        # Corrected: Use properties(r) to ensure a dictionary is returned for relationship properties
        rel_query = "MATCH (s)-[r]->(t) RETURN s.id as source_id, type(r) as type, t.id as target_id, properties(r) as properties, elementId(r) as element_id"
        try:
            neo4j_rels = self.neo4j_store.structured_query(rel_query)
            for result in neo4j_rels:
                source_id = result["source_id"]
                rel_type = result["type"]
                target_id = result["target_id"]
                rel_properties = result["properties"]
                # element_id = result["element_id"] # Not strictly needed for Relation object
                
                # Ensure source_id and target_id are not None
                if source_id is None or target_id is None:
                    print(f"WARNING: Skipping relationship due to missing source or target ID: Source={source_id}, Target={target_id}, Type={rel_type}")
                    continue

                # Create Relation object
                relation_obj = Relation(
                    source_id=source_id,
                    target_id=target_id,
                    label=rel_type,
                    properties=rel_properties # rel_properties is already a dict
                )
                relations_to_upsert.append(relation_obj)

                #print(f"DEBUG_SYNC: Prepared relation: Source='{source_id}', Type='{rel_type}', Target='{target_id}'")

            # Add all collected relationships to the graph in one go
            self.graphrag_store.upsert_relations(relations_to_upsert)
            print(f"✅ Transferred {len(relations_to_upsert)} relationships.")

        except Exception as e:
            logger.error(f"Error transferring relationships: {e}")
            raise # Re-raise the exception to propagate it

        print("✨ Graph synchronization complete.")

    # ... rest of the class ...    
    
    def _create_relation_description(self, source: str, target: str, 
                                   relation_type: str, properties: dict) -> str:
        """
        Create meaningful descriptions for relationships that help with community analysis.
        
        This method transforms technical relationship data into natural language
        descriptions that the community detection algorithm can better understand
        and that the language model can use for generating insights.
        Enhanced to handle historical co-occurrence patterns from menu data.
        """
        # Handle the new CO-OCCURS relationships with detailed context
        if relation_type == "CO_OCCURS":
            frequency = properties.get('frequency', 1)
            strength = properties.get('strength', 'unknown')
            
            # Create rich descriptions that emphasize the historical success of these pairings
            if strength == "high":
                return (f"The dishes '{source}' and '{target}' have been successfully paired "
                       f"together {frequency} times across multiple menus, representing a "
                       f"proven high-confidence combination based on historical service data")
            elif strength == "medium":
                return (f"The dishes '{source}' and '{target}' have appeared together "
                       f"{frequency} times, indicating a reliable pairing that has been "
                       f"validated through actual menu service")
            else:
                return (f"The dishes '{source}' and '{target}' have co-occurred "
                       f"{frequency} times, suggesting potential compatibility based on "
                       f"historical menu combinations")
    
        # Create context-aware descriptions based on relationship types in food service
        elif relation_type == "CONTAINS":
            return f"The dish '{source}' contains the ingredient '{target}' as a key component"
        elif relation_type == "BELONGS_TO":
            return f"The dish '{source}' belongs to the '{target}' category"
        elif relation_type == "SUITABLE_FOR":
            return f"The dish '{source}' is suitable for '{target}' events or occasions"
        elif relation_type == "SERVED_DURING":
            return f"The dish '{source}' is typically served during '{target}' meal periods"
        elif relation_type == "COMPLEMENTS":
            return f"The dish '{source}' complements or pairs well with '{target}'"
        else:
            # Generic description for unknown relationship types
            return f"'{source}' has a '{relation_type.lower()}' relationship with '{target}'"
    
    def build_communities_from_neo4j(self):
        """
        Complete workflow to analyze your Neo4j graph and build community summaries.
        
        This method orchestrates the entire process of transforming your static
        Neo4j graph into a dynamic, insight-rich community-based knowledge system.
        """
        
        print("🚀 Starting complete community analysis workflow...")
        
        # Step 1: Sync data from Neo4j
        if not self.sync_from_neo4j():
            print("❌ Failed to sync data from Neo4j")
            return False
        
        # Step 2: Run community detection and generate summaries
        self.graphrag_store.build_communities()
        
        # Step 3: Validate results
        summaries = self.graphrag_store.get_community_summaries()
        if summaries:
            print(f"🎉 Successfully generated {len(summaries)} community summaries!")
            
            # Show a preview of what was discovered
            print("\n📋 Community Summary Preview:")
            for community_id, summary in list(summaries.items())[:3]:  # Show first 3
                preview = summary[:150] + "..." if len(summary) > 150 else summary
                print(f"  Community {community_id}: {preview}")
            
            if len(summaries) > 3:
                print(f"  ... and {len(summaries) - 3} more communities")
            
            return True
        else:
            print("⚠️  No community summaries were generated")
            return False
    
    def create_query_engine(self, similarity_top_k=5):
        """
        Create a query engine that leverages the community summaries.
        
        Args:
            similarity_top_k: Number of most relevant entities to consider per query
        
        Returns:
            A FoodServiceGraphRAGQueryEngine ready for querying
        """
        
        if not self.graphrag_store.get_community_summaries():
            raise ValueError(
                "No community summaries available. Please run build_communities_from_neo4j() first."
            )
        
        return FoodServiceGraphRAGQueryEngine(
            graph_store=self.graphrag_store,
            llm=self.llm,
            similarity_top_k=similarity_top_k
        )

def setup_complete_community_graphrag_system():
    """
    Complete setup function that brings together all the components:
    local models, Neo4j integration, and community-based GraphRAG.
    
    This function orchestrates the entire system setup, from configuring
    local models to building community summaries to creating the query engine.
    """
    
    print("🔧 Setting up complete Community-based GraphRAG system...")
    
    # Step 1: Configure local models (as we did before)
    print("\n1️⃣ Configuring local models...")
    print("\n setting up embedding model...")
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./model_cache/embeddings"
    )
    print("\n1️⃣ Configuring OpenAI API...")
    import os
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please get your API key from: https://aistudio.google.com/app/apikey")
        return None, None
    try:
     llm = OpenAI(
             model="gpt-4o",  # Optimal for GraphRAG tasks
             temperature=0.1,  # Consistent for entity extraction
             max_tokens=2048   # Sufficient for community analysis
         )
         
         # Test the connection
     test_response = llm.complete("Test connection")
     print(f"✓ API connected successfully")
         
    except Exception as e:
        print(f"❌ API setup failed: {e}")
        print("Check your API key and internet connection")
        return None, None
    
        # try:
        #     # Fallback to minimal configuration that should work with any API version
        #     llm = HuggingFaceLLM(
        #         model_name="google/flan-t5-large",
        #         tokenizer_name="google/flan-t5-large",
        #         max_new_tokens=512
        #         # Using only the most essential parameters to ensure compatibility
        #     )
        #     print("✅ Language model configured with simplified parameters")
            
        # except Exception as e2:
        #     print(f"❌ Language model configuration failed completely: {e2}")
        #     print("This suggests a deeper compatibility issue with the HuggingFace integration")
        #     return None, None
    
    # except Exception as e:
    #     print(f"❌ Unexpected error in language model setup: {e}")
    #     return None, None
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    print(f"✓ Local models configured")
    
    # Step 2: Connect to Neo4j
    print("\n2️⃣ Connecting to Neo4j...")
    try:
     from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
     
     neo4j_store = Neo4jPropertyGraphStore(
         url="bolt://127.0.0.1:7687",
         username="neo4j",
         password="Ashfaq8790",
         encrypted=False,
         refresh_schema=True,
         create_indexes=False
     )
     
     print("✓ Connected to Neo4j")
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        print("Please verify that Neo4j is running and your credentials are correct")
        return None, None
    
    # Step 3: Create the adapter and build communities
    print("\n3️⃣ Creating community analysis adapter...")
    try:
     adapter = Neo4jGraphRAGAdapter(
         neo4j_store=neo4j_store,
         llm=llm,
         max_cluster_size=64,
         min_cluster_size=3
         # Adjust based on your data size
     )
    
     print("✓ Adapter created")
    except Exception as e:
        print(f"❌ Failed to create adapter: {e}")
        return None, None
    
    # Step 4: Build communities from your Neo4j data
    print("\n4️⃣ Building communities and generating summaries...")
    print("This process may take several minutes depending on your graph size...")
    success = adapter.build_communities_from_neo4j()
    
    if not success:
        print("❌ Failed to build communities")
        return None, None
    
    # Step 5: Create the query engine
    print("\n5️⃣ Creating community-powered query engine...")
    
    query_engine = adapter.create_query_engine(similarity_top_k=5)
    
    print("✅ Complete Community GraphRAG system ready!")
     # Provide a summary of what was accomplished
    summaries = adapter.graphrag_store.get_community_summaries()
    if summaries:
        print(f"📊 System contains {len(summaries)} community summaries")
        print("🔍 Ready for complex food service queries")
        
        # Show a brief preview of the communities that were discovered
        print("\n📋 Sample communities discovered:")
        for i, (community_id, summary) in enumerate(list(summaries.items())[:3]):
            preview = summary[:100] + "..." if len(summary) > 100 else summary
            print(f"  Community {community_id}: {preview}")
        
        if len(summaries) > 3:
            print(f"  ... and {len(summaries) - 3} more communities")
    
    return adapter, query_engine
def setup_graphrag_core_system():
    """
    Core setup function that initializes essential components:
    local models, Neo4j integration, and the GraphRAG adapter.
    
    This function prepares the system for community analysis but does
    not perform the analysis itself.
    """
    
    print("🔧 Setting up GraphRAG Core System...")
    
    # Step 1: Configure local models (as we did before)
    print("\n1️⃣ Configuring local models...")
    print("\n setting up embedding model...")
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./model_cache/embeddings"
    )
    print("\n1️⃣ Configuring Gemini API...")
    import os
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please get your API key from: https://aistudio.google.com/app/apikey")
        return None
    try:
     llm = OpenAI(
            model="gpt-4o",  # Optimal for GraphRAG tasks
            temperature=0.1,  # Consistent for entity extraction
            max_tokens=2048   # Sufficient for community analysis
         )
         
         # Test the connection
     test_response = llm.complete("Test connection")
     print(f"✓ Gemini API connected successfully")
         
    except Exception as e:
        print(f"❌ Gemini API setup failed: {e}")
        print("Check your API key and internet connection")
        return None
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    print(f"✓ Local models configured")
    
    # Step 2: Connect to Neo4j
    print("\n2️⃣ Connecting to Neo4j...")
    try:
     from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
     
     neo4j_store = Neo4jPropertyGraphStore(
         url="bolt://127.0.0.1:7687",
         username="neo4j",
         password="Ashfaq8790",
         encrypted=False,
         refresh_schema=True,
         create_indexes=False
     )
     
     print("✓ Connected to Neo4j")
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        print("Please verify that Neo4j is running and your credentials are correct")
        return None
    
    # Step 3: Create the adapter
    print("\n3️⃣ Creating community analysis adapter...")
    try:
     adapter = Neo4jGraphRAGAdapter(
         neo4j_store=neo4j_store,
         llm=llm,
         max_cluster_size=64,
         min_cluster_size=3
         # Adjust based on your data size
     )
    
     print("✓ Adapter created")
    except Exception as e:
        print(f"❌ Failed to create adapter: {e}")
        return None
    
    print("✅ GraphRAG Core System ready!")
    return adapter


def test_community_graphrag_system(query_engine):
    """
    Test the community-based system with sample queries to demonstrate its capabilities.
    """
    
    print("\n🧪 Testing Community-based GraphRAG system...")
    
    # Test queries that should benefit from community analysis
    test_queries = [
        "What ingredients work well together in breakfast dishes?",
        "How do dessert components relate to each other?",
        "What are the main themes in our appetizer offerings?",
        "Which ingredient combinations define our comfort food menu items?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i} ---")
        print(f"Question: {query}")
        
        try:
            response = query_engine.query(query)
            print(f"Community-based Answer: {response}")
        except Exception as e:
            print(f"Query failed: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    # Check system capabilities
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Set up the complete system
    adapter, query_engine = setup_complete_community_graphrag_system()
    
    if adapter and query_engine:
        # Test the system
        test_community_graphrag_system(query_engine)
        
        # Interactive query loop
        print("\n🎯 Ready for interactive queries!")
        print("Ask questions about your food service data, or type 'quit' to exit.")
        
        while True:
            user_query = input("\nYour question: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_query:
                try:
                    response = query_engine.query(user_query)
                    print(f"\nAnswer: {response}")
                except Exception as e:
                    print(f"Error processing query: {e}")
    else:
        print("❌ System setup failed. Please check your configuration and try again.")