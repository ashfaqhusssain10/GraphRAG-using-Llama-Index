import re
import networkx as nx
from graspologic.partition import hierarchical_leiden
from collections import defaultdict
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.llms import ChatMessage
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import google.generativeai as genai
import torch
from typing import Dict, List, Any
from dotenv import load_dotenv
load_dotenv()

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
    
    def __init__(self, llm=None, max_cluster_size=10):
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
    
    def __init__(self, neo4j_store, llm=None, max_cluster_size=10):
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
            max_cluster_size=max_cluster_size
        )
        
        # Track synchronization state
        self._last_sync_time = None
    
    def sync_from_neo4j(self):
        """
        Transfer data from your Neo4j store to the community analysis store.
        
        This method reads your existing graph structure from Neo4j and prepares
        it for community detection analysis. We preserve all the relationship
        information while adapting the data format for the clustering algorithms.
        """
        
        print("🔄 Synchronizing data from Neo4j to community analysis store...")
        
        # Clear existing data in the analysis store
        self.graphrag_store.graph.nodes.clear()
        self.graphrag_store.graph.relations.clear()
        
        # Transfer nodes from Neo4j
        print("📊 Transferring nodes...")
        node_count = 0
        
        # Query all nodes from Neo4j
        cypher_query = "MATCH (n) RETURN n, labels(n) as labels"
        try:
            neo4j_nodes = self.neo4j_store.structured_query(cypher_query)
            
            for result in neo4j_nodes:
                node = result['n']
                labels = result['labels']
                
                # Extract node properties and create appropriate node for analysis
                node_id = node.get('name', str(node.get('id', f'node_{node_count}')))
                node_properties = dict(node)
                
                # Add label information to properties for better community analysis
                if labels:
                    node_properties['node_type'] = labels[0]  # Primary label
                    node_properties['all_labels'] = labels
                
                # Create node in our analysis store
                from llama_index.core.graph_stores.types import EntityNode
                entity_node = EntityNode(
                    id=node_id,
                    label=labels[0] if labels else "Entity",
                    name=node_id,
                    properties=node_properties
                )
                
                self.graphrag_store.graph.nodes[node_id] = entity_node
                node_count += 1
            
            print(f"✓ Transferred {node_count} nodes")
            
        except Exception as e:
            print(f"❌ Failed to transfer nodes: {e}")
            return False
        
        # Transfer relationships from Neo4j
        print("🔗 Transferring relationships...")
        relation_count = 0
        
        # Query all relationships from Neo4j with detailed information
        cypher_query = """
        MATCH (source)-[r]->(target)
        RETURN source.name as source_name, target.name as target_name, 
               type(r) as relation_type, properties(r) as properties
        """
        
        try:
            neo4j_relations = self.neo4j_store.structured_query(cypher_query)
            
            for result in neo4j_relations:
                source_name = result['source_name']
                target_name = result['target_name']
                relation_type = result['relation_type']
                relation_props = result['properties'] or {}
                
                # Create meaningful relationship description for community analysis
                relation_description = self._create_relation_description(
                    source_name, target_name, relation_type, relation_props
                )
                
                # Add relationship description to properties
                enhanced_props = dict(relation_props)
                enhanced_props['relationship_description'] = relation_description
                
                # Create relation in our analysis store
                from llama_index.core.graph_stores.types import Relation
                relation = Relation(
                    source_id=source_name,
                    target_id=target_name,
                    label=relation_type,
                    properties=enhanced_props
                )
                
                relation_id = f"{source_name}_{relation_type}_{target_name}_{relation_count}"
                self.graphrag_store.graph.relations[relation_id] = relation
                relation_count += 1
            
            print(f"✓ Transferred {relation_count} relationships")
            
        except Exception as e:
            print(f"❌ Failed to transfer relationships: {e}")
            return False
        
        print("✅ Neo4j synchronization complete!")
        return True
    
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
    print("\n1️⃣ Configuring Gemini API...")
    import os
    # Check if API key is available
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not found in environment variables")
        print("Please get your API key from: https://aistudio.google.com/app/apikey")
        return None, None
    try:
     llm = Gemini(
             model="models/gemini-1.5-flash",  # Optimal for GraphRAG tasks
             temperature=0.1,  # Consistent for entity extraction
             max_tokens=2048   # Sufficient for community analysis
         )
         
         # Test the connection
     test_response = llm.complete("Test connection")
     print(f"✓ Gemini API connected successfully")
         
    except Exception as e:
        print(f"❌ Gemini API setup failed: {e}")
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
         max_cluster_size=8  # Adjust based on your data size
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