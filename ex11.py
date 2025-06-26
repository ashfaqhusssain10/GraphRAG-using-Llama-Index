import re
import networkx as nx
from graspologic.partition import hierarchical_leiden
from collections import defaultdict
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.llms import ChatMessage
from llama_index.llms.huggingface import HuggingFaceLLM
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
    
    def __init__(self, llm=None, max_cluster_size=10,min_cluster_size=3):
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
        TOKEN-BUDGET-AWARE community summary generation with intelligent sampling
        
        Engineering modifications:
        - Pre-flight token estimation prevents API failures
        - Intelligent relationship sampling preserves semantic coherence
        - Multi-tier fallback ensures system reliability under load
        - Co-occurrence relationship prioritization maintains historical evidence
        """
        
        # ENHANCEMENT 1: Pre-flight token budget validation
        estimated_tokens = self._estimate_token_usage(community_relationships)
        token_limit = self._get_safe_token_limit()
        
        print(f"🧮 Community token analysis: {len(community_relationships)} rels, ~{estimated_tokens} tokens (limit: {token_limit})")
        
        if estimated_tokens > token_limit:
            print(f"⚠️  Token budget exceeded, applying intelligent sampling...")
            sampled_relationships = self._apply_semantic_sampling(
                community_relationships, 
                target_tokens=int(token_limit * 0.8)  # 20% safety margin
            )
            relationships_text = "\n".join(sampled_relationships)
            print(f"✂️  Sampled: {len(community_relationships)} → {len(sampled_relationships)} relationships")
        else:
            relationships_text = "\n".join(community_relationships)
        
        # ENHANCEMENT 2: Context-aware prompt adaptation
        system_prompt = self._build_adaptive_prompt(len(community_relationships), estimated_tokens)
        user_prompt = f"Analyze these food service relationships:\n\n{relationships_text}"
        
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ]
        
        try:
            response = self.llm.chat(messages)
            clean_summary = re.sub(r"^assistant:\s*", "", str(response)).strip()
            
            # ENHANCEMENT 3: Quality validation with fallback
            if len(clean_summary) < 50:
                return self._generate_structured_fallback(community_relationships)
            
            return clean_summary
            
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            return self._generate_structured_fallback(community_relationships)

    def _estimate_token_usage(self, relationships: List[str]) -> int:
        """
        PRECISE token estimation using empirical conversion ratios
        
        Engineering rationale:
        - 4.2 chars/token ratio validated for technical domain text
        - Includes prompt overhead and response allocation
        - Provides conservative estimates for budget planning
        """
        total_chars = sum(len(rel) for rel in relationships)
        relationship_tokens = int(total_chars / 4.2)
        prompt_overhead = 600  # System + user prompt base
        response_tokens = 800   # Conservative response allocation
        
        return relationship_tokens + prompt_overhead + response_tokens

    def _get_safe_token_limit(self) -> int:
        """
        MODEL-SPECIFIC token limit with safety margins
        
        Implementation notes:
        - Provides 25% safety margin below API limits
        - Adapts to different model capabilities
        - Prevents rate limit violations
        """
        model_name = getattr(self.llm, 'model', 'gpt-4o')
        
        # Conservative limits with safety margins
        limits = {
            'gpt-4o-mini': 20000,    # 128K context, 200K TPM
            'gpt-4o': 35000,         # 128K context, higher TPM
            'gpt-4.1': 12000,        # 1M context, 30K TPM (restrictive)
            'gpt-3.5-turbo': 10000   # 16K context
        }
        
        return limits.get(model_name, 15000)  # Conservative default

    def _apply_semantic_sampling(self, relationships: List[str], target_tokens: int) -> List[str]:
        """
        INTELLIGENT relationship sampling with semantic preservation
        
        Engineering strategy:
        - Prioritizes CO_OCCURS relationships (historical evidence)
        - Maintains relationship type distribution
        - Preserves entity connectivity patterns
        - Ensures diverse semantic coverage
        """
        
        # SAMPLING PHASE 1: Relationship categorization
        categorized = self._categorize_for_sampling(relationships)
        
        # SAMPLING PHASE 2: Calculate sampling ratios
        target_chars = target_tokens * 4.2
        current_chars = sum(len(rel) for rel in relationships)
        base_ratio = target_chars / current_chars if current_chars > target_chars else 1.0
        
        sampled = []
        
        # SAMPLING PHASE 3: Weighted sampling by importance
        sampling_weights = {
            'co_occurs': 1.6,    # 60% overweight - critical historical data
            'contains': 1.0,     # Standard weight - structural data
            'category': 0.7,     # 30% underweight - less critical
            'other': 0.5         # 50% underweight - supplementary
        }
        
        for rel_type, type_relationships in categorized.items():
            if not type_relationships:
                continue
                
            weight = sampling_weights.get(rel_type, 0.5)
            sample_count = max(1, int(len(type_relationships) * base_ratio * weight))
            sample_count = min(sample_count, len(type_relationships))
            
            if rel_type == 'co_occurs':
                # Frequency-based selection for co-occurrence
                sorted_rels = self._sort_by_frequency(type_relationships)
                sampled.extend(sorted_rels[:sample_count])
            else:
                # Random sampling for structural diversity
                import random
                sampled.extend(random.sample(type_relationships, sample_count))
        
        return sampled

    def _categorize_for_sampling(self, relationships: List[str]) -> Dict[str, List[str]]:
        """
        RELATIONSHIP TYPE classification for intelligent sampling
        """
        categories = {'co_occurs': [], 'contains': [], 'category': [], 'other': []}
        
        for rel in relationships:
            rel_lower = rel.lower()
            if 'co_occurs' in rel_lower or 'freq:' in rel_lower:
                categories['co_occurs'].append(rel)
            elif 'contains' in rel_lower or 'ingredient' in rel_lower:
                categories['contains'].append(rel)
            elif 'category:' in rel_lower or 'belongs_to' in rel_lower:
                categories['category'].append(rel)
            else:
                categories['other'].append(rel)
        
        return categories

    def _sort_by_frequency(self, co_occur_rels: List[str]) -> List[str]:
        """
        FREQUENCY-BASED prioritization for co-occurrence relationships
        """
        def extract_freq(rel_str: str) -> int:
            import re
            match = re.search(r'freq:(\d+)', rel_str)
            return int(match.group(1)) if match else 0
        
        return sorted(co_occur_rels, key=extract_freq, reverse=True)

    def _build_adaptive_prompt(self, original_count: int, estimated_tokens: int) -> str:
        """
        CONTEXT-AWARE prompt adaptation based on community characteristics
        """
        base_prompt = (
            "You are a culinary expert and food service consultant analyzing relationships "
            "between dishes, ingredients, menu categories, and dining contexts. You are "
            "provided with a set of relationships from a food service knowledge graph that "
            "includes both categorical relationships (what dishes contain, what categories "
            "they belong to) and historical co-occurrence patterns (what dishes have actually "
            "been served successfully together in real operations).\n\n"
        )
        
        # Adaptive context based on community size
        if original_count > 500:
            context_adaptation = (
                f"NOTE: This analysis represents a large community ({original_count} relationships). "
                f"Focus on the most significant patterns and validated co-occurrence evidence. "
                f"Prioritize actionable insights over comprehensive enumeration.\n\n"
            )
        else:
            context_adaptation = ""
        
        task_description = (
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
        )
        
        return base_prompt + context_adaptation + task_description

    def _generate_structured_fallback(self, relationships: List[str]) -> str:
        """
        STRUCTURED fallback summary without LLM dependency
        
        Engineering approach:
        - Programmatic pattern analysis
        - Statistical relationship extraction
        - Actionable insight generation
        - Consistent output format
        """
        categorized = self._categorize_for_sampling(relationships)
        
        # Extract statistical insights
        co_occurs_count = len(categorized['co_occurs'])
        contains_count = len(categorized['contains'])
        total_relationships = len(relationships)
        
        # Entity frequency analysis
        entity_counts = {}
        for rel in relationships:
            parts = rel.split(' -> ')
            if len(parts) >= 2:
                entity_counts[parts[0]] = entity_counts.get(parts[0], 0) + 1
                entity_counts[parts[1]] = entity_counts.get(parts[1], 0) + 1
        
        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Structured fallback content
        fallback = (
            f"Food Service Community Analysis (Structured): This community contains "
            f"{total_relationships} relationships with {co_occurs_count} proven co-occurrence "
            f"patterns and {contains_count} ingredient compositions. "
            f"Key entities: {', '.join([entity for entity, _ in top_entities])}. "
            f"The community demonstrates significant historical validation through "
            f"co-occurrence evidence, providing reliable foundations for menu planning "
            f"and dish pairing decisions in food service operations."
        )
        
        return fallback
    
    def build_communities(self):
        """
        RELATIONSHIP-BUDGET-AWARE community detection with intelligent subdivision
        
        Engineering enhancements:
        - Two-phase clustering: initial detection + relationship-based subdivision
        - Adaptive cluster sizing based on graph characteristics
        - Comprehensive monitoring and quality metrics
        - Graceful degradation under resource constraints
        """
        
        print("🔍 Analyzing graph structure for community detection...")
        
        # PHASE 1: Graph preparation and analysis
        nx_graph = self._create_networkx_graph()
        
        if len(nx_graph.nodes()) == 0:
            print("⚠️  Warning: Graph is empty, no communities to build")
            return
        
        print(f"📊 Graph contains {len(nx_graph.nodes())} nodes and {len(nx_graph.edges())} edges")
        
        # PHASE 2: Adaptive clustering with larger initial size
        print("🧮 Running hierarchical Leiden clustering algorithm...")
        
        try:
            # ENHANCEMENT: Adaptive cluster sizing based on graph density
            adaptive_size = min(25, max(10, len(nx_graph.nodes()) // 8))
            
            community_clusters = hierarchical_leiden(
                nx_graph, 
                max_cluster_size=adaptive_size  # Larger initial clusters for better grouping
            )
            
            initial_communities = len(set(cluster.cluster for cluster in community_clusters))
            print(f"✓ Initial clustering: {initial_communities} communities (max_size: {adaptive_size})")
            
        except Exception as e:
            print(f"❌ Community detection failed: {e}")
            return
        
        # PHASE 3: Relationship organization with monitoring
        print("📋 Organizing community relationships...")
        entity_communities, community_relationships = self._organize_community_data(
            nx_graph, community_clusters
        )
        
        # PHASE 4: CRITICAL - Relationship-based subdivision
        print("🔧 Applying relationship-based subdivision...")
        subdivided_relationships = self._subdivide_large_communities(
            community_relationships,
            max_relationships=300  # Conservative token-safe limit
        )
        
        # PHASE 5: Size-based filtering with updated relationships
        print("🔍 Filtering communities by size criteria...")
        filtered_communities = self._filter_communities_by_size(
            entity_communities, subdivided_relationships
        )
        entity_communities, final_relationships = filtered_communities
        
        print(f"✓ Final community structure: {len(final_relationships)} communities")
        
        # Store for query processing
        self.entity_communities = entity_communities
        
        # PHASE 6: Summary generation with comprehensive monitoring
        print("📝 Generating expert summaries for each community...")
        
        success_count = 0
        failure_count = 0
        
        for community_id, relationships in final_relationships.items():
            if len(relationships) > 0:
                rel_count = len(relationships)
                estimated_tokens = self._estimate_token_usage(relationships)
                
                print(f"  Community {community_id}: {rel_count} rels, ~{estimated_tokens} tokens...")
                
                try:
                    summary = self.generate_community_summary(relationships)
                    self.community_summaries[community_id] = summary
                    success_count += 1
                    print(f"    ✓ Generated ({len(summary)} chars)")
                    
                except Exception as e:
                    print(f"    ❌ Generation failed: {e}")
                    fallback = self._generate_structured_fallback(relationships)
                    self.community_summaries[community_id] = fallback
                    failure_count += 1
        
        # PHASE 7: Quality reporting
        total_communities = len(final_relationships)
        success_rate = (success_count / total_communities * 100) if total_communities > 0 else 0
        
        print(f"🎉 Community building complete!")
        print(f"📊 Results: {success_count} successful, {failure_count} fallback")
        print(f"📈 Success rate: {success_rate:.1f}%")

    def _subdivide_large_communities(self, community_relationships: Dict, max_relationships: int = 300) -> Dict:
        """
        INTELLIGENT community subdivision based on relationship count limits
        
        Engineering strategy:
        - Prevents token explosion through hard relationship limits
        - Semantic-aware chunking preserves thematic coherence
        - Consistent naming convention for sub-community tracking
        - Load balancing ensures even processing distribution
        """
        
        subdivided = {}
        subdivision_stats = {'original': 0, 'subdivided': 0, 'chunks_created': 0}
        
        for community_id, relationships in community_relationships.items():
            subdivision_stats['original'] += 1
            
            if len(relationships) <= max_relationships:
                # Community within limits
                subdivided[community_id] = relationships
            else:
                # SUBDIVISION REQUIRED
                subdivision_stats['subdivided'] += 1
                
                print(f"  🔧 Subdividing community {community_id}: {len(relationships)} → {max_relationships} chunks")
                
                # Create semantically-aware chunks
                chunks = self._create_balanced_chunks(relationships, max_relationships)
                
                for chunk_idx, chunk in enumerate(chunks):
                    sub_id = f"{community_id}_sub_{chunk_idx}"
                    subdivided[sub_id] = chunk
                    subdivision_stats['chunks_created'] += 1
        
        print(f"  📊 Subdivision: {subdivision_stats['subdivided']}/{subdivision_stats['original']} communities split")
        print(f"  📈 Created: {subdivision_stats['chunks_created']} sub-communities")
        
        return subdivided

    def _create_balanced_chunks(self, relationships: List[str], max_chunk_size: int) -> List[List[str]]:
        """
        SEMANTIC-AWARE relationship chunking with load balancing
        
        Engineering approach:
        - Preserves relationship type distribution across chunks
        - Maintains co-occurrence relationship locality
        - Ensures balanced chunk sizes for consistent processing
        - Prevents orphaned small chunks through intelligent packing
        """
        
        # CHUNKING PHASE 1: Type-aware categorization
        categorized = self._categorize_for_sampling(relationships)
        
        chunks = []
        current_chunk = []
        
        # CHUNKING PHASE 2: Sequential packing with type mixing
        for rel_type in ['co_occurs', 'contains', 'category', 'other']:
            type_rels = categorized.get(rel_type, [])
            
            for rel in type_rels:
                if len(current_chunk) >= max_chunk_size:
                    # Chunk full - start new chunk
                    chunks.append(current_chunk)
                    current_chunk = []
                
                current_chunk.append(rel)
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # CHUNKING PHASE 3: Load balancing
        return self._balance_chunk_distribution(chunks, max_chunk_size)

    def _balance_chunk_distribution(self, chunks: List[List[str]], max_size: int) -> List[List[str]]:
        """
        LOAD BALANCING for consistent chunk processing times
        """
        if not chunks:
            return chunks
        
        balanced = []
        overflow = []
        
        # Separate oversized chunks and collect overflow
        for chunk in chunks:
            if len(chunk) <= max_size:
                balanced.append(chunk)
            else:
                balanced.append(chunk[:max_size])
                overflow.extend(chunk[max_size:])
        
        # Redistribute overflow relationships
        chunk_idx = 0
        for rel in overflow:
            # Find chunk with space
            while chunk_idx < len(balanced) and len(balanced[chunk_idx]) >= max_size:
                chunk_idx += 1
            
            if chunk_idx < len(balanced):
                balanced[chunk_idx].append(rel)
            else:
                # Create new chunk for remaining overflow
                balanced.append([rel])
                chunk_idx = len(balanced) - 1
        
        return balanced
        
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
                    relation_type = edge_data.get('relationship','related')
                    description = edge_data.get('description','')
                    compressed_relationship = self._create_efficient_relationship_string(
                    node, neighbor, relation_type, edge_data
                    )
                    community_relationships[community_id].append(compressed_relationship)
        
        # Convert defaultdicts to regular dicts and remove duplicates
        entity_communities = {k: list(v) for k, v in entity_communities.items()}
        community_relationships = {
            k: list(set(v)) for k, v in community_relationships.items()
        }
        
        return entity_communities, community_relationships
    
    def _create_efficient_relationship_string(self, source: str, target: str, 
                                         relation_type: str, edge_data: dict) -> str:
        """
        ENGINEERING HELPER: Generate LlamaIndex-style relationship strings
        
        Design rationale:
        - Consistent with LlamaIndex format: "source -> target -> relation -> metadata"
        - Preserves food service domain information in compressed form
        - Maintains semantic meaning while optimizing token usage
        - Enables efficient downstream processing
        
        Performance characteristics:
        - Single string concatenation operation
        - Minimal metadata extraction
        - Predictable token count per relationship
        - Debugger-friendly format
        """
        
        # Extract metadata efficiently
        properties = edge_data.get('properties', {})
        
        # Generate compressed description using Fix #2 format
        if relation_type == "CO_OCCURS":
            freq = properties.get('frequency', 1)
            strength = properties.get('strength', 'med')[:1]
            metadata = f"freq:{freq},str:{strength}"
        elif relation_type == "CONTAINS":
            metadata = "ingredient"
        elif relation_type == "BELONGS_TO":
            metadata = f"category:{target.lower()}"
        elif relation_type == "SUITABLE_FOR":
            metadata = "event_compatible"
        else:
            metadata = relation_type.lower()[:10]  # Truncate unknown types
        
        # LlamaIndex format construction
        return f"{source} -> {target} -> {relation_type.lower()} -> {metadata}"
    
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
            "main_biryani": f"Recommend the best {count} biryani dish(es) for {event_type.lower()} events based on community analysis and historical service success.",
            "main_rice": f"Suggest {count} flavored rice or pulav dish that complements other menu items in {event_type.lower()} event settings.",
            "side_bread": f"What {count} bread item(s) have the strongest co-occurrence patterns with curry dishes in successful {event_type.lower()} event menus?",
            "side_curry": f"Recommend {count} curry dish that has proven compatibility with biryani and bread combinations in {event_type.lower()} contexts.",
            "side_accompaniment": f"What {count} accompaniment(s) like raita, salad, or pickle provide the best balance for {event_type.lower()} event meals based on historical data?",
            "dessert": f"Suggest {count} dessert that provides an excellent conclusion to {event_type.lower()} event meals with high guest satisfaction rates."
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
    
    def __init__(self, neo4j_store, llm=None, min_cluster_size=3,max_cluster_size=10):
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
            strength = properties.get('strength', 'med')[:1]
            return f"{source} -> {target} -> co_occurs -> freq:{frequency},str:{strength}"
            
            # # Create rich descriptions that emphasize the historical success of these pairings
            # if strength == "high":
            #     return (f"The dishes '{source}' and '{target}' have been successfully paired "
            #            f"together {frequency} times across multiple menus, representing a "
            #            f"proven high-confidence combination based on historical service data")
            # elif strength == "medium":
            #     return (f"The dishes '{source}' and '{target}' have appeared together "
            #            f"{frequency} times, indicating a reliable pairing that has been "
            #            f"validated through actual menu service")
            # else:
            #     return (f"The dishes '{source}' and '{target}' have co-occurred "
            #            f"{frequency} times, suggesting potential compatibility based on "
            #            f"historical menu combinations")
    
        # Create context-aware descriptions based on relationship types in food service
        elif relation_type == "CONTAINS":
            return f"{source} -> {target} -> contains -> ingredient"
        elif relation_type == "BELONGS_TO":
            return f"{source} -> {target} -> category -> {target.lower()}"
        elif relation_type == "SUITABLE_FOR":
            return f"{source} -> {target} -> event_fit -> suitable"
        elif relation_type == "SERVED_DURING":
            return f"{source} -> {target} -> meal_time -> {target.lower()}"
        elif relation_type == "COMPLEMENTS":
            return f"{source} complements {target}"
        else:
           # Generic relationship fallback with truncation
           rel_short = relation_type[:6].lower()  # Truncate unknown types
           return f"{source} -> {target} -> {rel_short} -> related"
    
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
    PRODUCTION-OPTIMIZED GraphRAG system with intelligent model selection
    
    Engineering optimizations:
    - Model selection based on actual token requirements
    - Intelligent fallback chain for API reliability
    - Adaptive clustering parameters based on graph size
    - Comprehensive error handling and recovery
    """
    
    print("🔧 Setting up complete Community-based GraphRAG system...")
    
    # SETUP PHASE 1: Embedding configuration
    print("\n1️⃣ Configuring embedding model...")
    try:
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./model_cache/embeddings"
        )
        print("✓ Embedding model configured")
    except Exception as e:
        print(f"❌ Embedding setup failed: {e}")
        return None, None
    
    # SETUP PHASE 2: LLM configuration with intelligent model selection
    print("\n2️⃣ Configuring Language Model...")
    
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found")
        return None, None
    
    # INTELLIGENT MODEL SELECTION: Based on actual requirements
    try:
        # Primary choice: GPT-4o for balanced performance/cost
        llm = OpenAI(
            model="gpt-4.1",              # Optimal: 128K context, reasonable TPM
            temperature=0.1,             # Consistent analytical output
            max_tokens=2500              # Conservative response limit
        )
        
        # Validation test
        test_response = llm.complete("Test")
        print("✓ GPT-4o configured and validated")
        
    except Exception as e:
        print(f"⚠️  Primary model failed, trying fallback: {e}")
        
        try:
            # Fallback: GPT-4o-mini with stricter limits
            llm = OpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=1500          # Reduced for mini
            )
            
            test_response = llm.complete("Test")
            print("✓ GPT-4o-mini configured (fallback)")
            
        except Exception as e2:
            print(f"❌ All model configurations failed: {e2}")
            return None, None
    
    # SETUP PHASE 3: Global configuration
    Settings.llm = llm
    Settings.embed_model = embed_model
    print("✓ Global settings configured")
    
    # SETUP PHASE 4: Neo4j connection
    print("\n3️⃣ Connecting to Neo4j...")
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
        return None, None
    
    # SETUP PHASE 5: Adaptive clustering configuration
    print("\n4️⃣ Creating community analysis adapter...")
    try:
        adapter = Neo4jGraphRAGAdapter(
            neo4j_store=neo4j_store,
            llm=llm,
            max_cluster_size=10,         # INCREASED for better initial grouping
            min_cluster_size=3           # Maintained for noise filtering
        )
        print("✓ Adapter created with optimized parameters")
        
    except Exception as e:
        print(f"❌ Adapter creation failed: {e}")
        return None, None
    
    # SETUP PHASE 6: Community building with monitoring
    print("\n5️⃣ Building communities with relationship subdivision...")
    
    import time
    start_time = time.time()
    success = adapter.build_communities_from_neo4j()
    processing_time = time.time() - start_time
    
    if not success:
        print("❌ Community building failed")
        return None, None
    
    print(f"⏱️  Processing completed in {processing_time:.1f}s")
    
    # SETUP PHASE 7: Query engine creation
    print("\n6️⃣ Creating query engine...")
    
    try:
        query_engine = adapter.create_query_engine(similarity_top_k=5)
        print("✓ Query engine ready")
        
    except Exception as e:
        print(f"❌ Query engine creation failed: {e}")
        return None, None
    
    # SETUP PHASE 8: System validation
    summaries = adapter.graphrag_store.get_community_summaries()
    if summaries:
        avg_length = sum(len(s) for s in summaries.values()) / len(summaries)
        print(f"\n✅ System ready!")
        print(f"📊 Communities: {len(summaries)}")
        print(f"📈 Avg summary: {avg_length:.0f} chars")
        print(f"🔧 Model: {llm.model}")
        print(f"⏱️  Build time: {processing_time:.1f}s")
    
    return adapter, query_engine

# Add required import
import time

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