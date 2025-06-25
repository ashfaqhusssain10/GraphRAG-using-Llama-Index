# graphrag_query_engine.py
"""
GraphRAG Query Engine for Template Integration
Main engine that connects template requirements with GraphRAG system
"""

import re
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import logging
import random

# Import your existing GraphRAG components
try:
    from ex1 import (
        FoodServiceGraphRAGStore, 
        FoodServiceGraphRAGQueryEngine,
        Neo4jGraphRAGAdapter,
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
        """Initialize the existing GraphRAG system from ex1.py"""
        try:
            logger.info("Initializing GraphRAG system...")
            
            # Use your existing setup function
            adapter, query_engine = setup_complete_community_graphrag_system()
            
            if adapter and query_engine:
                self.graphrag_adapter = adapter
                self.graphrag_query_engine = query_engine
                logger.info("GraphRAG system initialized successfully")
            else:
                raise Exception("Failed to initialize GraphRAG system")
                
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
            
            logger.info(f"Querying GraphRAG for {count_needed} {category} items")
            
            try:
                # Execute category-specific query
                category_suggestions = self._query_category_items(query_spec, event_type)
                
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
    
    def _query_category_items(self, query_spec: Dict[str, Any], event_type: str) -> List[Dict[str, Any]]:
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
        
        # Build GraphRAG query string
        query_text = self._build_category_query_text(category, event_type, count)
        
        # Execute query using existing GraphRAG system
        try:
            raw_response = self.graphrag_query_engine.query(query_text)
            
            # Extract specific items from GraphRAG response
            extracted_items = self._extract_items_from_response(raw_response, category, count)
            
            # Enhance with co-occurrence data
            enhanced_items = self._enhance_with_co_occurrence_data(extracted_items, event_type)
            
            return enhanced_items
            
        except Exception as e:
            logger.error(f"GraphRAG query execution failed for {category}: {e}")
            raise
    
    def _build_category_query_text(self, category: str, event_type: str, count: int) -> str:
        """
        Build natural language query for GraphRAG based on category
        
        Args:
            category: GraphRAG category (e.g., "starter", "main_biryani")
            event_type: Event type context
            count: Number of items needed
        
        Returns:
            Natural language query string
        """
        category_queries = {
            "starter": f"What are the best {count} starter items for {event_type.lower()} events based on historical co-occurrence patterns?",
            
            "main_biryani": f"Recommend {count} biryani dish that works well for {event_type.lower()} events with high success rates.",
            
            "main_rice": f"Suggest {count} flavored rice or pulav dish suitable for {event_type.lower()} events.",
            
            "side_bread": f"What {count} bread item pairs well with curry dishes in {event_type.lower()} event menus?",
            
            "side_curry": f"Recommend {count} curry dish that complements biryani and bread in {event_type.lower()} settings.",
            
            "side_accompaniment": f"What {count} accompaniment (raita, salad, or pickle) works best for {event_type.lower()} events?",
            
            "dessert": f"Suggest {count} dessert that provides a good ending for {event_type.lower()} event meals."
        }
        
        query = category_queries.get(category, f"Recommend {count} items from {category} category for {event_type.lower()} events")
        
        logger.debug(f"Built query for {category}: {query}")
        return query
    
    def _extract_items_from_response(self, response: str, category: str, count: int) -> List[Dict[str, Any]]:
        """
        Extract specific item names from GraphRAG response text
        
        Args:
            response: Raw text response from GraphRAG
            category: Category being queried
            count: Number of items expected
        
        Returns:
            List of extracted items with basic info
        """
        # This method extracts item names from the natural language response
        # and matches them against known items in the pricing database
        
        extracted_items = []
        
        # Load pricing data to validate item names
        try:
            with open("items_price_uom.json", 'r', encoding='utf-8') as f:
                pricing_data = json.load(f)
                known_items = {item["item_name"].lower(): item for item in pricing_data}
        except Exception as e:
            logger.error(f"Could not load pricing data: {e}")
            known_items = {}
        
        # Extract item names using multiple strategies
        response_lower = response.lower()
        
        # Strategy 1: Look for quoted items or items in lists
        patterns = [
            r'"([^"]+)"',  # Quoted items
            r"'([^']+)'",  # Single quoted items
            r'•\s*([^\n•]+)',  # Bullet points
            r'\d+\.\s*([^\n\d]+)',  # Numbered lists
            r'-\s*([^\n-]+)',  # Dash lists
        ]
        
        found_names = set()
        for pattern in patterns:
            matches = re.findall(pattern, response_lower)
            for match in matches:
                clean_name = match.strip()
                
                # Try to match with known items
                for known_name, item_data in known_items.items():
                    if self._is_item_match(clean_name, known_name, category):
                        if known_name not in found_names:
                            extracted_items.append({
                                "name": item_data["item_name"],  # Use proper case
                                "category": item_data["category"],
                                "source": "graphrag_extraction",
                                "match_confidence": self._calculate_match_confidence(clean_name, known_name)
                            })
                            found_names.add(known_name)
                            break
        
        # Strategy 2: Direct name matching for common items
        if len(extracted_items) < count:
            for known_name, item_data in known_items.items():
                if known_name in response_lower and known_name not in found_names:
                    if self._is_category_appropriate(item_data["category"], category):
                        extracted_items.append({
                            "name": item_data["item_name"],
                            "category": item_data["category"],
                            "source": "direct_match",
                            "match_confidence": 1.0
                        })
                        found_names.add(known_name)
                        
                        if len(extracted_items) >= count:
                            break
        
        # Sort by confidence and return top items
        extracted_items.sort(key=lambda x: x["match_confidence"], reverse=True)
        return extracted_items[:count]
    
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
        """
        Enhance item suggestions with co-occurrence insights
        
        Args:
            items: List of basic item suggestions
            event_type: Event type for context
        
        Returns:
            Enhanced items with co-occurrence insights
        """
        enhanced_items = []
        
        for item in items:
            enhanced_item = item.copy()
            
            # Generate insight based on co-occurrence data
            insight = self._generate_item_insight(item["name"], event_type)
            enhanced_item["insight"] = insight
            enhanced_item["co_occurrence_score"] = self._get_co_occurrence_score(item["name"], event_type)
            
            enhanced_items.append(enhanced_item)
        
        return enhanced_items
    
    def _generate_item_insight(self, item_name: str, event_type: str) -> str:
        """Generate condensed insight for an item"""
        # This would ideally query the actual co-occurrence data
        # For now, generate contextual insights based on item type and event
        
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
    
    def _get_co_occurrence_score(self, item_name: str, event_type: str) -> float:
        """Get co-occurrence score for item (placeholder implementation)"""
        # This would query actual co-occurrence data from your graph
        # For now, return a reasonable score based on item type
        return random.uniform(0.6, 0.9)
    
    def _handle_insufficient_suggestions(self, query_spec: Dict[str, Any], current_suggestions: List[Dict[str, Any]], event_type: str) -> List[Dict[str, Any]]:
        """
        Handle cases where GraphRAG doesn't return enough suggestions
        
        Args:
            query_spec: Original query specification
            current_suggestions: Currently found suggestions
            event_type: Event type context
        
        Returns:
            Completed list with fallback suggestions
        """
        category = query_spec["category"]
        count_needed = query_spec["count"]
        count_current = len(current_suggestions)
        count_missing = count_needed - count_current
        
        logger.info(f"Finding {count_missing} fallback suggestions for {category}")
        
        # Strategy 1: Use fallback categories
        fallback_categories = query_spec.get("fallback_categories", [])
        fallback_suggestions = []
        
        for fallback_category in fallback_categories:
            if len(fallback_suggestions) >= count_missing:
                break
                
            fallback_query = self._build_fallback_query(fallback_category, event_type, count_missing)
            try:
                fallback_items = self._query_fallback_items(fallback_query, fallback_category, count_missing)
                fallback_suggestions.extend(fallback_items)
            except Exception as e:
                logger.warning(f"Fallback query failed for {fallback_category}: {e}")
        
        # Strategy 2: Use community defaults if still insufficient
        if len(fallback_suggestions) < count_missing:
            community_defaults = self._get_community_defaults(category, count_missing - len(fallback_suggestions))
            fallback_suggestions.extend(community_defaults)
        
        # Mark fallback items
        for item in fallback_suggestions:
            item["source"] = "fallback"
            item["insight"] = f"Community backup choice"
        
        # Combine and return
        complete_suggestions = current_suggestions + fallback_suggestions[:count_missing]
        return complete_suggestions
    
    def _get_community_defaults(self, category: str, count: int) -> List[Dict[str, Any]]:
        """Get community default items for a category"""
        # Hardcoded safe defaults for each category
        defaults = {
            "starter": [
                {"name": "Veg Samosa", "category": "Snacks"},
                {"name": "Chicken 65", "category": "Starters"},
                {"name": "Paneer Tikka", "category": "Starters"}
            ],
            "main_biryani": [
                {"name": "Veg Biryani", "category": "Main Course"},
                {"name": "Chicken Biryani", "category": "Main Course"}
            ],
            "main_rice": [
                {"name": "Jeera Rice", "category": "Main Course"},
                {"name": "Pulihora", "category": "Main Course"}
            ],
            "side_bread": [
                {"name": "Chapati", "category": "Main Course"},
                {"name": "Butter Naan", "category": "Main Course"}
            ],
            "side_curry": [
                {"name": "Dal Tadka", "category": "Main Course"},
                {"name": "Mixed Vegetable Curry", "category": "Main Course"}
            ],
            "side_accompaniment": [
                {"name": "Raita", "category": "Sides & Accompaniments"},
                {"name": "Pickle", "category": "Sides & Accompaniments"}
            ],
            "dessert": [
                {"name": "Gulab Jamun", "category": "Sweets"},
                {"name": "Double Ka Meetha", "category": "Sweets"}
            ]
        }
        
        category_defaults = defaults.get(category, [])
        selected_defaults = category_defaults[:count]
        
        # Add metadata
        for item in selected_defaults:
            item["source"] = "community_default"
            item["match_confidence"] = 0.8
        
        return selected_defaults
    
    def _fill_template_with_suggestions(self, template: Dict[str, Any], requirements: Dict[str, Any], suggestions: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Fill template structure with GraphRAG suggestions
        
        Args:
            template: Original template
            requirements: Template requirements
            suggestions: GraphRAG suggestions by category
        
        Returns:
            Template with specific item names filled in
        """
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
                    "quantity": original_item.get("quantity"),
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
    
    def _handle_no_template_found(self, event_type: str, budget: int) -> Dict[str, Any]:
        """Handle case where no template matches budget/event type"""
        return {
            "error": "No matching template found",
            "event_type": event_type,
            "budget": budget,
            "suggestion": "Please adjust budget or try different event type",
            "available_ranges": self._get_available_budget_ranges(event_type)
        }
    
    def _handle_recommendation_error(self, event_type: str, budget: int, error_msg: str) -> Dict[str, Any]:
        """Handle general recommendation errors"""
        return {
            "error": "Recommendation generation failed",
            "event_type": event_type,
            "budget": budget,
            "error_details": error_msg,
            "fallback": "Please try again or contact support"
        }
    
    def _get_available_budget_ranges(self, event_type: str) -> List[str]:
        """Get available budget ranges for an event type"""
        ranges = []
        for template in self.template_bridge.templates.values():
            if self.template_bridge._is_event_compatible(template["tag"], event_type):
                ranges.append(template["pricing"]["budget"])
        return sorted(set(ranges))


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

def create_graphrag_template_engine(template_file: str = None) -> GraphRAGTemplateEngine:
    """
    Factory function to create GraphRAG Template Engine
    
    Args:
        template_file: Optional custom template file path
    
    Returns:
        Configured GraphRAGTemplateEngine instance
    """
    return GraphRAGTemplateEngine(template_file)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing GraphRAG Template Engine...")
    
    try:
        # Create engine
        engine = create_graphrag_template_engine()
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