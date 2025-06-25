# template_bridge.py
"""
Template Bridge for GraphRAG-Frontend Integration
Handles template parsing, category extraction, and GraphRAG query construction
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import logging

from config import (
    CATEGORY_MAPPING, 
    EVENT_TYPE_MAPPING, 
    TEMPLATE_CONFIG,
    get_category_mapping,
    get_cypher_filter,
    get_fallback_categories
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemplateBridge:
    """
    Bridge between JSON templates and GraphRAG system
    Handles template parsing, category mapping, and query construction
    """
    
    def __init__(self, template_file: str = None):
        """
        Initialize Template Bridge
        
        Args:
            template_file: Path to template JSON file (default from config)
        """
        self.template_file = template_file or TEMPLATE_CONFIG["template_file"]
        self.templates = self.load_templates()
        self.category_mapper = CategoryMapper()
        
        logger.info(f"TemplateBridge initialized with {len(self.templates)} templates")
    
    def load_templates(self) -> Dict[str, Any]:
        """Load templates from JSON file"""
        try:
            with open(self.template_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {template["id"]: template for template in data["menu"]}
        except FileNotFoundError:
            logger.error(f"Template file not found: {self.template_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in template file: {e}")
            raise
    
    def find_template_by_budget(self, budget: int, event_type: str) -> Optional[Dict[str, Any]]:
        """
        Find matching template based on budget and event type
        
        Args:
            budget: Budget per head
            event_type: "Traditional", "Party", or "Premium"
        
        Returns:
            Template dictionary or None if no match
        """
        matching_templates = []
        
        for template_id, template in self.templates.items():
            template_price = template["pricing"]["price_per_head"]
            template_tag = template["tag"]
            
            # Check if budget fits within template range
            budget_fits = budget >= template_price
            
            # For premium tier, also check upper bound from budget range
            if "–" in template["pricing"]["budget"]:
                price_range = template["pricing"]["budget"].replace("₹", "").replace("+", "")
                if "–" in price_range:
                    min_price, max_price = price_range.split("–")
                    budget_fits = int(min_price) <= budget <= int(max_price)
                else:
                    # Handle "₹500+" case
                    min_price = int(price_range)
                    budget_fits = budget >= min_price
            
            # Check event type compatibility
            event_compatible = self._is_event_compatible(template_tag, event_type)
            
            if budget_fits and event_compatible:
                matching_templates.append((template_price, template))
        
        if not matching_templates:
            logger.warning(f"No template found for budget {budget} and event type {event_type}")
            return None
        
        # Return template with highest price that fits budget (best value)
        matching_templates.sort(key=lambda x: x[0], reverse=True)
        selected_template = matching_templates[0][1]
        
        logger.info(f"Selected template: {selected_template['name']} for budget {budget}")
        return selected_template
    
    def _is_event_compatible(self, template_tag: str, event_type: str) -> bool:
        """Check if template tag is compatible with event type"""
        # Premium templates can be used for any event type (budget tier)
        if template_tag == "Premium":
            return True
        
        # Direct match
        if template_tag == event_type:
            return True
        
        # Cross-compatibility rules
        compatibility_matrix = {
            "Traditional": ["Traditional"],
            "Party": ["Party", "Premium"],
            "Premium": ["Traditional", "Party", "Premium"]
        }
        
        return template_tag in compatibility_matrix.get(event_type, [])
    
    def extract_requirements(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract specific category requirements from template
        
        Args:
            template: Template dictionary
        
        Returns:
            Dictionary with category requirements and metadata
        """
        requirements = {
            "template_id": template["id"],
            "template_name": template["name"],
            "event_tag": template["tag"],
            "budget_info": template["pricing"],
            "categories": defaultdict(list),
            "total_items": len(template["items"]),
            "category_counts": Counter(),
            "weights_and_quantities": {}
        }
        
        for item in template["items"]:
            category = item["category"]
            
            # Map JSON category to GraphRAG category
            graphrag_category = self.category_mapper.map_to_graphrag_category(category, item)
            
            item_info = {
                "json_category": category,
                "graphrag_category": graphrag_category,
                "weight": item.get("weight"),
                "quantity": item.get("quantity"),
                "item_metadata": item
            }
            
            requirements["categories"][graphrag_category].append(item_info)
            requirements["category_counts"][graphrag_category] += 1
            
            # Store weight/quantity info for later use
            key = f"{graphrag_category}_{len(requirements['categories'][graphrag_category])}"
            requirements["weights_and_quantities"][key] = {
                "weight": item.get("weight"),
                "quantity": item.get("quantity")
            }
        
        # Add summary statistics
        requirements["category_summary"] = dict(requirements["category_counts"])
        
        logger.info(f"Extracted requirements for {template['name']}: {dict(requirements['category_counts'])}")
        return requirements
    
    def build_graphrag_queries(self, requirements: Dict[str, Any], event_type: str) -> List[Dict[str, Any]]:
        """
        Build specific GraphRAG queries for each category requirement
        
        Args:
            requirements: Template requirements from extract_requirements()
            event_type: Event type for context
        
        Returns:
            List of query specifications for GraphRAG
        """
        queries = []
        
        for graphrag_category, items in requirements["categories"].items():
            count_needed = len(items)
            
            query_spec = {
                "category": graphrag_category,
                "count": count_needed,
                "event_type": event_type,
                "cypher_filter": get_cypher_filter(graphrag_category),
                "fallback_categories": get_fallback_categories(graphrag_category),
                "weights_quantities": [
                    {
                        "weight": item.get("weight"),
                        "quantity": item.get("quantity")
                    }
                    for item in items
                ],
                "context": {
                    "template_name": requirements["template_name"],
                    "template_tag": requirements["event_tag"],
                    "total_items": requirements["total_items"]
                }
            }
            
            queries.append(query_spec)
        
        logger.info(f"Built {len(queries)} GraphRAG queries for template {requirements['template_name']}")
        return queries
    
    def validate_template_structure(self, template: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate template structure and identify potential issues
        
        Args:
            template: Template dictionary
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        required_fields = ["id", "name", "tag", "pricing", "items"]
        for field in required_fields:
            if field not in template:
                issues.append(f"Missing required field: {field}")
        
        # Check pricing structure
        if "pricing" in template:
            pricing = template["pricing"]
            if "price_per_head" not in pricing or "budget" not in pricing:
                issues.append("Invalid pricing structure")
        
        # Check items structure
        if "items" in template:
            for i, item in enumerate(template["items"]):
                if "category" not in item or "name" not in item:
                    issues.append(f"Item {i} missing required fields")
                
                # Check that each item has either weight or quantity
                if not item.get("weight") and not item.get("quantity"):
                    issues.append(f"Item {i} missing weight or quantity specification")
        
        # Check category mappings exist
        for item in template.get("items", []):
            category = item.get("category")
            if category:
                mapped_category = self.category_mapper.map_to_graphrag_category(category, item)
                if not mapped_category:
                    issues.append(f"No GraphRAG mapping for category: {category}")
        
        is_valid = len(issues) == 0
        return is_valid, issues


class CategoryMapper:
    """
    Handles mapping between JSON template categories and GraphRAG categories
    """
    
    def __init__(self):
        """Initialize category mapper with mapping rules"""
        self.mapping_rules = self._build_mapping_rules()
        logger.info("CategoryMapper initialized with mapping rules")
    
    def _build_mapping_rules(self) -> Dict[str, Any]:
        """Build category mapping rules from configuration"""
        rules = {}
        
        # Direct category mappings
        rules["starter"] = "starter"
        rules["dessert"] = "dessert"
        
        # Context-dependent mappings for "main" and "side"
        rules["context_mappings"] = {
            "main": {
                "biryani_keywords": ["biryani"],
                "rice_keywords": ["rice", "pulav", "pulao"],
                "default": "main_biryani"  # Default to biryani if unclear
            },
            "side": {
                "bread_keywords": ["bread", "roti", "naan", "chapati", "phulka", "paratha"],
                "curry_keywords": ["curry"],
                "accompaniment_keywords": ["raita", "salad", "pickle", "chutney"],
                "default": "side_accompaniment"  # Default to accompaniment if unclear
            }
        }
        
        return rules
    
    def map_to_graphrag_category(self, json_category: str, item_context: Dict[str, Any]) -> str:
        """
        Map JSON category to GraphRAG category using context
        
        Args:
            json_category: Category from JSON template
            item_context: Item dictionary with name and other metadata
        
        Returns:
            GraphRAG category string
        """
        json_category = json_category.lower()
        item_name = item_context.get("name", "").lower()
        
        # Direct mappings
        if json_category in ["starter"]:
            return "starter"
        elif json_category in ["dessert"]:
            return "dessert"
        
        # Context-dependent mappings
        elif json_category == "main":
            return self._map_main_category(item_name, item_context)
        elif json_category == "side":
            return self._map_side_category(item_name, item_context)
        
        # Fallback
        logger.warning(f"Unknown category mapping for: {json_category}")
        return json_category  # Return as-is if no mapping found
    
    def _map_main_category(self, item_name: str, item_context: Dict[str, Any]) -> str:
        """Map 'main' category based on item context"""
        context_rules = self.mapping_rules["context_mappings"]["main"]
        
        # Check for biryani
        if any(keyword in item_name for keyword in context_rules["biryani_keywords"]):
            return "main_biryani"
        
        # Check for rice/pulav
        if any(keyword in item_name for keyword in context_rules["rice_keywords"]):
            return "main_rice"
        
        # Default to biryani (most common main in templates)
        return context_rules["default"]
    
    def _map_side_category(self, item_name: str, item_context: Dict[str, Any]) -> str:
        """Map 'side' category based on item context"""
        context_rules = self.mapping_rules["context_mappings"]["side"]
        
        # Check for bread items
        if any(keyword in item_name for keyword in context_rules["bread_keywords"]):
            return "side_bread"
        
        # Check for curry items
        if any(keyword in item_name for keyword in context_rules["curry_keywords"]):
            return "side_curry"
        
        # Check for accompaniments
        if any(keyword in item_name for keyword in context_rules["accompaniment_keywords"]):
            return "side_accompaniment"
        
        # Default to accompaniment
        return context_rules["default"]
    
    def get_all_mappings(self) -> Dict[str, str]:
        """Get all possible category mappings for reference"""
        return {
            "starter": "starter",
            "main (biryani)": "main_biryani", 
            "main (rice/pulav)": "main_rice",
            "side (bread)": "side_bread",
            "side (curry)": "side_curry",
            "side (accompaniment)": "side_accompaniment",
            "dessert": "dessert"
        }


class TemplateValidator:
    """
    Validates templates and their GraphRAG compatibility
    """
    
    @staticmethod
    def validate_all_templates(template_bridge: TemplateBridge) -> Dict[str, Any]:
        """
        Validate all loaded templates
        
        Args:
            template_bridge: TemplateBridge instance
        
        Returns:
            Validation report
        """
        report = {
            "total_templates": len(template_bridge.templates),
            "valid_templates": 0,
            "invalid_templates": 0,
            "issues": {},
            "category_coverage": Counter(),
            "event_type_coverage": Counter()
        }
        
        for template_id, template in template_bridge.templates.items():
            is_valid, issues = template_bridge.validate_template_structure(template)
            
            if is_valid:
                report["valid_templates"] += 1
                
                # Analyze category coverage
                requirements = template_bridge.extract_requirements(template)
                for category in requirements["category_summary"]:
                    report["category_coverage"][category] += 1
                
                # Track event type coverage
                report["event_type_coverage"][template["tag"]] += 1
            else:
                report["invalid_templates"] += 1
                report["issues"][template_id] = issues
        
        return report


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_template_bridge(template_file: str = None) -> TemplateBridge:
    """
    Factory function to create TemplateBridge instance
    
    Args:
        template_file: Optional custom template file path
    
    Returns:
        Configured TemplateBridge instance
    """
    return TemplateBridge(template_file)

def validate_template_compatibility() -> bool:
    """
    Validate that templates are compatible with GraphRAG categories
    
    Returns:
        True if compatible, False otherwise
    """
    try:
        bridge = create_template_bridge()
        validator = TemplateValidator()
        report = validator.validate_all_templates(bridge)
        
        if report["invalid_templates"] > 0:
            logger.error(f"Found {report['invalid_templates']} invalid templates")
            for template_id, issues in report["issues"].items():
                logger.error(f"Template {template_id}: {issues}")
            return False
        
        logger.info(f"All {report['valid_templates']} templates are valid")
        logger.info(f"Category coverage: {dict(report['category_coverage'])}")
        logger.info(f"Event type coverage: {dict(report['event_type_coverage'])}")
        
        return True
    
    except Exception as e:
        logger.error(f"Template validation failed: {e}")
        return False


# ============================================================================
# TESTING AND DEBUGGING
# ============================================================================

if __name__ == "__main__":
    print("Testing Template Bridge...")
    
    try:
        # Create bridge and validate
        bridge = create_template_bridge()
        print(f"✓ Loaded {len(bridge.templates)} templates")
        
        # Test template selection
        test_budget = 300
        test_event = "Party"
        template = bridge.find_template_by_budget(test_budget, test_event)
        
        if template:
            print(f"✓ Found template for ₹{test_budget} {test_event}: {template['name']}")
            
            # Test requirement extraction
            requirements = bridge.extract_requirements(template)
            print(f"✓ Extracted requirements: {requirements['category_summary']}")
            
            # Test query building
            queries = bridge.build_graphrag_queries(requirements, test_event)
            print(f"✓ Built {len(queries)} GraphRAG queries")
            
            # Display query details
            for i, query in enumerate(queries):
                print(f"  Query {i+1}: {query['category']} (count: {query['count']})")
        
        # Validate all templates
        if validate_template_compatibility():
            print("✓ All templates are compatible with GraphRAG")
        else:
            print("✗ Template compatibility issues found")
            
    except Exception as e:
        print(f"✗ Template Bridge testing failed: {e}")
        import traceback
        traceback.print_exc()