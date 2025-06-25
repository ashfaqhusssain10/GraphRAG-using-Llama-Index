# config.py
"""
Configuration file for GraphRAG-Frontend Integration
Contains category mappings, configuration parameters, and system settings
"""

import os
from typing import Dict, List, Any, Optional

# ============================================================================
# CATEGORY MAPPING CONFIGURATION
# ============================================================================

# Maps JSON template categories to GraphRAG node labels and queries
CATEGORY_MAPPING = {
    "starter": {
        "graphrag_labels": ["Starters"],
        "cypher_filter": "n.node_type = 'Starters'",
        "description": "Appetizers and starter items for events",
        "fallback_categories": ["Snacks"]  # Backup if not enough starters
    },
    
    "main_biryani": {
        "graphrag_labels": ["Main Course"],
        "cypher_filter": "n.node_type = 'Main Course' AND toLower(n.name) CONTAINS 'biryani'",
        "description": "Biryani dishes as main course centerpiece",
        "fallback_categories": ["Main Course"]  # Broader main course if needed
    },
    
    "main_rice": {
        "graphrag_labels": ["Main Course"],
        "cypher_filter": "n.node_type = 'Main Course' AND (toLower(n.name) CONTAINS 'rice' OR toLower(n.name) CONTAINS 'pulav' OR toLower(n.name) CONTAINS 'pulao')",
        "description": "Flavored rice dishes and pulav varieties",
        "fallback_categories": ["Main Course"]
    },
    
    "side_bread": {
        "graphrag_labels": ["Main Course"],
        "cypher_filter": "n.node_type = 'Main Course' AND (toLower(n.name) CONTAINS 'roti' OR toLower(n.name) CONTAINS 'naan' OR toLower(n.name) CONTAINS 'chapati' OR toLower(n.name) CONTAINS 'phulka' OR toLower(n.name) CONTAINS 'paratha')",
        "description": "Bread items that accompany curries",
        "fallback_categories": ["Main Course"]
    },
    
    "side_curry": {
        "graphrag_labels": ["Main Course"],
        "cypher_filter": "n.node_type = 'Main Course' AND toLower(n.name) CONTAINS 'curry'",
        "description": "Curry dishes that pair with bread and rice",
        "fallback_categories": ["Main Course"]
    },
    
    "side_accompaniment": {
        "graphrag_labels": ["Sides & Accompaniments"],
        "cypher_filter": "n.node_type = 'Sides & Accompaniments'",
        "description": "Raita, salads, pickles, and other accompaniments",
        "fallback_categories": ["Sides & Accompaniments"]
    },
    
    "dessert": {
        "graphrag_labels": ["Desserts", "Sweets"],
        "cypher_filter": "(n.node_type = 'Desserts' OR n.node_type = 'Sweets')",
        "description": "Sweet dishes and desserts for meal completion",
        "fallback_categories": ["Desserts", "Sweets"]
    }
}

# Event type mapping between frontend and GraphRAG
EVENT_TYPE_MAPPING = {
    "Traditional": {
        "graphrag_event_types": ["Traditional"],
        "description": "Traditional and cultural events with classic food preferences",
        "preference_weight": {
            "traditional_items": 1.0,
            "fusion_items": 0.3,
            "comfort_food": 0.8
        }
    },
    
    "Party": {
        "graphrag_event_types": ["Party"],
        "description": "Party and celebration events with variety and engagement focus",
        "preference_weight": {
            "traditional_items": 0.6,
            "fusion_items": 1.0,
            "comfort_food": 0.7
        }
    },
    
    "Premium": {
        "graphrag_event_types": ["Party", "Traditional"],  # Premium is budget tier, not event type
        "description": "Higher budget events with premium item selection",
        "preference_weight": {
            "traditional_items": 0.8,
            "fusion_items": 0.9,
            "comfort_food": 0.6
        }
    }
}

# ============================================================================
# GRAPHRAG CONFIGURATION
# ============================================================================

GRAPHRAG_CONFIG = {
    # Query performance settings
    "max_retry_attempts": 3,
    "query_timeout_seconds": 10,
    "batch_query_size": 5,
    
    # Item selection criteria
    "co_occurrence_min_frequency": 2,  # Minimum times items appeared together
    "community_relevance_threshold": 0.6,
    "fallback_similarity_threshold": 0.7,
    
    # Insight generation
    "community_insight_max_length": 150,
    "individual_insight_max_length": 50,
    "include_success_rates": True,
    "include_co_occurrence_data": True,
    
    # Neo4j connection (matches your existing setup)
    "neo4j_config": {
        "url": "bolt://127.0.0.1:7687",
        "username": "neo4j",
        "password": "Ashfaq8790",
        "encrypted": False
    }
}

# ============================================================================
# TEMPLATE CONFIGURATION
# ============================================================================

TEMPLATE_CONFIG = {
    # Template processing rules
    "template_file": "temp1.json",
    "strict_category_matching": True,  # Must match exact categories
    "allow_substitutions": False,      # No category substitutions allowed
    "require_exact_counts": True,      # Must return exact item counts
    
    # Budget handling
    "premium_budget_tolerance": 0.2,   # 20% above budget for premium recommendations
    "base_budget_tolerance": 0.05,     # 5% tolerance for base recommendations
    
    # Fallback strategies
    "enable_fallback_queries": True,
    "max_fallback_attempts": 2,
    "prefer_community_defaults": True
}

# ============================================================================
# PRICING CONFIGURATION
# ============================================================================

PRICING_CONFIG = {
    "price_file": "items_price_uom.json",
    "default_use_peak_pricing": True,
    "price_calculation_precision": 2,  # Decimal places
    
    # Unit conversion settings
    "weight_conversions": {
        "kg_to_g": 1000,
        "g_to_kg": 0.001,
        "default_unit": "kg"
    },
    
    # Quantity parsing
    "quantity_patterns": {
        "weight": r"([\d.]+)\s*(kg|g|gm)",
        "pieces": r"([\d.]+)\s*(pcs|pieces|piece)",
        "volume": r"([\d.]+)\s*(ml|litre|l)"
    }
}

# ============================================================================
# INSIGHT GENERATION CONFIGURATION
# ============================================================================

INSIGHT_CONFIG = {
    # Insight types and their weights
    "insight_types": {
        "co_occurrence": {
            "weight": 0.4,
            "min_frequency": 2,
            "templates": {
                "high": "High co-occurrence (freq: {frequency})",
                "medium": "Proven pairing (freq: {frequency})",
                "low": "Compatible combination (freq: {frequency})"
            }
        },
        
        "community_analysis": {
            "weight": 0.3,
            "templates": {
                "traditional": "Traditional comfort food pattern",
                "modern": "Contemporary fusion approach",
                "balanced": "Balanced traditional-modern combination"
            }
        },
        
        "event_suitability": {
            "weight": 0.3,
            "templates": {
                "Traditional": "Classic choice for traditional events",
                "Party": "Popular party selection",
                "Premium": "Premium event preference"
            }
        }
    },
    
    # Overall insight generation
    "success_rate_calculation": True,
    "include_culinary_reasoning": True,
    "condensed_format": True
}

# ============================================================================
# ERROR HANDLING CONFIGURATION
# ============================================================================

ERROR_CONFIG = {
    # Retry strategies
    "max_query_retries": 3,
    "retry_backoff_factor": 1.5,
    "enable_graceful_degradation": True,
    
    # Fallback behaviors
    "fallback_to_community_defaults": True,
    "allow_partial_recommendations": True,
    "log_fallback_usage": True,
    
    # Error logging
    "log_failed_queries": True,
    "log_insufficient_results": True,
    "log_budget_calculation_errors": True
}

# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================

PERFORMANCE_CONFIG = {
    # Caching settings
    "enable_caching": True,
    "cache_ttl_seconds": 3600,  # 1 hour
    "cache_community_summaries": True,
    "cache_category_mappings": True,
    
    # Query optimization
    "batch_similar_queries": True,
    "optimize_cypher_queries": True,
    "limit_results_per_query": 20,
    
    # Response time targets
    "target_response_times": {
        "template_identification": 0.1,  # 100ms
        "graphrag_queries": 2.0,         # 2 seconds
        "budget_processing": 0.2,        # 200ms
        "complete_recommendation": 3.0   # 3 seconds
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_category_mapping(category: str) -> Optional[Dict[str, Any]]:
    """Get category mapping configuration for a specific category."""
    return CATEGORY_MAPPING.get(category)

def get_event_type_mapping(event_type: str) -> Optional[Dict[str, Any]]:
    """Get event type mapping configuration."""
    return EVENT_TYPE_MAPPING.get(event_type)

def get_cypher_filter(category: str) -> Optional[str]:
    """Get Cypher filter query for a specific category."""
    mapping = CATEGORY_MAPPING.get(category)
    return mapping.get("cypher_filter") if mapping else None

def get_fallback_categories(category: str) -> List[str]:
    """Get fallback categories for a specific category."""
    mapping = CATEGORY_MAPPING.get(category)
    return mapping.get("fallback_categories", []) if mapping else []

def validate_configuration() -> bool:
    """Validate that all required configuration is present and valid."""
    required_files = [
        TEMPLATE_CONFIG["template_file"],
        PRICING_CONFIG["price_file"]
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Warning: Required file not found: {file_path}")
            return False
    
    # Validate category mappings
    for category, config in CATEGORY_MAPPING.items():
        required_keys = ["graphrag_labels", "cypher_filter", "description"]
        if not all(key in config for key in required_keys):
            print(f"Warning: Invalid category mapping for {category}")
            return False
    
    return True

# ============================================================================
# RUNTIME CONFIGURATION VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("Validating GraphRAG-Frontend Integration Configuration...")
    
    if validate_configuration():
        print("✓ Configuration validation passed")
        print(f"✓ Found {len(CATEGORY_MAPPING)} category mappings")
        print(f"✓ Found {len(EVENT_TYPE_MAPPING)} event type mappings")
        print("✓ All required files present")
    else:
        print("✗ Configuration validation failed")
        print("Please check file paths and configuration completeness")
    
    # Display key configuration summary
    print("\n--- Configuration Summary ---")
    print(f"Template file: {TEMPLATE_CONFIG['template_file']}")
    print(f"Pricing file: {PRICING_CONFIG['price_file']}")
    print(f"Neo4j URL: {GRAPHRAG_CONFIG['neo4j_config']['url']}")
    print(f"Max retry attempts: {GRAPHRAG_CONFIG['max_retry_attempts']}")
    print(f"Premium budget tolerance: {TEMPLATE_CONFIG['premium_budget_tolerance']*100}%")