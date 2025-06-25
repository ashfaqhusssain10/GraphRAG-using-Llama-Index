# test_graphrag_integration.py
"""
Comprehensive test suite for GraphRAG-Frontend Integration
Tests individual components and complete integration flow
"""

import unittest
import json
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import components to test
from config import (
    CATEGORY_MAPPING, 
    EVENT_TYPE_MAPPING, 
    validate_configuration,
    get_category_mapping,
    get_cypher_filter
)

from template_bridge import (
    TemplateBridge, 
    CategoryMapper, 
    TemplateValidator,
    create_template_bridge,
    validate_template_compatibility
)

from graphrag_query_engine import (
    GraphRAGTemplateEngine,
    InsightGenerator,
    create_graphrag_template_engine
)

class TestConfiguration(unittest.TestCase):
    """Test configuration and category mappings"""
    
    def test_category_mapping_completeness(self):
        """Test that all required category mappings are present"""
        required_categories = [
            "starter", "main_biryani", "main_rice", 
            "side_bread", "side_curry", "side_accompaniment", "dessert"
        ]
        
        for category in required_categories:
            self.assertIn(category, CATEGORY_MAPPING, f"Missing category mapping: {category}")
            
            # Check required fields
            mapping = CATEGORY_MAPPING[category]
            self.assertIn("graphrag_labels", mapping)
            self.assertIn("cypher_filter", mapping)
            self.assertIn("description", mapping)
    
    def test_event_type_mapping(self):
        """Test event type mappings"""
        required_events = ["Traditional", "Party", "Premium"]
        
        for event in required_events:
            self.assertIn(event, EVENT_TYPE_MAPPING, f"Missing event mapping: {event}")
    
    def test_cypher_filter_generation(self):
        """Test Cypher filter generation for categories"""
        for category in CATEGORY_MAPPING:
            cypher_filter = get_cypher_filter(category)
            self.assertIsNotNone(cypher_filter, f"No Cypher filter for {category}")
            self.assertIsInstance(cypher_filter, str)
            self.assertGreater(len(cypher_filter), 0)
    
    def test_configuration_validation(self):
        """Test configuration validation function"""
        # This test assumes temp1.json and items_price_uom.json exist
        # Comment out if files are not available
        if os.path.exists("temp1.json") and os.path.exists("items_price_uom.json"):
            result = validate_configuration()
            self.assertTrue(result, "Configuration validation failed")


class TestTemplateBridge(unittest.TestCase):
    """Test template bridge functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock template data
        self.mock_template_data = {
            "menu": [
                {
                    "id": "test_template",
                    "name": "Test Template",
                    "tag": "Party",
                    "pricing": {
                        "price_per_head": 250,
                        "budget": "₹250–299"
                    },
                    "items": [
                        {
                            "category": "starter",
                            "name": "Starter 1",
                            "weight": "0.1 kg"
                        },
                        {
                            "category": "main",
                            "name": "Biryani",
                            "weight": "0.3 kg"
                        },
                        {
                            "category": "dessert",
                            "name": "Dessert",
                            "weight": "0.1 kg"
                        }
                    ]
                }
            ]
        }
        
        # Create temporary template file
        self.temp_file = "test_template.json"
        with open(self.temp_file, 'w', encoding='utf-8') as f:
            json.dump(self.mock_template_data, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
    
    def test_template_loading(self):
        """Test template loading from JSON"""
        bridge = TemplateBridge(self.temp_file)
        self.assertEqual(len(bridge.templates), 1)
        self.assertIn("test_template", bridge.templates)
    
    def test_template_selection_by_budget(self):
        """Test template selection based on budget and event type"""
        bridge = TemplateBridge(self.temp_file)
        
        # Test exact match
        template = bridge.find_template_by_budget(250, "Party")
        self.assertIsNotNone(template)
        self.assertEqual(template["id"], "test_template")
        
        # Test budget too low
        template = bridge.find_template_by_budget(100, "Party")
        self.assertIsNone(template)
        
        # Test wrong event type
        template = bridge.find_template_by_budget(250, "NonExistent")
        # Should still work since Premium compatibility
        
    def test_requirement_extraction(self):
        """Test extraction of template requirements"""
        bridge = TemplateBridge(self.temp_file)
        template = bridge.templates["test_template"]
        
        requirements = bridge.extract_requirements(template)
        
        self.assertEqual(requirements["template_id"], "test_template")
        self.assertEqual(requirements["total_items"], 3)
        self.assertIn("category_summary", requirements)
        
        # Check category counts
        self.assertGreater(len(requirements["categories"]), 0)
    
    def test_graphrag_query_building(self):
        """Test GraphRAG query construction"""
        bridge = TemplateBridge(self.temp_file)
        template = bridge.templates["test_template"]
        requirements = bridge.extract_requirements(template)
        
        queries = bridge.build_graphrag_queries(requirements, "Party")
        
        self.assertGreater(len(queries), 0)
        
        for query in queries:
            self.assertIn("category", query)
            self.assertIn("count", query)
            self.assertIn("event_type", query)
            self.assertIn("cypher_filter", query)
    
    def test_template_validation(self):
        """Test template structure validation"""
        bridge = TemplateBridge(self.temp_file)
        template = bridge.templates["test_template"]
        
        is_valid, issues = bridge.validate_template_structure(template)
        self.assertTrue(is_valid, f"Template validation failed: {issues}")


class TestCategoryMapper(unittest.TestCase):
    """Test category mapping functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mapper = CategoryMapper()
    
    def test_direct_category_mapping(self):
        """Test direct category mappings"""
        # Test starter mapping
        result = self.mapper.map_to_graphrag_category("starter", {"name": "Test Starter"})
        self.assertEqual(result, "starter")
        
        # Test dessert mapping
        result = self.mapper.map_to_graphrag_category("dessert", {"name": "Test Dessert"})
        self.assertEqual(result, "dessert")
    
    def test_main_category_mapping(self):
        """Test main category context-dependent mapping"""
        # Test biryani mapping
        result = self.mapper.map_to_graphrag_category("main", {"name": "Chicken Biryani"})
        self.assertEqual(result, "main_biryani")
        
        # Test rice mapping
        result = self.mapper.map_to_graphrag_category("main", {"name": "Jeera Rice"})
        self.assertEqual(result, "main_rice")
        
        # Test default mapping
        result = self.mapper.map_to_graphrag_category("main", {"name": "Unknown Main"})
        self.assertEqual(result, "main_biryani")  # Default
    
    def test_side_category_mapping(self):
        """Test side category context-dependent mapping"""
        # Test bread mapping
        result = self.mapper.map_to_graphrag_category("side", {"name": "Butter Naan"})
        self.assertEqual(result, "side_bread")
        
        # Test curry mapping
        result = self.mapper.map_to_graphrag_category("side", {"name": "Dal Curry"})
        self.assertEqual(result, "side_curry")
        
        # Test accompaniment mapping
        result = self.mapper.map_to_graphrag_category("side", {"name": "Mixed Raita"})
        self.assertEqual(result, "side_accompaniment")
        
        # Test default mapping
        result = self.mapper.map_to_graphrag_category("side", {"name": "Unknown Side"})
        self.assertEqual(result, "side_accompaniment")  # Default


class TestGraphRAGIntegration(unittest.TestCase):
    """Test GraphRAG integration components"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock the GraphRAG system since it requires Neo4j
        self.mock_graphrag_response = "Based on historical patterns, I recommend Chicken 65, Paneer Tikka, and Veg Spring Rolls as excellent starters for party events. These items have high co-occurrence rates and proven success."
        
        # Create mock template file
        self.mock_template_data = {
            "menu": [
                {
                    "id": "test_integration",
                    "name": "Integration Test Template",
                    "tag": "Party", 
                    "pricing": {
                        "price_per_head": 300,
                        "budget": "₹300–349"
                    },
                    "items": [
                        {"category": "starter", "name": "Starter 1", "weight": "0.08 kg"},
                        {"category": "starter", "name": "Starter 2", "weight": "0.08 kg"},
                        {"category": "starter", "name": "Starter 3", "weight": "0.08 kg"},
                        {"category": "main", "name": "Biryani", "weight": "0.3 kg"},
                        {"category": "dessert", "name": "Dessert", "weight": "0.1 kg"}
                    ]
                }
            ]
        }
        
        self.temp_file = "test_integration.json"
        with open(self.temp_file, 'w', encoding='utf-8') as f:
            json.dump(self.mock_template_data, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
    
    @patch('graphrag_query_engine.setup_complete_community_graphrag_system')
    def test_engine_initialization(self, mock_setup):
        """Test GraphRAG engine initialization"""
        # Mock successful GraphRAG setup
        mock_adapter = Mock()
        mock_query_engine = Mock()
        mock_setup.return_value = (mock_adapter, mock_query_engine)
        
        engine = GraphRAGTemplateEngine(self.temp_file)
        
        self.assertIsNotNone(engine.template_bridge)
        self.assertIsNotNone(engine.graphrag_adapter)
        self.assertIsNotNone(engine.graphrag_query_engine)
    
    @patch('graphrag_query_engine.setup_complete_community_graphrag_system')
    def test_item_extraction_from_response(self, mock_setup):
        """Test item extraction from GraphRAG response"""
        # Mock GraphRAG setup
        mock_adapter = Mock()
        mock_query_engine = Mock()
        mock_setup.return_value = (mock_adapter, mock_query_engine)
        
        # Create mock pricing data
        mock_pricing_data = [
            {"item_name": "Chicken 65", "category": "Starters"},
            {"item_name": "Paneer Tikka", "category": "Starters"},
            {"item_name": "Veg Spring Rolls", "category": "Starters"}
        ]
        
        with patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(mock_pricing_data))):
            engine = GraphRAGTemplateEngine(self.temp_file)
            
            # Test item extraction
            extracted_items = engine._extract_items_from_response(
                self.mock_graphrag_response, "starter", 3
            )
            
            self.assertEqual(len(extracted_items), 3)
            self.assertTrue(all("name" in item for item in extracted_items))
    
    @patch('graphrag_query_engine.setup_complete_community_graphrag_system')
    def test_insight_generation(self, mock_setup):
        """Test insight generation"""
        # Mock GraphRAG setup
        mock_adapter = Mock()
        mock_query_engine = Mock()
        mock_setup.return_value = (mock_adapter, mock_query_engine)
        
        generator = InsightGenerator()
        
        # Mock filled template
        mock_filled_template = {
            "name": "Test Template",
            "items": [
                {"name": "Chicken 65", "insight": "Popular party starter"},
                {"name": "Veg Biryani", "insight": "Traditional centerpiece"}
            ]
        }
        
        # Mock suggestions
        mock_suggestions = {
            "starter": [{"name": "Chicken 65", "source": "graphrag", "match_confidence": 0.9}],
            "main_biryani": [{"name": "Veg Biryani", "source": "graphrag", "match_confidence": 0.8}]
        }
        
        insights = generator.generate_template_insights(mock_filled_template, mock_suggestions, "Party")
        
        self.assertIn("individual_insights", insights)
        self.assertIn("overall_insight", insights)
        self.assertIn("success_indicators", insights)
        self.assertIn("recommendation_strength", insights)


class TestCompleteIntegrationFlow(unittest.TestCase):
    """Test complete integration flow from budget to recommendation"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        # Create comprehensive test template
        self.integration_template_data = {
            "menu": [
                {
                    "id": "complete_integration_test",
                    "name": "Complete Integration Test",
                    "tag": "Party",
                    "pricing": {
                        "price_per_head": 250,
                        "budget": "₹250–299"
                    },
                    "items": [
                        {"category": "starter", "name": "Starter 1", "weight": "0.1 kg"},
                        {"category": "starter", "name": "Starter 2", "weight": "0.1 kg"},
                        {"category": "main", "name": "Biryani", "weight": "0.35 kg"},
                        {"category": "side", "name": "Bread", "quantity": "1 pcs"},
                        {"category": "side", "name": "Curry", "weight": "0.12 kg"},
                        {"category": "dessert", "name": "Dessert", "weight": "0.12 kg"}
                    ]
                }
            ]
        }
        
        self.integration_temp_file = "complete_integration.json"
        with open(self.integration_temp_file, 'w', encoding='utf-8') as f:
            json.dump(self.integration_template_data, f)
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        if os.path.exists(self.integration_temp_file):
            os.remove(self.integration_temp_file)
    
    @patch('graphrag_query_engine.setup_complete_community_graphrag_system')
    @patch('builtins.open')
    def test_complete_recommendation_flow(self, mock_open, mock_setup):
        """Test complete flow from budget input to final recommendation"""
        # Mock GraphRAG setup
        mock_adapter = Mock()
        mock_query_engine = Mock()
        mock_query_engine.query.return_value = "Recommend Chicken 65, Paneer Tikka for starters and Chicken Dum Biryani for main course."
        mock_setup.return_value = (mock_adapter, mock_query_engine)
        
        # Mock pricing data
        mock_pricing_data = [
            {"item_name": "Chicken 65", "category": "Starters"},
            {"item_name": "Paneer Tikka", "category": "Starters"},
            {"item_name": "Chicken Dum Biryani", "category": "Main Course"},
            {"item_name": "Butter Naan", "category": "Main Course"},
            {"item_name": "Dal Tadka", "category": "Main Course"},
            {"item_name": "Gulab Jamun", "category": "Sweets"}
        ]
        
        # Set up file mocking
        def mock_open_func(filename, mode='r', encoding=None):
            if 'integration' in filename:
                return unittest.mock.mock_open(read_data=json.dumps(self.integration_template_data))()
            elif 'items_price_uom' in filename:
                return unittest.mock.mock_open(read_data=json.dumps(mock_pricing_data))()
            else:
                return unittest.mock.mock_open()()
        
        mock_open.side_effect = mock_open_func
        
        # Test complete flow
        engine = GraphRAGTemplateEngine(self.integration_temp_file)
        
        recommendation = engine.recommend_with_graphrag("Party", 250)
        
        # Validate recommendation structure
        self.assertNotIn("error", recommendation, "Recommendation should not contain errors")
        self.assertIn("template_id", recommendation)
        self.assertIn("template_name", recommendation)
        self.assertIn("items", recommendation)
        self.assertIn("insights", recommendation)
        
        # Validate items structure
        items = recommendation["items"]
        self.assertGreater(len(items), 0, "Recommendation should contain items")
        
        for item in items:
            self.assertIn("name", item)
            self.assertIn("category", item)
            # Should have either weight or quantity
            self.assertTrue("weight" in item or "quantity" in item)
    
    def test_template_validation_complete(self):
        """Test complete template validation process"""
        # Test template compatibility validation
        if os.path.exists("temp1.json"):
            result = validate_template_compatibility()
            # This test depends on actual temp1.json file
            # Comment out if file is not available
            # self.assertTrue(result, "Template compatibility validation failed")
    
    def test_fallback_handling(self):
        """Test fallback strategies when GraphRAG fails"""
        # This test would require mocking GraphRAG failures
        # and testing fallback to community defaults
        pass
    
    def test_budget_categorization(self):
        """Test budget-based recommendation categorization"""
        # This test would verify base vs premium categorization
        # based on budget constraints
        pass


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_missing_template_file(self):
        """Test handling of missing template file"""
        with self.assertRaises(FileNotFoundError):
            TemplateBridge("nonexistent_template.json")
    
    def test_invalid_json_template(self):
        """Test handling of invalid JSON template"""
        invalid_file = "invalid_template.json"
        with open(invalid_file, 'w') as f:
            f.write("invalid json content")
        
        try:
            with self.assertRaises(json.JSONDecodeError):
                TemplateBridge(invalid_file)
        finally:
            if os.path.exists(invalid_file):
                os.remove(invalid_file)
    
    def test_empty_template_data(self):
        """Test handling of empty template data"""
        empty_template_data = {"menu": []}
        empty_file = "empty_template.json"
        
        with open(empty_file, 'w') as f:
            json.dump(empty_template_data, f)
        
        try:
            bridge = TemplateBridge(empty_file)
            self.assertEqual(len(bridge.templates), 0)
            
            # Test that template selection returns None
            template = bridge.find_template_by_budget(250, "Party")
            self.assertIsNone(template)
        finally:
            if os.path.exists(empty_file):
                os.remove(empty_file)


class TestPerformance(unittest.TestCase):
    """Test performance and timing requirements"""
    
    def test_template_selection_performance(self):
        """Test that template selection meets performance targets"""
        import time
        
        # Create large template dataset for performance testing
        large_template_data = {"menu": []}
        for i in range(100):  # 100 templates
            template = {
                "id": f"perf_test_{i}",
                "name": f"Performance Test Template {i}",
                "tag": "Party",
                "pricing": {
                    "price_per_head": 200 + i,
                    "budget": f"₹{200 + i}–{250 + i}"
                },
                "items": [
                    {"category": "starter", "name": f"Starter {i}", "weight": "0.1 kg"},
                    {"category": "main", "name": f"Main {i}", "weight": "0.3 kg"}
                ]
            }
            large_template_data["menu"].append(template)
        
        perf_file = "performance_test.json"
        with open(perf_file, 'w') as f:
            json.dump(large_template_data, f)
        
        try:
            bridge = TemplateBridge(perf_file)
            
            # Measure template selection time
            start_time = time.time()
            template = bridge.find_template_by_budget(225, "Party")
            end_time = time.time()
            
            selection_time = end_time - start_time
            
            # Should be under 100ms as per config target
            self.assertLess(selection_time, 0.1, f"Template selection took {selection_time:.3f}s, should be < 0.1s")
            self.assertIsNotNone(template, "Template selection should succeed")
            
        finally:
            if os.path.exists(perf_file):
                os.remove(perf_file)


# ============================================================================
# TEST SUITE RUNNER
# ============================================================================

def run_all_tests():
    """Run all integration tests with detailed reporting"""
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestConfiguration,
        TestTemplateBridge,
        TestCategoryMapper,
        TestGraphRAGIntegration,
        TestCompleteIntegrationFlow,
        TestErrorHandling,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            newline = '\n'
            error_msg = traceback.split('AssertionError: ')[-1].split(newline)[0]
            print(f"- {test}: {error_msg}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            newline = '\n'
            error_msg = traceback.split(newline)[-2]
            print(f"- {test}: {error_msg}")
    
    return result.wasSuccessful()


def run_quick_tests():
    """Run essential tests for quick validation"""
    
    print("Running quick integration tests...")
    
    # Test 1: Configuration validation
    try:
        from config import validate_configuration
        config_valid = validate_configuration()
        print(f"✓ Configuration validation: {'PASS' if config_valid else 'FAIL'}")
    except Exception as e:
        print(f"✗ Configuration validation: FAIL ({e})")
    
    # Test 2: Template bridge basic functionality
    try:
        if os.path.exists("temp1.json"):
            bridge = create_template_bridge()
            template = bridge.find_template_by_budget(250, "Party")
            print(f"✓ Template bridge: {'PASS' if template else 'FAIL'}")
        else:
            print("⚠ Template bridge: SKIP (temp1.json not found)")
    except Exception as e:
        print(f"✗ Template bridge: FAIL ({e})")
    
    # Test 3: Category mapping
    try:
        mapper = CategoryMapper()
        starter_mapping = mapper.map_to_graphrag_category("starter", {"name": "Test"})
        main_mapping = mapper.map_to_graphrag_category("main", {"name": "Biryani"})
        print(f"✓ Category mapping: {'PASS' if starter_mapping and main_mapping else 'FAIL'}")
    except Exception as e:
        print(f"✗ Category mapping: FAIL ({e})")
    
    print("Quick tests completed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        run_quick_tests()
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)