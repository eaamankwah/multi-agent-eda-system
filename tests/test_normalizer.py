# ============================================================================
# test_normalizer.py
# ============================================================================

"""
Test Suite for JSON/XML Normalizer Agent
Tests data normalization, format detection, and nested structure flattening
"""

import pytest
import pandas as pd
import numpy as np
import json
from src.agents.normalizer_agent import JSONXMLNormalizerAgent


class TestJSONXMLNormalizerAgent:
    """Test cases for JSONXMLNormalizerAgent"""
    
    @pytest.fixture
    def agent(self):
        """Initialize normalizer agent for testing"""
        return JSONXMLNormalizerAgent()
    
    @pytest.fixture
    def sample_json_dict(self):
        """Sample JSON dictionary"""
        return {
            'name': 'John Doe',
            'age': 30,
            'email': 'john@example.com',
            'address': {
                'street': '123 Main St',
                'city': 'New York',
                'zip': '10001'
            }
        }
    
    @pytest.fixture
    def sample_json_list(self):
        """Sample JSON list"""
        return [
            {'id': 1, 'name': 'Alice', 'score': 95},
            {'id': 2, 'name': 'Bob', 'score': 87},
            {'id': 3, 'name': 'Charlie', 'score': 92}
        ]
    
    @pytest.fixture
    def sample_xml(self):
        """Sample XML string"""
        return """
        <root>
            <person>
                <name>John Doe</name>
                <age>30</age>
                <email>john@example.com</email>
            </person>
        </root>
        """
    
    @pytest.fixture
    def nested_dataframe(self):
        """DataFrame with nested structures"""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'details': [
                {'age': 25, 'city': 'NYC'},
                {'age': 30, 'city': 'LA'},
                {'age': 35, 'city': 'Chicago'}
            ]
        })
    
    # Test 1: JSON Dictionary Normalization
    def test_normalize_json_dict(self, agent, sample_json_dict):
        """Test normalizing JSON dictionary"""
        df = agent.normalize_json(sample_json_dict)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert df.shape[0] == 1  # Single record
        assert 'name' in df.columns
        assert df['name'].iloc[0] == 'John Doe'
        print("✓ JSON dictionary normalization passed")
    
    # Test 2: JSON List Normalization
    def test_normalize_json_list(self, agent, sample_json_list):
        """Test normalizing JSON list"""
        df = agent.normalize_json(sample_json_list)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 3  # Three records
        assert 'id' in df.columns
        assert 'name' in df.columns
        assert 'score' in df.columns
        assert df['name'].tolist() == ['Alice', 'Bob', 'Charlie']
        print("✓ JSON list normalization passed")
    
    # Test 3: JSON String Normalization
    def test_normalize_json_string(self, agent, sample_json_dict):
        """Test normalizing JSON string"""
        json_string = json.dumps(sample_json_dict)
        df = agent.normalize_json(json_string)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'name' in df.columns
        print("✓ JSON string normalization passed")
    
    # Test 4: XML Normalization
    def test_normalize_xml(self, agent, sample_xml):
        """Test normalizing XML data"""
        df = agent.normalize_xml(sample_xml)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        print("✓ XML normalization passed")
    
    # Test 5: Auto-detect DataFrame
    def test_auto_detect_dataframe(self, agent):
        """Test auto-detection with existing DataFrame"""
        original_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        result_df = agent.auto_detect_and_normalize(original_df)
        
        assert isinstance(result_df, pd.DataFrame)
        assert result_df.equals(original_df)
        print("✓ DataFrame auto-detection passed")
    
    # Test 6: Auto-detect JSON
    def test_auto_detect_json(self, agent, sample_json_list):
        """Test auto-detection with JSON data"""
        df = agent.auto_detect_and_normalize(sample_json_list)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 3
        print("✓ JSON auto-detection passed")
    
    # Test 7: Auto-detect JSON String
    def test_auto_detect_json_string(self, agent, sample_json_dict):
        """Test auto-detection with JSON string"""
        json_string = json.dumps(sample_json_dict)
        df = agent.auto_detect_and_normalize(json_string)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        print("✓ JSON string auto-detection passed")
    
    # Test 8: Flatten Nested Columns
    def test_flatten_nested_columns(self, agent, nested_dataframe):
        """Test flattening nested column structures"""
        flattened_df = agent.flatten_nested_columns(nested_dataframe)
        
        assert isinstance(flattened_df, pd.DataFrame)
        # Check if nested columns were expanded
        assert 'details.age' in flattened_df.columns or 'age' in flattened_df.columns
        print("✓ Nested column flattening passed")
    
    # Test 9: Empty Data Handling
    def test_normalize_empty_data(self, agent):
        """Test handling of empty data"""
        empty_dict = {}
        df = agent.normalize_json(empty_dict)
        
        assert isinstance(df, pd.DataFrame)
        # Empty dict should produce empty DataFrame or single row
        print("✓ Empty data handling passed")
    
    # Test 10: Invalid JSON Handling
    def test_invalid_json_handling(self, agent):
        """Test handling of invalid JSON"""
        invalid_json = "invalid json string {{"
        df = agent.normalize_json(invalid_json)
        
        assert isinstance(df, pd.DataFrame)
        # Should return empty DataFrame on error
        print("✓ Invalid JSON handling passed")
    
    # Test 11: Invalid XML Handling
    def test_invalid_xml_handling(self, agent):
        """Test handling of invalid XML"""
        invalid_xml = "<invalid><unclosed>"
        df = agent.normalize_xml(invalid_xml)
        
        assert isinstance(df, pd.DataFrame)
        # Should return empty DataFrame on error
        print("✓ Invalid XML handling passed")
    
    # Test 12: Complex Nested JSON
    def test_complex_nested_json(self, agent):
        """Test normalization of complex nested JSON"""
        complex_json = {
            'user': {
                'profile': {
                    'name': 'John',
                    'contacts': {
                        'email': 'john@example.com',
                        'phone': '123-456-7890'
                    }
                },
                'stats': {
                    'posts': 100,
                    'followers': 500
                }
            }
        }
        
        df = agent.normalize_json(complex_json)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        # Check if nested fields are flattened
        assert any('name' in col or 'email' in col for col in df.columns)
        print("✓ Complex nested JSON normalization passed")
    
    # Test 13: Large Dataset Performance
    def test_large_dataset_normalization(self, agent):
        """Test normalization performance with large dataset"""
        large_data = [
            {'id': i, 'value': np.random.rand(), 'category': f'cat_{i%10}'}
            for i in range(10000)
        ]
        
        df = agent.normalize_json(large_data)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 10000
        assert df.shape[1] == 3
        print("✓ Large dataset normalization passed")
    
    # Test 14: Mixed Data Types
    def test_mixed_data_types(self, agent):
        """Test handling of mixed data types in JSON"""
        mixed_data = {
            'string': 'text',
            'integer': 42,
            'float': 3.14,
            'boolean': True,
            'null': None,
            'list': [1, 2, 3],
            'dict': {'nested': 'value'}
        }
        
        df = agent.normalize_json(mixed_data)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        print("✓ Mixed data types handling passed")
    
    # Test 15: Agent Logging
    def test_agent_logging(self, agent, sample_json_list):
        """Test that agent logs are being recorded"""
        initial_log_count = len(agent.logs)
        agent.normalize_json(sample_json_list)
        
        assert len(agent.logs) > initial_log_count
        assert any('Normalizing JSON' in log for log in agent.logs)
        print("✓ Agent logging passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])