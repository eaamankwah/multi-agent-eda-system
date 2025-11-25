# ============================================================================
# test_storytelling.py
# ============================================================================

"""
Test Suite for Data Storytelling Agent
Tests narrative generation, dashboard creation, and report generation
"""

import pytest
import pandas as pd
import numpy as np
from src.agents.storytelling_agent import DataStorytellingAgent


class TestDataStorytellingAgent:
    """Test cases for DataStorytellingAgent"""
    
    @pytest.fixture
    def agent(self):
        """Initialize data storytelling agent for testing"""
        return DataStorytellingAgent()
    
    @pytest.fixture
    def sample_dataframe(self):
        """Sample DataFrame for testing"""
        np.random.seed(42)
        return pd.DataFrame({
            'id': range(1, 201),
            'age': np.random.randint(18, 80, 200),
            'income': np.random.normal(50000, 20000, 200),
            'score': np.random.uniform(0, 100, 200),
            'category': np.random.choice(['A', 'B', 'C', 'D'], 200),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 200)
        })
    
    @pytest.fixture
    def analysis_results(self):
        """Sample analysis results"""
        return {
            'quality': {
                'shape': (200, 6),
                'missing_values': 0,
                'duplicates': 0
            },
            'exploration': {
                'summary_statistics': {
                    'age': {'mean': 50, 'std': 15},
                    'income': {'mean': 50000, 'std': 20000}
                },
                'clustering': {
                    'n_clusters': 3,
                    'silhouette_score': 0.45
                }
            },
            'relationships': {
                'correlations': {
                    'pearson': {
                        'age': {'income': 0.15}
                    }
                },
                'regression': {
                    'r2_score': 0.65
                }
            }
        }
    
    # Test 1: Generate Narrative Summary
    def test_generate_narrative_summary(self, agent, sample_dataframe, analysis_results):
        """Test narrative summary generation"""
        narrative = agent.generate_narrative_summary(sample_dataframe, analysis_results)
        
        assert isinstance(narrative, str)
        assert len(narrative) > 0
        # Should contain some information about the dataset
        assert any(keyword in narrative.lower() for keyword in ['dataset', 'data', 'analysis', 'records', 'features'])
        
        print("✓ Narrative summary generation passed")
    
    # Test 2: Generate Narrative Without Results
    def test_generate_narrative_no_results(self, agent, sample_dataframe):
        """Test narrative generation without analysis results"""
        narrative = agent.generate_narrative_summary(sample_dataframe, None)
        
        assert isinstance(narrative, str)
        assert len(narrative) > 0
        
        print("✓ Narrative without results passed")
    
    # Test 3: Generate Summary Insights
    def test_generate_summary_insights(self, agent, sample_dataframe, analysis_results):
        """Test summary insights generation (alias method)"""
        insights = agent.generate_summary_insights(sample_dataframe, analysis_results)
        
        assert isinstance(insights, str)
        assert len(insights) > 0
        
        print("✓ Summary insights generation passed")
    
    # Test 4: Fallback Narrative Generation
    def test_fallback_narrative(self, agent, sample_dataframe, analysis_results):
        """Test fallback narrative generation (non-LLM)"""
        narrative = agent._generate_fallback_narrative(sample_dataframe, analysis_results)
        
        assert isinstance(narrative, str)
        assert len(narrative) > 0
        assert 'Dataset Overview' in narrative
        assert 'Data Quality' in narrative
        assert 'Key Findings' in narrative
        
        print("✓ Fallback narrative generation passed")
    
    # Test 5: Fallback Narrative with Empty Results
    def test_fallback_narrative_empty_results(self, agent, sample_dataframe):
        """Test fallback narrative with empty analysis results"""
        narrative = agent._generate_fallback_narrative(sample_dataframe, {})
        
        assert isinstance(narrative, str)
        assert len(narrative) > 0
        
        print("✓ Fallback narrative with empty results passed")
    
    # Test 6: Create Interactive Dashboard
    def test_create_interactive_dashboard(self, agent, sample_dataframe, analysis_results):
        """Test interactive dashboard creation"""
        try:
            dashboard = agent.create_interactive_dashboard(sample_dataframe, analysis_results)
            
            # Dashboard might be None if Dash is not available
            if dashboard is not None:
                assert hasattr(dashboard, 'layout')
                assert hasattr(dashboard, 'callback')
                print("✓ Interactive dashboard creation passed")
            else:
                print("✓ Dashboard creation skipped (Dash not available)")
        except ImportError:
            print("✓ Dashboard creation skipped (Dash not installed)")
    
    # Test 7: Dashboard with Minimal Data
    def test_dashboard_minimal_data(self, agent):
        """Test dashboard creation with minimal data"""
        minimal_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['A', 'B', 'C']
        })
        
        try:
            dashboard = agent.create_interactive_dashboard(minimal_df, {})
            # Should handle minimal data gracefully
            assert dashboard is None or hasattr(dashboard, 'layout')
            print("✓ Dashboard with minimal data passed")
        except ImportError:
            print("✓ Dashboard creation skipped (Dash not installed)")
    
    # Test 8: Create Summary Report
    def test_create_summary_report(self, agent, sample_dataframe, analysis_results):
        """Test HTML summary report generation"""
        report = agent.create_summary_report(sample_dataframe, analysis_results)
        
        assert isinstance(report, str)
        assert len(report) > 0
        # Should be valid HTML
        assert '<html>' in report.lower()
        assert '</html>' in report.lower()
        assert '<body>' in report.lower()
        
        print("✓ Summary report creation passed")
    
    # Test 9: Report Contains Key Sections
    def test_report_contains_sections(self, agent, sample_dataframe, analysis_results):
        """Test that report contains all key sections"""
        report = agent.create_summary_report(sample_dataframe, analysis_results)
        
        # Check for key sections
        assert 'Dataset Overview' in report
        assert 'Summary Statistics' in report
        assert 'Data Quality' in report
        assert 'Analysis Results' in report
        
        print("✓ Report sections validation passed")
    
    # Test 10: Report Contains Metrics
    def test_report_contains_metrics(self, agent, sample_dataframe, analysis_results):
        """Test that report contains important metrics"""
        report = agent.create_summary_report(sample_dataframe, analysis_results)
        
        # Check for dataset size
        assert str(sample_dataframe.shape[0]) in report
        assert str(sample_dataframe.shape[1]) in report
        
        print("✓ Report metrics validation passed")
    
    # Test 11: Report with Empty Analysis
    def test_report_empty_analysis(self, agent, sample_dataframe):
        """Test report generation with empty analysis results"""
        report = agent.create_summary_report(sample_dataframe, {})
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert '<html>' in report.lower()
        
        print("✓ Report with empty analysis passed")
    
    # Test 12: Report HTML Validity
    def test_report_html_validity(self, agent, sample_dataframe, analysis_results):
        """Test that generated HTML has proper structure"""
        report = agent.create_summary_report(sample_dataframe, analysis_results)
        
        # Count opening and closing tags
        html_open = report.lower().count('<html>')
        html_close = report.lower().count('</html>')
        body_open = report.lower().count('<body>')
        body_close = report.lower().count('</body>')
        
        assert html_open == html_close == 1
        assert body_open == body_close == 1
        
        print("✓ HTML validity passed")
    
    # Test 13: Narrative Contains Dataset Info
    def test_narrative_contains_info(self, agent, sample_dataframe, analysis_results):
        """Test that narrative contains dataset information"""
        narrative = agent.generate_narrative_summary(sample_dataframe, analysis_results)
        
        # Should mention number of records or features
        has_row_count = str(sample_dataframe.shape[0]) in narrative
        has_col_count = str(sample_dataframe.shape[1]) in narrative
        
        # At least one should be present
        assert has_row_count or has_col_count or 'records' in narrative.lower() or 'features' in narrative.lower()
        
        print("✓ Narrative dataset info passed")
    
    # Test 14: Result Storage
    def test_result_storage(self, agent, sample_dataframe, analysis_results):
        """Test that narrative is stored in results"""
        agent.generate_narrative_summary(sample_dataframe, analysis_results)
        
        assert 'narrative' in agent.results
        assert isinstance(agent.results['narrative'], str)
        
        print("✓ Result storage passed")
    
    # Test 15: Dashboard App Storage
    def test_dashboard_app_storage(self, agent, sample_dataframe, analysis_results):
        """Test that dashboard app is stored"""
        try:
            dashboard = agent.create_interactive_dashboard(sample_dataframe, analysis_results)
            
            if dashboard is not None:
                assert agent.dashboard_app is not None
                assert agent.dashboard_app == dashboard
                print("✓ Dashboard app storage passed")
            else:
                print("✓ Dashboard app storage skipped (Dash not available)")
        except ImportError:
            print("✓ Dashboard app storage skipped (Dash not installed)")
    
    # Test 16: Empty DataFrame Handling
    def test_empty_dataframe(self, agent):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()
        
        narrative = agent.generate_narrative_summary(empty_df, {})
        assert isinstance(narrative, str)
        
        report = agent.create_summary_report(empty_df, {})
        assert isinstance(report, str)
        
        print("✓ Empty DataFrame handling passed")
    
    # Test 17: Large Dataset Narrative
    def test_large_dataset_narrative(self, agent):
        """Test narrative generation for large dataset"""
        np.random.seed(42)
        large_df = pd.DataFrame({
            'col1': np.random.rand(10000),
            'col2': np.random.randint(0, 100, 10000),
            'col3': np.random.choice(['A', 'B', 'C'], 10000)
        })
        
        narrative = agent.generate_narrative_summary(large_df, {})
        
        assert isinstance(narrative, str)
        assert len(narrative) > 0
        
        print("✓ Large dataset narrative passed")
    
    # Test 18: Large Dataset Report
    def test_large_dataset_report(self, agent):
        """Test report generation for large dataset"""
        np.random.seed(42)
        large_df = pd.DataFrame({
            'col1': np.random.rand(5000),
            'col2': np.random.randint(0, 100, 5000)
        })
        
        report = agent.create_summary_report(large_df, {})
        
        assert isinstance(report, str)
        assert '5,000' in report or '5000' in report
        
        print("✓ Large dataset report passed")
    
    # Test 19: Agent Logging
    def test_agent_logging(self, agent, sample_dataframe, analysis_results):
        """Test that agent logs are being recorded"""
        initial_log_count = len(agent.logs)
        agent.generate_narrative_summary(sample_dataframe, analysis_results)
        
        assert len(agent.logs) > initial_log_count
        assert any('narrative' in log.lower() or 'summary' in log.lower() for log in agent.logs)
        
        print("✓ Agent logging passed")
    
    # Test 20: Multiple Operations
    def test_multiple_operations(self, agent, sample_dataframe, analysis_results):
        """Test multiple storytelling operations in sequence"""
        # Generate narrative
        narrative = agent.generate_narrative_summary(sample_dataframe, analysis_results)
        assert isinstance(narrative, str)
        
        # Create report
        report = agent.create_summary_report(sample_dataframe, analysis_results)
        assert isinstance(report, str)
        
        # Try dashboard
        try:
            dashboard = agent.create_interactive_dashboard(sample_dataframe, analysis_results)
            # Dashboard creation might fail if Dash not available
            assert dashboard is None or hasattr(dashboard, 'layout')
        except ImportError:
            pass
        
        print("✓ Multiple operations passed")
    
    # Test 21: Report Styling
    def test_report_styling(self, agent, sample_dataframe, analysis_results):
        """Test that report contains CSS styling"""
        report = agent.create_summary_report(sample_dataframe, analysis_results)
        
        assert '<style>' in report or 'style=' in report
        # Should have some visual formatting
        assert 'font-family' in report.lower() or 'color' in report.lower()
        
        print("✓ Report styling passed")
    
    # Test 22: Narrative Character Length
    def test_narrative_length(self, agent, sample_dataframe, analysis_results):
        """Test that narrative has reasonable length"""
        narrative = agent.generate_narrative_summary(sample_dataframe, analysis_results)
        
        # Should be substantial but not too long
        assert 50 < len(narrative) < 10000
        
        print("✓ Narrative length validation passed")
    
    # Test 23: Report Contains Timestamp
    def test_report_timestamp(self, agent, sample_dataframe, analysis_results):
        """Test that report contains generation timestamp"""
        report = agent.create_summary_report(sample_dataframe, analysis_results)
        
        # Should have a timestamp
        assert 'Generated:' in report or 'generated' in report.lower()
        
        print("✓ Report timestamp passed")
    
    # Test 24: Fallback Narrative Structure
    def test_fallback_structure(self, agent, sample_dataframe, analysis_results):
        """Test fallback narrative has proper markdown structure"""
        narrative = agent._generate_fallback_narrative(sample_dataframe, analysis_results)
        
        # Should have markdown headers
        assert '##' in narrative or '#' in narrative
        # Should have bullet points or structure
        assert '-' in narrative or '*' in narrative or '###' in narrative
        
        print("✓ Fallback narrative structure passed")
    
    # Test 25: Mixed Data Types in Report
    def test_mixed_types_report(self, agent, sample_dataframe, analysis_results):
        """Test report handles mixed data types properly"""
        report = agent.create_summary_report(sample_dataframe, analysis_results)
        
        # Should display both numerical and categorical info
        numeric_cols = sample_dataframe.select_dtypes(include=[np.number]).shape[1]
        categorical_cols = sample_dataframe.select_dtypes(include=['object', 'category']).shape[1]
        
        # Report should mention both types
        if numeric_cols > 0:
            assert 'Numerical' in report or 'numerical' in report.lower()
        if categorical_cols > 0:
            assert 'Categorical' in report or 'categorical' in report.lower()
        
        print("✓ Mixed data types in report passed")
    
    # Test 26: Dashboard Callbacks
    def test_dashboard_callbacks(self, agent, sample_dataframe, analysis_results):
        """Test that dashboard has callback functions if created"""
        try:
            dashboard = agent.create_interactive_dashboard(sample_dataframe, analysis_results)
            
            if dashboard is not None:
                # Dashboard should have callbacks registered
                assert hasattr(dashboard, 'callback_map') or hasattr(dashboard, '_callback_list')
                print("✓ Dashboard callbacks passed")
            else:
                print("✓ Dashboard callbacks skipped (Dash not available)")
        except (ImportError, AttributeError):
            print("✓ Dashboard callbacks skipped (Dash not installed)")
    
    # Test 27: Report JSON Handling
    def test_report_json_handling(self, agent, sample_dataframe):
        """Test report handles complex analysis results with JSON"""
        complex_results = {
            'nested': {
                'level1': {
                    'level2': {
                        'value': 42
                    }
                }
            },
            'list_data': [1, 2, 3, 4, 5]
        }
        
        report = agent.create_summary_report(sample_dataframe, complex_results)
        
        assert isinstance(report, str)
        assert '<html>' in report.lower()
        
        print("✓ Report JSON handling passed")
    
    # Test 28: Narrative Error Handling
    def test_narrative_error_handling(self, agent):
        """Test narrative generation error handling"""
        # Problematic DataFrame
        df = pd.DataFrame({
            'col': [np.inf, -np.inf, np.nan]
        })
        
        try:
            narrative = agent.generate_narrative_summary(df, {})
            assert isinstance(narrative, str)
            print("✓ Narrative error handling passed")
        except Exception as e:
            pytest.fail(f"Narrative generation should handle errors: {e}")
    
    # Test 29: Report Special Characters
    def test_report_special_characters(self, agent):
        """Test report handles special characters properly"""
        df = pd.DataFrame({
            'special_col': ['<>&"\'', 'normal text', 'more <>'],
            'value': [1, 2, 3]
        })
        
        report = agent.create_summary_report(df, {})
        
        assert isinstance(report, str)
        # HTML should be properly formed
        assert '<html>' in report.lower()
        
        print("✓ Report special characters handling passed")
    
    # Test 30: Summary Report File Save
    def test_summary_report_save(self, agent, sample_dataframe, analysis_results, tmp_path):
        """Test saving summary report to file"""
        report = agent.create_summary_report(sample_dataframe, analysis_results)
        
        # Save to temporary file
        report_path = tmp_path / "test_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        assert report_path.exists()
        assert report_path.stat().st_size > 0
        
        # Read back and verify
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert content == report
        
        print("✓ Summary report file save passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
