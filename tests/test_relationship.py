# ============================================================================
# test_relationship.py
# ============================================================================
"""
Test Suite for Relationship Discovery Agent
Tests correlation analysis, regression, mutual information, and relationship visualization
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.agents.relationship_agent import RelationshipDiscoveryAgent


class TestRelationshipDiscoveryAgent:
    """Test cases for RelationshipDiscoveryAgent"""
    
    @pytest.fixture
    def agent(self):
        """Initialize relationship discovery agent for testing"""
        return RelationshipDiscoveryAgent()
    
    @pytest.fixture
    def correlated_dataframe(self):
        """DataFrame with known correlations"""
        np.random.seed(42)
        x = np.random.normal(50, 10, 200)
        y = 2 * x + np.random.normal(0, 5, 200)
        z = -1.5 * x + np.random.normal(0, 5, 200)
        w = np.random.normal(100, 15, 200)
        
        return pd.DataFrame({
            'feature_x': x,
            'feature_y': y,
            'feature_z': z,
            'independent': w
        })
    
    @pytest.fixture
    def mixed_dataframe(self):
        """DataFrame with mixed numerical and categorical data"""
        np.random.seed(42)
        return pd.DataFrame({
            'numerical1': np.random.normal(50, 10, 200),
            'numerical2': np.random.normal(100, 20, 200),
            'numerical3': np.random.exponential(5, 200),
            'category': np.random.choice(['A', 'B', 'C'], 200),
            'region': np.random.choice(['North', 'South'], 200)
        })
    
    @pytest.fixture
    def regression_dataframe(self):
        """DataFrame suitable for regression analysis"""
        np.random.seed(42)
        x1 = np.random.normal(50, 10, 200)
        x2 = np.random.normal(30, 5, 200)
        x3 = np.random.uniform(0, 100, 200)
        target = 3 * x1 + 2 * x2 - 0.5 * x3 + np.random.normal(0, 10, 200)
        
        return pd.DataFrame({
            'feature1': x1,
            'feature2': x2,
            'feature3': x3,
            'target': target
        })
    
    def test_compute_correlations_pearson(self, agent, correlated_dataframe):
        """Test Pearson correlation computation"""
        correlations = agent.compute_correlations(correlated_dataframe, plot_heatmap=False)
        
        assert isinstance(correlations, dict)
        assert 'pearson' in correlations
        assert 'spearman' in correlations
        for col in correlated_dataframe.columns:
            assert col in correlations['pearson']
        print("✓ Pearson correlation computation passed")
    
    def test_compute_correlations_with_heatmap(self, agent, correlated_dataframe):
        """Test correlation computation with heatmap plotting"""
        plt.close('all')
        try:
            correlations = agent.compute_correlations(correlated_dataframe, plot_heatmap=True)
            assert isinstance(correlations, dict)
            print("✓ Correlation with heatmap passed")
        except Exception as e:
            pytest.fail(f"Correlation with heatmap failed: {e}")
        finally:
            plt.close('all')
    
    def test_plot_correlation_heatmap(self, agent, correlated_dataframe):
        """Test correlation heatmap plotting"""
        plt.close('all')
        try:
            agent.plot_correlation_heatmap(correlated_dataframe)
            assert True
            print("✓ Correlation heatmap plotting passed")
        except Exception as e:
            pytest.fail(f"Heatmap plotting failed: {e}")
        finally:
            plt.close('all')
    
    def test_plot_pairplot(self, agent, correlated_dataframe):
        """Test pairplot generation"""
        plt.close('all')
        try:
            agent.plot_pairplot(correlated_dataframe, max_cols=3)
            assert True
            print("✓ Pairplot generation passed")
        except Exception as e:
            pytest.fail(f"Pairplot generation failed: {e}")
        finally:
            plt.close('all')
    
    def test_plot_categorical_vs_numerical(self, agent, mixed_dataframe):
        """Test categorical vs numerical plotting"""
        plt.close('all')
        try:
            agent.plot_categorical_vs_numerical(
                mixed_dataframe,
                categorical_col='category',
                numerical_col='numerical1'
            )
            assert True
            print("✓ Categorical vs numerical plotting passed")
        except Exception as e:
            pytest.fail(f"Categorical vs numerical plotting failed: {e}")
        finally:
            plt.close('all')
    
    def test_plot_boxplots_categorical_vs_numerical(self, agent, mixed_dataframe):
        """Test multiple boxplots for categorical vs numerical"""
        plt.close('all')
        try:
            agent.plot_boxplots_categorical_vs_numerical(mixed_dataframe)
            assert True
            print("✓ Multiple boxplots passed")
        except Exception as e:
            pytest.fail(f"Multiple boxplots failed: {e}")
        finally:
            plt.close('all')
    
    def test_perform_regression_analysis(self, agent, regression_dataframe):
        """Test linear regression analysis"""
        plt.close('all')
        regression_results = agent.perform_regression_analysis(
            regression_dataframe,
            target_col='target'
        )
        
        assert isinstance(regression_results, dict)
        assert 'target' in regression_results
        assert 'features' in regression_results
        assert 'coefficients' in regression_results
        assert 'intercept' in regression_results
        assert 'r2_score' in regression_results
        assert 0 <= regression_results['r2_score'] <= 1
        print("✓ Regression analysis passed")
        plt.close('all')
    
    def test_identify_strong_relationships(self, agent, correlated_dataframe):
        """Test identification of strong correlations"""
        strong_rels = agent.identify_strong_relationships(
            correlated_dataframe,
            threshold=0.5
        )
        
        assert isinstance(strong_rels, list)
        assert len(strong_rels) > 0
        for rel in strong_rels:
            assert 'var1' in rel
            assert 'var2' in rel
            assert 'correlation' in rel
            assert 'strength' in rel
            assert abs(rel['correlation']) >= 0.5
        print("✓ Strong relationship identification passed")
    
    def test_compute_mutual_information(self, agent, regression_dataframe):
        """Test mutual information computation"""
        plt.close('all')
        mi_scores = agent.compute_mutual_information(
            regression_dataframe,
            target_col='target'
        )
        
        assert isinstance(mi_scores, dict)
        assert len(mi_scores) == len(regression_dataframe.columns) - 1
        for feature, score in mi_scores.items():
            assert isinstance(score, (int, float))
            assert score >= 0
        print("✓ Mutual information computation passed")
        plt.close('all')
    
    def test_result_storage(self, agent, correlated_dataframe):
        """Test that results are properly stored"""
        agent.compute_correlations(correlated_dataframe)
        assert 'correlations' in agent.results
        assert isinstance(agent.results['correlations'], dict)
        print("✓ Result storage passed")
    
    def test_empty_dataframe(self, agent):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()
        correlations = agent.compute_correlations(empty_df)
        assert isinstance(correlations, dict)
        print("✓ Empty DataFrame handling passed")
    
    def test_agent_logging(self, agent, correlated_dataframe):
        """Test that agent logs are being recorded"""
        initial_log_count = len(agent.logs)
        agent.compute_correlations(correlated_dataframe)
        assert len(agent.logs) > initial_log_count
        assert any('correlation' in log.lower() for log in agent.logs)
        print("✓ Agent logging passed")
    
    def test_correlation_values(self, agent, correlated_dataframe):
        """Test that correlation values are valid"""
        correlations = agent.compute_correlations(correlated_dataframe)
        for col1, row in correlations['pearson'].items():
            for col2, value in row.items():
                assert -1 <= value <= 1, f"Invalid correlation: {value}"
        print("✓ Correlation values validation passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])