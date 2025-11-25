# ============================================================================
# test_exploration.py
# ============================================================================
"""
Test Suite for Data Exploration Agent
Tests statistical analysis, visualization, clustering, and outlier detection
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.agents.exploration_agent import DataExplorationAgent


class TestDataExplorationAgent:
    """Test cases for DataExplorationAgent"""
    
    @pytest.fixture
    def agent(self):
        """Initialize data exploration agent for testing"""
        return DataExplorationAgent()
    
    @pytest.fixture
    def numerical_dataframe(self):
        """Sample DataFrame with numerical data"""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.normal(50, 10, 200),
            'feature2': np.random.normal(100, 20, 200),
            'feature3': np.random.exponential(5, 200),
            'feature4': np.random.uniform(0, 100, 200),
            'target': np.random.normal(75, 15, 200)
        })
    
    @pytest.fixture
    def mixed_dataframe(self):
        """Sample DataFrame with mixed data types"""
        np.random.seed(42)
        return pd.DataFrame({
            'id': range(1, 201),
            'age': np.random.randint(18, 80, 200),
            'income': np.random.normal(50000, 20000, 200),
            'score': np.random.uniform(0, 100, 200),
            'category': np.random.choice(['A', 'B', 'C', 'D'], 200),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 200),
            'status': np.random.choice(['Active', 'Inactive'], 200)
        })
    
    @pytest.fixture
    def outlier_dataframe(self):
        """DataFrame with outliers"""
        np.random.seed(42)
        df = pd.DataFrame({
            'normal_col': np.random.normal(50, 10, 200),
            'outlier_col': np.random.normal(100, 15, 200)
        })
        # Add extreme outliers
        df.loc[0, 'outlier_col'] = 500
        df.loc[1, 'outlier_col'] = -200
        df.loc[2, 'outlier_col'] = 450
        return df
    
    def test_generate_summary_statistics(self, agent, mixed_dataframe):
        """Test comprehensive summary statistics generation"""
        summary = agent.generate_summary_statistics(mixed_dataframe)
        
        assert isinstance(summary, dict)
        assert 'numerical_summary' in summary
        assert 'categorical_summary' in summary
        assert 'data_types' in summary
        assert 'unique_counts' in summary
        assert isinstance(summary['numerical_summary'], dict)
        assert isinstance(summary['categorical_summary'], dict)
        print("✓ Summary statistics generation passed")
    
    def test_numerical_summary_content(self, agent, numerical_dataframe):
        """Test numerical summary contains correct statistics"""
        summary = agent.generate_summary_statistics(numerical_dataframe)
        
        for col in numerical_dataframe.columns:
            assert col in summary['numerical_summary']
            col_stats = summary['numerical_summary'][col]
            assert 'mean' in col_stats
            assert 'std' in col_stats
            assert 'min' in col_stats
            assert 'max' in col_stats
        print("✓ Numerical summary content passed")
    
    def test_categorical_summary_content(self, agent, mixed_dataframe):
        """Test categorical summary contains value counts"""
        summary = agent.generate_summary_statistics(mixed_dataframe)
        
        categorical_cols = mixed_dataframe.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            assert col in summary['categorical_summary']
            assert 'unique_values' in summary['categorical_summary'][col]
            assert 'top_values' in summary['categorical_summary'][col]
        print("✓ Categorical summary content passed")
    
    def test_identify_outliers_zscore(self, agent, outlier_dataframe):
        """Test outlier identification using z-score method"""
        outliers = agent.identify_outliers(outlier_dataframe, method='zscore')
        
        assert isinstance(outliers, dict)
        assert 'outlier_col' in outliers
        assert outliers['outlier_col']['count'] > 0
        assert outliers['outlier_col']['percentage'] > 0
        assert 'outlier_values' in outliers['outlier_col']
        print("✓ Z-score outlier identification passed")
    
    def test_identify_outliers_iqr(self, agent, outlier_dataframe):
        """Test outlier identification using IQR method"""
        outliers = agent.identify_outliers(outlier_dataframe, method='iqr')
        
        assert isinstance(outliers, dict)
        assert 'outlier_col' in outliers
        assert outliers['outlier_col']['count'] >= 0
        assert 'percentage' in outliers['outlier_col']
        print("✓ IQR outlier identification passed")
    
    def test_plot_distributions(self, agent, numerical_dataframe):
        """Test distribution plotting"""
        plt.close('all')
        try:
            agent.plot_distributions(numerical_dataframe)
            assert True
            print("✓ Distribution plotting passed")
        except Exception as e:
            pytest.fail(f"Distribution plotting failed: {e}")
        finally:
            plt.close('all')
    
    def test_plot_distributions_save(self, agent, numerical_dataframe, tmp_path):
        """Test distribution plotting with save path"""
        plt.close('all')
        save_path = tmp_path / "distributions.png"
        try:
            agent.plot_distributions(numerical_dataframe, save_path=str(save_path))
            assert save_path.exists()
            print("✓ Distribution plotting with save passed")
        except Exception as e:
            pytest.fail(f"Distribution plotting with save failed: {e}")
        finally:
            plt.close('all')
    
    def test_plot_countplots(self, agent, mixed_dataframe):
        """Test categorical count plotting"""
        plt.close('all')
        try:
            agent.plot_countplots(mixed_dataframe)
            assert True
            print("✓ Count plot generation passed")
        except Exception as e:
            pytest.fail(f"Count plot generation failed: {e}")
        finally:
            plt.close('all')
    
    def test_plot_boxplots(self, agent, numerical_dataframe):
        """Test box plot generation"""
        plt.close('all')
        try:
            agent.plot_boxplots(numerical_dataframe)
            assert True
            print("✓ Box plot generation passed")
        except Exception as e:
            pytest.fail(f"Box plot generation failed: {e}")
        finally:
            plt.close('all')
    
    def test_perform_clustering_auto(self, agent, numerical_dataframe):
        """Test clustering with automatic cluster selection"""
        plt.close('all')
        clustering_results = agent.perform_clustering(numerical_dataframe, n_clusters=None)
        
        assert isinstance(clustering_results, dict)
        assert 'n_clusters' in clustering_results
        assert 'labels' in clustering_results
        assert 'silhouette_score' in clustering_results
        assert clustering_results['n_clusters'] >= 2
        assert len(clustering_results['labels']) == len(numerical_dataframe)
        assert -1 <= clustering_results['silhouette_score'] <= 1
        print("✓ Auto clustering passed")
        plt.close('all')
    
    def test_perform_clustering_fixed(self, agent, numerical_dataframe):
        """Test clustering with fixed number of clusters"""
        plt.close('all')
        n_clusters = 4
        clustering_results = agent.perform_clustering(numerical_dataframe, n_clusters=n_clusters)
        
        assert isinstance(clustering_results, dict)
        assert clustering_results['n_clusters'] == n_clusters
        assert len(set(clustering_results['labels'])) == n_clusters
        print("✓ Fixed clustering passed")
        plt.close('all')
    
    def test_create_scatter_matrix(self, agent, numerical_dataframe):
        """Test scatter matrix generation"""
        plt.close('all')
        try:
            agent.create_scatter_matrix(numerical_dataframe, max_cols=4)
            assert True
            print("✓ Scatter matrix creation passed")
        except Exception as e:
            pytest.fail(f"Scatter matrix creation failed: {e}")
        finally:
            plt.close('all')
    
    def test_result_storage(self, agent, numerical_dataframe):
        """Test that results are properly stored"""
        agent.generate_summary_statistics(numerical_dataframe)
        assert 'summary_statistics' in agent.results
        assert isinstance(agent.results['summary_statistics'], dict)
        print("✓ Result storage passed")
    
    def test_empty_dataframe(self, agent):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()
        summary = agent.generate_summary_statistics(empty_df)
        assert isinstance(summary, dict)
        print("✓ Empty DataFrame handling passed")
    
    def test_agent_logging(self, agent, numerical_dataframe):
        """Test that agent logs are being recorded"""
        initial_log_count = len(agent.logs)
        agent.generate_summary_statistics(numerical_dataframe)
        assert len(agent.logs) > initial_log_count
        assert any('summary' in log.lower() for log in agent.logs)
        print("✓ Agent logging passed")