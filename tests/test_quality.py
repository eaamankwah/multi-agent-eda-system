# ============================================================================
# test_quality.py
# ============================================================================
"""
Test Suite for Data Quality Agent
Tests data quality assessment, cleaning, validation, and encoding
"""

import pytest
import pandas as pd
import numpy as np
from src.agents.quality_agent import DataQualityAgent


class TestDataQualityAgent:
    """Test cases for DataQualityAgent"""
    
    @pytest.fixture
    def agent(self):
        """Initialize data quality agent for testing"""
        return DataQualityAgent()
    
    @pytest.fixture
    def clean_dataframe(self):
        """Sample clean DataFrame"""
        return pd.DataFrame({
            'id': range(1, 101),
            'age': np.random.randint(18, 80, 100),
            'income': np.random.normal(50000, 15000, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'score': np.random.uniform(0, 100, 100)
        })
    
    @pytest.fixture
    def messy_dataframe(self):
        """DataFrame with quality issues"""
        df = pd.DataFrame({
            'id': range(1, 101),
            'age': np.random.randint(18, 80, 100),
            'income': np.random.normal(50000, 15000, 100),
            'category': np.random.choice(['A', 'B', 'C', None], 100),
            'score': np.random.uniform(0, 100, 100)
        })
        
        # Add missing values
        df.loc[np.random.choice(100, 20, replace=False), 'income'] = np.nan
        df.loc[np.random.choice(100, 15, replace=False), 'score'] = np.nan
        
        # Add duplicates
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)
        
        # Add outliers
        df.loc[0, 'income'] = 1000000  # Extreme outlier
        df.loc[1, 'age'] = 150  # Invalid age
        
        return df
    
    @pytest.fixture
    def high_missing_dataframe(self):
        """DataFrame with high percentage of missing values"""
        df = pd.DataFrame({
            'id': range(1, 101),
            'mostly_missing': [np.nan] * 95 + [1, 2, 3, 4, 5],
            'some_missing': [np.nan] * 30 + list(range(70)),
            'no_missing': range(100)
        })
        return df
    
    # Test 1: Data Quality Assessment
    def test_assess_data_quality(self, agent, clean_dataframe):
        """Test comprehensive data quality assessment"""
        report = agent.assess_data_quality(clean_dataframe)
        
        assert isinstance(report, dict)
        assert 'shape' in report
        assert 'columns' in report
        assert 'dtypes' in report
        assert 'missing_values' in report
        assert 'duplicates' in report
        assert 'memory_usage' in report
        
        assert report['shape'] == clean_dataframe.shape
        assert len(report['columns']) == clean_dataframe.shape[1]
        print("✓ Data quality assessment passed")
    
    # Test 2: Identify Data Types
    def test_identify_data_types(self, agent, clean_dataframe):
        """Test data type identification"""
        dtypes_info = agent.identify_data_types(clean_dataframe)
        
        assert isinstance(dtypes_info, dict)
        assert 'numerical' in dtypes_info
        assert 'categorical' in dtypes_info
        assert 'datetime' in dtypes_info
        assert 'all_dtypes' in dtypes_info
        
        assert isinstance(dtypes_info['numerical'], list)
        assert isinstance(dtypes_info['categorical'], list)
        assert len(dtypes_info['numerical']) > 0
        print("✓ Data type identification passed")
    
    # Test 3: Handle Missing Values
    def test_handle_missing_values(self, agent, messy_dataframe):
        """Test missing value handling"""
        initial_missing = messy_dataframe.isnull().sum().sum()
        cleaned_df = agent.handle_missing_values(messy_dataframe, strategy='smart')
        final_missing = cleaned_df.isnull().sum().sum()
        
        assert isinstance(cleaned_df, pd.DataFrame)
        assert final_missing < initial_missing or initial_missing == 0
        print("✓ Missing value handling passed")
    
    # Test 4: Handle Missing Values - Mean Strategy
    def test_handle_missing_values_mean(self, agent, messy_dataframe):
        """Test missing value handling with mean strategy"""
        cleaned_df = agent.handle_missing_values(messy_dataframe, strategy='mean')
        
        assert isinstance(cleaned_df, pd.DataFrame)
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in messy_dataframe.columns:
                assert cleaned_df[col].isnull().sum() <= messy_dataframe[col].isnull().sum()
        print("✓ Mean strategy missing value handling passed")
    
    # Test 5: Handle Missing Values - Median Strategy
    def test_handle_missing_values_median(self, agent, messy_dataframe):
        """Test missing value handling with median strategy"""
        cleaned_df = agent.handle_missing_values(messy_dataframe, strategy='median')
        
        assert isinstance(cleaned_df, pd.DataFrame)
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in messy_dataframe.columns:
                assert cleaned_df[col].isnull().sum() <= messy_dataframe[col].isnull().sum()
        print("✓ Median strategy missing value handling passed")
    
    # Test 6: Drop High Missing Columns
    def test_drop_high_missing_columns(self, agent, high_missing_dataframe):
        """Test dropping columns with high missing percentage"""
        cleaned_df = agent.handle_missing_values(high_missing_dataframe)
        
        assert isinstance(cleaned_df, pd.DataFrame)
        # Column with >50% missing should be dropped
        assert 'mostly_missing' not in cleaned_df.columns
        assert 'no_missing' in cleaned_df.columns
        print("✓ High missing column dropping passed")
    
    # Test 7: Handle Duplicates
    def test_handle_duplicates(self, agent, messy_dataframe):
        """Test duplicate row handling"""
        initial_rows = len(messy_dataframe)
        cleaned_df = agent.handle_duplicates(messy_dataframe)
        final_rows = len(cleaned_df)
        
        assert isinstance(cleaned_df, pd.DataFrame)
        assert final_rows <= initial_rows
        assert cleaned_df.duplicated().sum() == 0
        print("✓ Duplicate handling passed")
    
    # Test 8: Detect Outliers - Z-score
    def test_detect_outliers_zscore(self, agent, messy_dataframe):
        """Test outlier detection using z-score method"""
        outlier_report = agent.detect_outliers(messy_dataframe, method='zscore')
        
        assert isinstance(outlier_report, dict)
        numeric_cols = messy_dataframe.select_dtypes(include=[np.number]).columns
        assert len(outlier_report) == len(numeric_cols)
        
        for col, stats in outlier_report.items():
            assert 'count' in stats
            assert 'percentage' in stats
            assert stats['count'] >= 0
        print("✓ Z-score outlier detection passed")
    
    # Test 9: Encode Categorical Features - Label Encoding
    def test_encode_categorical_label(self, agent, clean_dataframe):
        """Test label encoding of categorical features"""
        encoded_df = agent.encode_categorical_features(clean_dataframe, method='label')
        
        assert isinstance(encoded_df, pd.DataFrame)
        # Check if encoded columns were created
        assert 'category_encoded' in encoded_df.columns or len(encoded_df.columns) >= len(clean_dataframe.columns)
        assert len(agent.label_encoders) > 0
        print("✓ Label encoding passed")
    
    # Test 10: Encode Categorical Features - One-hot Encoding
    def test_encode_categorical_onehot(self, agent, clean_dataframe):
        """Test one-hot encoding of categorical features"""
        encoded_df = agent.encode_categorical_features(clean_dataframe, method='onehot')
        
        assert isinstance(encoded_df, pd.DataFrame)
        # One-hot encoding should increase number of columns
        assert encoded_df.shape[1] >= clean_dataframe.shape[1]
        print("✓ One-hot encoding passed")
    
    # Test 11: Normalize Data
    def test_normalize_data(self, agent, clean_dataframe):
        """Test data normalization"""
        normalized_df = agent.normalize_data(clean_dataframe)
        
        assert isinstance(normalized_df, pd.DataFrame)
        assert normalized_df.shape == clean_dataframe.shape
        
        # Check if numerical columns are normalized (mean ≈ 0, std ≈ 1)
        numeric_cols = normalized_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean_val = normalized_df[col].mean()
            std_val = normalized_df[col].std()
            assert abs(mean_val) < 0.1  # Mean should be close to 0
            assert abs(std_val - 1.0) < 0.1  # Std should be close to 1
        print("✓ Data normalization passed")
    
    # Test 12: Normalize Data - No Numeric Columns
    def test_normalize_data_no_numeric(self, agent):
        """Test normalization with no numeric columns"""
        df = pd.DataFrame({
            'category1': ['A', 'B', 'C'],
            'category2': ['X', 'Y', 'Z']
        })
        
        normalized_df = agent.normalize_data(df)
        
        assert isinstance(normalized_df, pd.DataFrame)
        assert normalized_df.equals(df)
        print("✓ Normalization with no numeric columns passed")
    
    # Test 13: Validate with Great Expectations
    def test_validate_great_expectations(self, agent, clean_dataframe):
        """Test Great Expectations validation"""
        validation_results = agent.validate_with_great_expectations(clean_dataframe)
        
        assert isinstance(validation_results, dict)
        assert 'columns_exist' in validation_results
        assert 'no_null_in_required' in validation_results
        assert 'valid_dtypes' in validation_results
        print("✓ Great Expectations validation passed")
    
    # Test 14: Quality Report Persistence
    def test_quality_report_persistence(self, agent, clean_dataframe):
        """Test that quality report is stored"""
        agent.assess_data_quality(clean_dataframe)
        
        assert hasattr(agent, 'quality_report')
        assert isinstance(agent.quality_report, dict)
        assert len(agent.quality_report) > 0
        print("✓ Quality report persistence passed")
    
    # Test 15: Multiple Sequential Operations
    def test_sequential_operations(self, agent, messy_dataframe):
        """Test multiple cleaning operations in sequence"""
        # Step 1: Assess quality
        report = agent.assess_data_quality(messy_dataframe)
        assert isinstance(report, dict)
        
        # Step 2: Handle duplicates
        df = agent.handle_duplicates(messy_dataframe)
        assert df.duplicated().sum() == 0
        
        # Step 3: Handle missing values
        df = agent.handle_missing_values(df)
        
        # Step 4: Detect outliers
        outliers = agent.detect_outliers(df)
        assert isinstance(outliers, dict)
        
        # Step 5: Normalize
        df = agent.normalize_data(df)
        
        assert isinstance(df, pd.DataFrame)
        print("✓ Sequential operations passed")
    
    # Test 16: Empty DataFrame Handling
    def test_empty_dataframe(self, agent):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()
        
        report = agent.assess_data_quality(empty_df)
        assert isinstance(report, dict)
        assert report['shape'] == (0, 0)
        print("✓ Empty DataFrame handling passed")
    
    # Test 17: Agent Logging
    def test_agent_logging(self, agent, clean_dataframe):
        """Test that agent logs are being recorded"""
        initial_log_count = len(agent.logs)
        agent.assess_data_quality(clean_dataframe)
        
        assert len(agent.logs) > initial_log_count
        assert any('quality' in log.lower() for log in agent.logs)
        print("✓ Agent logging passed")
    
    # Test 18: Categorical with NaN Handling
    def test_categorical_nan_handling(self, agent):
        """Test handling categorical columns with NaN values"""
        df = pd.DataFrame({
            'category': ['A', 'B', None, 'C', None, 'A'],
            'value': [1, 2, 3, 4, 5, 6]
        })
        
        cleaned_df = agent.handle_missing_values(df)
        
        assert isinstance(cleaned_df, pd.DataFrame)
        # Categorical NaN should be filled
        assert cleaned_df['category'].isnull().sum() < df['category'].isnull().sum() or df['category'].isnull().sum() == 0
        print("✓ Categorical NaN handling passed")
    
    # Test 19: Large Dataset Performance
    def test_large_dataset_quality(self, agent):
        """Test quality assessment on large dataset"""
        large_df = pd.DataFrame({
            'col1': np.random.rand(10000),
            'col2': np.random.randint(0, 100, 10000),
            'col3': np.random.choice(['A', 'B', 'C'], 10000)
        })
        
        report = agent.assess_data_quality(large_df)
        
        assert isinstance(report, dict)
        assert report['shape'][0] == 10000
        print("✓ Large dataset quality assessment passed")
    
    # Test 20: Mixed Quality Issues
    def test_mixed_quality_issues(self, agent, messy_dataframe):
        """Test handling of multiple quality issues simultaneously"""
        initial_quality = agent.assess_data_quality(messy_dataframe)
        
        # Clean the data
        cleaned_df = agent.handle_duplicates(messy_dataframe)
        cleaned_df = agent.handle_missing_values(cleaned_df)
        
        final_quality = agent.assess_data_quality(cleaned_df)
        
        # Quality should improve
        assert final_quality['duplicates'] == 0
        assert sum(final_quality['missing_values'].values()) <= sum(initial_quality['missing_values'].values())
        print("✓ Mixed quality issues handling passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])