class DataQualityAgent(BaseAgent):
    """Implements data validation, cleaning, and quality checks"""

    def __init__(self):
        super().__init__("Data Quality")
        self.quality_report = {}
        self.label_encoders = {}

    def assess_data_quality(self, df: pd.DataFrame) -> Dict:
        self.log("Assessing data quality...")
        report = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2
        }
        self.quality_report = report
        self.log(f"Quality assessment complete: {report['shape'][0]} rows, {report['shape'][1]} columns")
        return report

    def identify_data_types(self, df: pd.DataFrame) -> Dict:
        """Identify and categorize data types"""
        self.log("Identifying data types...")
        dtypes_info = {
            'numerical': list(df.select_dtypes(include=[np.number]).columns),
            'categorical': list(df.select_dtypes(include=['object', 'category']).columns),
            'datetime': list(df.select_dtypes(include=['datetime64']).columns),
            'all_dtypes': df.dtypes.to_dict()
        }
        self.log(f"Found {len(dtypes_info['numerical'])} numerical, {len(dtypes_info['categorical'])} categorical columns")
        return dtypes_info

    def encode_categorical_features(self, df: pd.DataFrame, method: str = 'label') -> pd.DataFrame:
        """Encode categorical features for ML algorithms"""
        self.log(f"Encoding categorical features using {method} encoding...")
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns

        if len(categorical_cols) == 0:
            self.log("No categorical columns to encode")
            return df_encoded

        from sklearn.preprocessing import LabelEncoder

        if method == 'label':
            for col in categorical_cols:
                le = LabelEncoder()
                # Handle NaN values
                mask = df_encoded[col].notna()
                df_encoded.loc[mask, col + '_encoded'] = le.fit_transform(df_encoded.loc[mask, col])
                self.label_encoders[col] = le
                self.log(f"Encoded {col}: {df_encoded[col].nunique()} unique values")

        elif method == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, prefix=categorical_cols)
            self.log(f"One-hot encoded {len(categorical_cols)} categorical columns")

        self.log(f"Categorical encoding complete. New shape: {df_encoded.shape}")
        return df_encoded

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'smart') -> pd.DataFrame:
        self.log(f"Handling missing values with strategy: {strategy}")
        df_clean = df.copy()

        for col in df_clean.columns:
            missing_pct = df_clean[col].isnull().sum() / len(df_clean)

            if missing_pct > Config.MAX_MISSING_THRESHOLD:
                self.log(f"Dropping column {col}: {missing_pct*100:.1f}% missing")
                df_clean = df_clean.drop(columns=[col])
                continue

            if df_clean[col].dtype in ['float64', 'int64']:
                if strategy == 'mean':
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif strategy == 'median':
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                else:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown', inplace=True)

        self.log(f"Missing values handled: {df_clean.shape[1]} columns remaining")
        return df_clean

    def handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log("Handling duplicate rows...")
        initial_rows = len(df)
        df_clean = df.drop_duplicates().copy()
        dropped_rows = initial_rows - len(df_clean)
        self.log(f"Dropped {dropped_rows} duplicate rows")
        return df_clean

    def detect_outliers(self, df: pd.DataFrame, method: str = 'zscore') -> Dict:
        self.log(f"Detecting outliers using {method} method...")
        outlier_report = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > Config.OUTLIER_THRESHOLD
                outlier_report[col] = {
                    'count': outliers.sum(),
                    'percentage': (outliers.sum() / len(df)) * 100
                }

        self.log(f"Outlier detection complete for {len(numeric_cols)} columns")
        return outlier_report

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features using StandardScaler"""
        self.log("Normalizing numerical features...")
        df_normalized = df.copy()
        numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            self.log("No numerical columns to normalize")
            return df_normalized

        scaler = StandardScaler()
        df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])

        self.log(f"Normalized {len(numeric_cols)} numerical columns")
        return df_normalized

    def validate_with_great_expectations(self, df: pd.DataFrame) -> Dict:
        """Validate data using Great Expectations"""
        self.log("Running Great Expectations validation...")

        try:
            validation_results = {
                'columns_exist': all(col in df.columns for col in df.columns),
                'no_null_in_required': df.notnull().all().to_dict(),
                'valid_dtypes': df.dtypes.to_dict()
            }

            self.log("Great Expectations validation complete")
            return validation_results

        except Exception as e:
            self.log(f"Great Expectations validation error: {e}")
            return {'error': str(e)}