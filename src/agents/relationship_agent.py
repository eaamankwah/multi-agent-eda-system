class RelationshipDiscoveryAgent(BaseAgent):
    """Conducts correlation, mutual information, and regression analysis"""

    def __init__(self):
        super().__init__("Relationship Discovery")

    def _plot_helper(self, fig, title):
        """Helper method for displaying plots"""
        if isinstance(fig, plt.Figure):
            fig.suptitle(title, y=1.02)
            plt.tight_layout()
            plt.show()
        else:
            fig.update_layout(title_text=title)
            fig.show()
        self.log(f"Plot '{title}' generated")

    def compute_correlations(self, df: pd.DataFrame, plot_heatmap: bool = False) -> Dict:
        self.log("Computing correlation matrices...")
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            self.log("Not enough numerical columns for correlation")
            return {}

        results = {
            'pearson': numeric_df.corr(method='pearson').to_dict(),
            'spearman': numeric_df.corr(method='spearman').to_dict()
        }

        if plot_heatmap:
            self.plot_correlation_heatmap(df)

        self.save_result('correlations', results)
        self.log("Correlation analysis complete")
        return results

    def plot_correlation_heatmap(self, df: pd.DataFrame):
        self.log("Plotting correlation heatmap...")
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            self.log("Not enough numerical columns for correlation heatmap")
            return

        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, square=True)
        self._plot_helper(plt.gcf(), 'Pearson Correlation Heatmap')

    def plot_pairplot(self, df: pd.DataFrame, max_cols: int = 5):
        self.log("Plotting pairplot...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:max_cols]

        if len(numeric_cols) < 2:
            self.log("Not enough numerical columns for pairplot")
            return

        fig = sns.pairplot(df[numeric_cols])
        self._plot_helper(fig.fig, 'Pair Plot')

    def plot_categorical_vs_numerical(self, df: pd.DataFrame,
                                     categorical_col: str = None,
                                     numerical_col: str = None):
        self.log("Plotting categorical vs numerical relationship...")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns

        if not categorical_col and len(categorical_cols) > 0:
            categorical_col = categorical_cols[0]
        if not numerical_col and len(numerical_cols) > 0:
            numerical_col = numerical_cols[0]

        if not categorical_col or not numerical_col:
            self.log("Not enough categorical or numerical columns for plotting")
            return

        fig = px.box(df, x=categorical_col, y=numerical_col)
        self._plot_helper(fig, f'{numerical_col} by {categorical_col}')

    def plot_boxplots_categorical_vs_numerical(self, df: pd.DataFrame):
        """Plot box plots for all categorical vs numerical combinations"""
        self.log("Plotting box plots for categorical vs numerical...")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns

        if len(categorical_cols) == 0 or len(numerical_cols) == 0:
            self.log("Not enough categorical or numerical columns")
            return

        for cat_col in categorical_cols[:2]:  # Limit to first 2 categorical columns
            for num_col in numerical_cols[:2]:  # Limit to first 2 numerical columns
                fig = px.box(df, x=cat_col, y=num_col,
                           title=f'{num_col} by {cat_col}')
                fig.show()

        self.log("Box plots for categorical vs numerical generated")

    def perform_regression_analysis(self, df: pd.DataFrame, target_col: str = None) -> Dict:
        self.log("Performing regression analysis...")
        numeric_df = df.select_dtypes(include=[np.number]).dropna()

        if target_col is None:
            target_col = numeric_df.columns[-1]

        if target_col not in numeric_df.columns:
            self.log(f"Target column {target_col} not found")
            return {}

        X = numeric_df.drop(columns=[target_col])
        y = numeric_df[target_col]

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        results = {
            'target': target_col,
            'features': list(X.columns),
            'coefficients': dict(zip(X.columns, model.coef_)),
            'intercept': float(model.intercept_),
            'r2_score': float(r2)
        }

        plt.figure(figsize=Config.FIGURE_SIZE)
        plt.scatter(y, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        self._plot_helper(plt.gcf(), f'Linear Regression: R² = {r2:.3f}')

        self.save_result('regression', results)
        return results

    def identify_strong_relationships(self, df: pd.DataFrame, threshold: float = Config.CORRELATION_THRESHOLD) -> List:
        """Identify strong relationships between variables"""
        self.log(f"Identifying relationships above threshold {threshold}...")

        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()

        strong_relationships = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    strong_relationships.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val,
                        'strength': 'strong' if abs(corr_val) > 0.7 else 'moderate'
                    })

        self.log(f"Found {len(strong_relationships)} strong relationships")
        return strong_relationships

    def compute_mutual_information(self, df: pd.DataFrame, target_col: str = None) -> Dict:
        """Compute mutual information scores"""
        self.log("Computing mutual information...")

        numeric_df = df.select_dtypes(include=[np.number]).dropna()

        if target_col is None:
            target_col = numeric_df.columns[-1]

        if target_col not in numeric_df.columns:
            self.log(f"Target column {target_col} not found")
            return {}

        X = numeric_df.drop(columns=[target_col])
        y = numeric_df[target_col]

        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_dict = dict(zip(X.columns, mi_scores))

        # Plot
        plt.figure(figsize=Config.FIGURE_SIZE)
        sorted_mi = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
        features, scores = zip(*sorted_mi)
        plt.barh(features, scores)
        plt.xlabel('Mutual Information Score')
        plt.title(f'Mutual Information with {target_col}')
        plt.tight_layout()
        plt.show()

        self.save_result('mutual_information', mi_dict)
        self.log(f"Mutual information computed for {len(mi_dict)} features")
        return mi_dict

