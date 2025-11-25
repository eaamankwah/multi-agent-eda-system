class DataExplorationAgent(BaseAgent):
    """Generates summary statistics, distributions, and clustering"""

    def __init__(self):
        super().__init__("Data Exploration")

    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        self.log("Generating summary statistics...")
        summary = {
            'numerical_summary': df.describe().to_dict(),
            'categorical_summary': {},
            'data_types': df.dtypes.to_dict(),
            'unique_counts': df.nunique().to_dict()
        }

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            summary['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'top_values': df[col].value_counts().head(10).to_dict()
            }

        self.save_result('summary_statistics', summary)
        self.log("Summary statistics generated")
        return summary

    def identify_outliers(self, df: pd.DataFrame, method: str = 'zscore') -> Dict:
        """Identify outliers in numerical columns"""
        self.log(f"Identifying outliers using {method} method...")
        outlier_report = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > Config.OUTLIER_THRESHOLD
                outlier_report[col] = {
                    'count': int(outliers.sum()),
                    'percentage': float((outliers.sum() / len(df)) * 100),
                    'outlier_values': df[outliers][col].tolist()[:10]  # First 10 outliers
                }
            elif method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
                outlier_report[col] = {
                    'count': int(outliers.sum()),
                    'percentage': float((outliers.sum() / len(df)) * 100)
                }

        self.log(f"Outlier identification complete for {len(numeric_cols)} columns")
        return outlier_report

    def plot_distributions(self, df: pd.DataFrame, save_path: str = None):
        self.log("Plotting distributions...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_cols)

        if n_cols == 0:
            self.log("No numerical columns to plot")
            return

        n_rows = (n_cols + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_cols > 1 else [axes]

        for idx, col in enumerate(numeric_cols):
            axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')

        for idx in range(n_cols, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_countplots(self, df: pd.DataFrame):
        self.log("Plotting count plots...")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        if len(categorical_cols) == 0:
            self.log("No categorical columns to plot")
            return

        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            sns.countplot(y=df[col], order=df[col].value_counts().index, palette=Config.COLOR_PALETTE)
            plt.title(f'Count Plot of {col}')
            plt.tight_layout()
            plt.show()

    def plot_boxplots(self, df: pd.DataFrame):
        self.log("Plotting box plots...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            self.log("No numerical columns to plot")
            return

        for col in numeric_cols:
            fig = px.box(df, y=col, title=f'Box Plot of {col}')
            fig.show()

    def perform_clustering(self, df: pd.DataFrame, n_clusters: int = None) -> Dict:
        self.log("Performing K-means clustering...")
        numeric_df = df.select_dtypes(include=[np.number]).dropna()

        if numeric_df.shape[1] < 2:
            self.log("Not enough features for clustering")
            return {}

        if n_clusters is None:
            silhouette_scores = []
            K_range = range(2, min(10, len(numeric_df)//2))

            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(numeric_df)
                score = silhouette_score(numeric_df, labels)
                silhouette_scores.append(score)

            n_clusters = K_range[np.argmax(silhouette_scores)]
            self.log(f"Optimal number of clusters: {n_clusters}")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(numeric_df)

        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(numeric_df)

        plt.figure(figsize=Config.FIGURE_SIZE)
        scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1],
                            c=labels, cmap=Config.COLOR_PALETTE, alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('K-Means Clustering (PCA Projection)')
        plt.tight_layout()
        plt.show()

        clustering_results = {
            'n_clusters': n_clusters,
            'labels': labels.tolist(),
            'silhouette_score': float(silhouette_score(numeric_df, labels))
        }

        self.save_result('clustering', clustering_results)
        return clustering_results

    def create_scatter_matrix(self, df: pd.DataFrame, max_cols: int = 5):
        """Create scatter plot matrix for numerical columns"""
        self.log("Creating scatter matrix...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns[:max_cols]

        if len(numeric_cols) < 2:
            self.log("Not enough numerical columns for scatter matrix")
            return

        pd.plotting.scatter_matrix(df[numeric_cols], figsize=(15, 15), alpha=0.6, diagonal='hist')
        plt.suptitle('Scatter Matrix', y=1.0)
        plt.tight_layout()
        plt.show()

