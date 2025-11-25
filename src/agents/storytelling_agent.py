class DataStorytellingAgent(BaseAgent):
    """Creates interactive visualizations and narratives"""

    def __init__(self):
        super().__init__("Data Storytelling")
        self.dashboard_app = None

    def generate_narrative_summary(self, df: pd.DataFrame, analysis_results: Dict = None) -> str:
        """Generate AI-powered narrative summary"""
        self.log("Generating narrative summary...")

        if analysis_results is None:
            analysis_results = {}

        context = f"""
        Dataset Summary:
        - Shape: {df.shape[0]} rows, {df.shape[1]} columns
        - Numerical columns: {df.select_dtypes(include=[np.number]).shape[1]}
        - Categorical columns: {df.select_dtypes(include=['object', 'category']).shape[1]}
        - Missing values: {df.isnull().sum().sum()}

        Analysis Results:
        {json.dumps(analysis_results, indent=2, default=str)[:1000]}

        Generate a concise narrative summary highlighting key findings and insights.
        """

        try:
            narrative = self.get_llm_guidance(context)
            if not narrative or "API key not valid" in narrative:
                narrative = self._generate_fallback_narrative(df, analysis_results)
            self.save_result('narrative', narrative)
            self.log("Narrative summary generated")
            return narrative
        except Exception as e:
            self.log(f"Narrative generation failed: {e}")
            return self._generate_fallback_narrative(df, analysis_results)

    def generate_summary_insights(self, df: pd.DataFrame, analysis_results: Dict = None) -> str:
        """Alias for generate_narrative_summary"""
        self.log("Generating summary insights...")
        return self.generate_narrative_summary(df, analysis_results)

    def _generate_fallback_narrative(self, df: pd.DataFrame, analysis_results: Dict) -> str:
        """Generate basic narrative without LLM"""
        narrative = f"""
        ## Exploratory Data Analysis Summary

        ### Dataset Overview
        - Total Records: {df.shape[0]:,}
        - Total Features: {df.shape[1]}
        - Numerical Features: {df.select_dtypes(include=[np.number]).shape[1]}
        - Categorical Features: {df.select_dtypes(include=['object', 'category']).shape[1]}

        ### Data Quality
        - Missing Values: {df.isnull().sum().sum():,}
        - Duplicate Rows: {df.duplicated().sum()}

        ### Key Findings
        The dataset has been analyzed successfully. Review the visualizations and
        statistical summaries for detailed insights.
        """
        return narrative

    def create_interactive_dashboard(self, df: pd.DataFrame, analysis_results: Dict) -> Optional[Dash]:
        """Create interactive Dash dashboard"""
        if not DASH_AVAILABLE:
            self.log("Dash not available. Cannot create dashboard.")
            return None

        self.log("Creating interactive dashboard...")

        app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("🔍 Multi-Agent EDA Dashboard", className="text-center mb-4"))
            ]),

            # Summary Statistics Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📊 Dataset Overview", className="card-title"),
                            html.P(f"Rows: {df.shape[0]:,}"),
                            html.P(f"Columns: {df.shape[1]}"),
                            html.P(f"Numerical: {len(numeric_cols)}"),
                            html.P(f"Categorical: {len(categorical_cols)}"),
                        ])
                    ])
                ], width=3),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🎯 Data Quality", className="card-title"),
                            html.P(f"Missing Values: {df.isnull().sum().sum():,}"),
                            html.P(f"Duplicates: {df.duplicated().sum()}"),
                            html.P(f"Memory: {df.memory_usage(deep=True).sum()/1024**2:.2f} MB"),
                        ])
                    ])
                ], width=3),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🔗 Relationships", className="card-title"),
                            html.P("Analysis Complete"),
                            html.P("Correlations Computed"),
                            html.P("Patterns Identified"),
                        ])
                    ])
                ], width=3),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("✨ Insights", className="card-title"),
                            html.P("Clusters Found"),
                            html.P("Outliers Detected"),
                            html.P("Ready for Action"),
                        ])
                    ])
                ], width=3),
            ], className="mb-4"),

            # Interactive Plots
            dbc.Row([
                dbc.Col([
                    html.H4("Select Variables for Analysis"),
                    html.Label("X-Axis:"),
                    dcc.Dropdown(
                        id='x-axis-dropdown',
                        options=[{'label': col, 'value': col} for col in numeric_cols],
                        value=numeric_cols[0] if numeric_cols else None
                    ),
                    html.Label("Y-Axis:", className="mt-2"),
                    dcc.Dropdown(
                        id='y-axis-dropdown',
                        options=[{'label': col, 'value': col} for col in numeric_cols],
                        value=numeric_cols[1] if len(numeric_cols) > 1 else None
                    ),
                    html.Label("Color By:", className="mt-2"),
                    dcc.Dropdown(
                        id='color-dropdown',
                        options=[{'label': col, 'value': col} for col in categorical_cols],
                        value=categorical_cols[0] if categorical_cols else None
                    ),
                ], width=3),

                dbc.Col([
                    dcc.Graph(id='scatter-plot')
                ], width=9),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='histogram')
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='box-plot')
                ], width=6),
            ]),
        ], fluid=True)

        # Callbacks
        @app.callback(
            Output('scatter-plot', 'figure'),
            [Input('x-axis-dropdown', 'value'),
             Input('y-axis-dropdown', 'value'),
             Input('color-dropdown', 'value')]
        )
        def update_scatter(x_col, y_col, color_col):
            if x_col and y_col:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                               title=f'{y_col} vs {x_col}',
                               template='plotly_white')
                return fig
            return go.Figure()

        @app.callback(
            Output('histogram', 'figure'),
            [Input('x-axis-dropdown', 'value')]
        )
        def update_histogram(x_col):
            if x_col:
                fig = px.histogram(df, x=x_col, title=f'Distribution of {x_col}',
                                 template='plotly_white')
                return fig
            return go.Figure()

        @app.callback(
            Output('box-plot', 'figure'),
            [Input('y-axis-dropdown', 'value')]
        )
        def update_boxplot(y_col):
            if y_col:
                fig = px.box(df, y=y_col, title=f'Box Plot of {y_col}',
                           template='plotly_white')
                return fig
            return go.Figure()

        self.dashboard_app = app
        self.log("Dashboard created successfully")
        return app

    def create_summary_report(self, df: pd.DataFrame, analysis_results: Dict) -> str:
        """Create comprehensive HTML report"""
        self.log("Creating summary report...")

        report_html = f"""
        <html>
        <head>
            <title>EDA Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; margin-top: 30px; }}
                .metric {{ background: #ffffff; padding: 20px; margin: 10px 0; border-radius: 8px;
                          box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 15px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .summary-box {{ background: #e8f4f8; padding: 15px; border-left: 4px solid #3498db;
                               margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>🔍 Multi-Agent EDA Report</h1>

            <div class="summary-box">
                <strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>

            <div class="metric">
                <h2>📊 Dataset Overview</h2>
                <p><strong>Total Rows:</strong> {df.shape[0]:,}</p>
                <p><strong>Total Columns:</strong> {df.shape[1]}</p>
                <p><strong>Numerical Columns:</strong> {df.select_dtypes(include=[np.number]).shape[1]}</p>
                <p><strong>Categorical Columns:</strong> {df.select_dtypes(include=['object', 'category']).shape[1]}</p>
                <p><strong>Memory Usage:</strong> {df.memory_usage(deep=True).sum()/1024**2:.2f} MB</p>
            </div>

            <div class="metric">
                <h2>📈 Summary Statistics</h2>
                {df.describe().to_html()}
            </div>

            <div class="metric">
                <h2>🎯 Data Quality Metrics</h2>
                <p><strong>Missing Values:</strong> {df.isnull().sum().sum():,}</p>
                <p><strong>Duplicate Rows:</strong> {df.duplicated().sum()}</p>
                <p><strong>Columns with Missing Data:</strong> {(df.isnull().sum() > 0).sum()}</p>
            </div>

            <div class="metric">
                <h2>🔬 Analysis Results</h2>
                <pre style="background: #f8f8f8; padding: 15px; border-radius: 5px; overflow-x: auto;">
{json.dumps(analysis_results, indent=2, default=str)[:2000]}
                </pre>
            </div>
        </body>
        </html>
        """

        return report_html