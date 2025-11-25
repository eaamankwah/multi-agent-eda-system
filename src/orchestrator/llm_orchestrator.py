class LLMOrchestratorAgent(BaseAgent):
    """Orchestrates the entire EDA workflow using Gemini"""

    def __init__(self):
        super().__init__("LLM Orchestrator", Config.MODEL_NAME)
        self.agents = {}
        self.workflow_plan = []
        self.results = {}

    def initialize_agents(self):
        self.log("Initializing specialized agents...")
        self.agents = {
            'normalizer': JSONXMLNormalizerAgent(),
            'quality': DataQualityAgent(),
            'exploration': DataExplorationAgent(),
            'relationships': RelationshipDiscoveryAgent(),
            'storytelling': DataStorytellingAgent()
        }
        self.log(f"Initialized {len(self.agents)} agents")

    def analyze_dataset_requirements(self, df: pd.DataFrame) -> Dict:
        """Analyze dataset and determine analysis requirements"""
        self.log("Analyzing dataset requirements...")

        requirements = {
            'shape': df.shape,
            'has_numerical': df.select_dtypes(include=[np.number]).shape[1] > 0,
            'has_categorical': df.select_dtypes(include=['object', 'category']).shape[1] > 0,
            'has_missing': df.isnull().sum().sum() > 0,
            'has_duplicates': df.duplicated().sum() > 0,
            'recommended_analyses': []
        }

        # Determine recommended analyses
        if requirements['has_numerical']:
            requirements['recommended_analyses'].extend([
                'summary_statistics', 'distributions', 'correlations',
                'clustering', 'regression'
            ])

        if requirements['has_categorical']:
            requirements['recommended_analyses'].extend([
                'categorical_analysis', 'groupby_analysis'
            ])

        if requirements['has_missing'] or requirements['has_duplicates']:
            requirements['recommended_analyses'].insert(0, 'data_quality')

        return requirements

    def _validate_workflow(self, workflow_steps: List[Dict]) -> bool:
        """Validates if all actions in a workflow exist in their respective agents."""
        if not self.agents:
            self.initialize_agents() # Ensure agents are initialized for validation

        for step in workflow_steps:
            agent_name = step.get('agent')
            action = step.get('action')
            if not agent_name or not action:
                self.log(f"Validation failed: Invalid workflow step: missing agent or action in {step}")
                return False
            agent = self.agents.get(agent_name)
            if not agent:
                self.log(f"Validation failed: Agent '{agent_name}' not found for action '{action}'")
                return False
            if not hasattr(agent, action) or not callable(getattr(agent, action)):
                self.log(f"Validation failed: Agent '{agent_name}' has no callable method '{action}'")
                # For debugging: List available methods
                # self.log(f"  Available methods for {agent_name}: {[m for m in dir(agent) if not m.startswith('_') and callable(getattr(agent, m))]}")
                return False
        return True

    def create_workflow_plan(self, df: pd.DataFrame) -> List[Dict]:
        """Create intelligent workflow plan using LLM or rule-based fallback"""
        self.log("Creating workflow plan...")

        # Ensure agents are initialized before LLM planning or rule-based fallback
        if not self.agents:
            self.initialize_agents()

        requirements = self.analyze_dataset_requirements(df)

        # Define the rule-based fallback workflow here to ensure it's always available
        rule_based_workflow = [
            # Data Quality & Preprocessing Steps (High Priority)
            {'agent': 'quality', 'action': 'assess_data_quality', 'priority': 5},
            {'agent': 'quality', 'action': 'handle_duplicates', 'priority': 5, 'parameters': {}},
            {'agent': 'quality', 'action': 'handle_missing_values', 'priority': 5, 'parameters': {'strategy': 'smart'}},
            {'agent': 'quality', 'action': 'identify_data_types', 'priority': 4},
            {'agent': 'quality', 'action': 'encode_categorical_features', 'priority': 4, 'parameters': {'method': 'label'}},
            {'agent': 'quality', 'action': 'normalize_data', 'priority': 4},
            {'agent': 'quality', 'action': 'detect_outliers', 'priority': 3, 'parameters': {'method': 'zscore'}},

            # Data Exploration Steps (Medium Priority)
            {'agent': 'exploration', 'action': 'generate_summary_statistics', 'priority': 4},
            {'agent': 'exploration', 'action': 'plot_distributions', 'priority': 3},
            {'agent': 'exploration', 'action': 'plot_boxplots', 'priority': 3},
            {'agent': 'exploration', 'action': 'plot_countplots', 'priority': 3},
            {'agent': 'exploration', 'action': 'perform_clustering', 'priority': 3},
            {'agent': 'exploration', 'action': 'create_scatter_matrix', 'priority': 2},


            # Relationship Analysis Steps (Medium-Low Priority)
            {'agent': 'relationships', 'action': 'compute_correlations', 'priority': 3, 'parameters': {'plot_heatmap': True}},
            {'agent': 'relationships', 'action': 'plot_pairplot', 'priority': 2},
            {'agent': 'relationships', 'action': 'plot_boxplots_categorical_vs_numerical', 'priority': 2},
            {'agent': 'relationships', 'action': 'identify_strong_relationships', 'priority': 2},
            # Removed target_col for generic fallback
            {'agent': 'relationships', 'action': 'perform_regression_analysis', 'priority': 2, 'parameters': {}},
            {'agent': 'relationships', 'action': 'compute_mutual_information', 'priority': 2, 'parameters': {}},

            # Storytelling Steps (Lowest Priority, but always included)
            {'agent': 'storytelling', 'action': 'generate_narrative_summary', 'priority': 1},
            {'agent': 'storytelling', 'action': 'create_summary_report', 'priority': 1},
            {'agent': 'storytelling', 'action': 'create_interactive_dashboard', 'priority': 1}
        ]

        # Temporarily use only rule-based workflow to fix AttributeError
        self.workflow_plan = sorted(rule_based_workflow, key=lambda x: x.get('priority', 0), reverse=True)
        self.log(f"Using rule-based workflow with {len(self.workflow_plan)} steps (LLM planning temporarily bypassed due to method naming issues)")
        return self.workflow_plan

    def execute_workflow(self, df: pd.DataFrame) -> Dict:
        self.log("=" * 60)
        self.log("EXECUTING MULTI-AGENT EDA WORKFLOW")
        self.log("=" * 60)

        if not self.agents:
            self.initialize_agents()

        if not self.workflow_plan:
            # This path is now mostly for first run or if explicitly cleared
            self.create_workflow_plan(df)

        execution_results = {
            'steps_completed': [],
            'steps_failed': [],
            'agent_results': {}
        }

        current_df = df.copy()

        for step in self.workflow_plan:
            agent_name = step['agent']
            action = step['action']

            try:
                self.log(f"\n>>> Executing: {agent_name}.{action}")
                agent = self.agents.get(agent_name)

                if not agent:
                    raise ValueError(f"Agent '{agent_name}' not found")

                if not hasattr(agent, action):
                    raise AttributeError(f"Agent '{agent_name}' has no method '{action}'")

                method = getattr(agent, action)

                # Get parameters from the workflow step, default to an empty dict if not present
                params = step.get('parameters', {})

                # Handle methods that modify the DataFrame
                if action in ['handle_missing_values', 'handle_duplicates', 'normalize_data', 'encode_categorical_features']:
                    result = method(current_df, **params) # Pass parameters here
                    if isinstance(result, pd.DataFrame):
                        current_df = result
                        self.log(f"DataFrame updated: {current_df.shape}")
                # Handle methods that need analysis_results
                elif action in ['generate_summary_insights', 'generate_narrative_summary', 'create_interactive_dashboard', 'create_summary_report']:
                    result = method(current_df, execution_results.get('agent_results', {}), **params) # Pass parameters here
                else:
                    result = method(current_df, **params) # Pass parameters here

                execution_results['steps_completed'].append({
                    'agent': agent_name,
                    'action': action,
                    'status': 'success'
                })

                if agent_name not in execution_results['agent_results']:
                    execution_results['agent_results'][agent_name] = {}

                # Store result (avoid storing large DataFrames)
                if isinstance(result, pd.DataFrame):
                    result_summary = f"DataFrame: {result.shape}"
                elif isinstance(result, dict):
                    result_summary = result
                elif isinstance(result, str):
                    result_summary = result[:500] + "..." if len(result) > 500 else result
                else:
                    result_summary = str(result)[:500]

                execution_results['agent_results'][agent_name][action] = {
                    'result': result_summary
                }

                self.log(f"\u2713 {action} completed successfully")

            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                self.log(f"\u2718 {action} failed: {e}")
                self.log(f"Error details: {error_detail}")
                execution_results['steps_failed'].append({
                    'agent': agent_name,
                    'action': action,
                    'error': str(e),
                    'details': error_detail
                })

        self.log("\n" + "=" * 60)
        self.log("WORKFLOW EXECUTION COMPLETE")
        self.log(f"Steps completed: {len(execution_results['steps_completed'])}")
        self.log(f"Steps failed: {len(execution_results['steps_failed'])}")

        if execution_results['steps_failed']:
            self.log("\nFailed steps:")
            for failed in execution_results['steps_failed']:
                self.log(f"  - {failed['agent']}.{failed['action']}: {failed['error']}")

        self.log("=" * 60)

        self.results = execution_results
        return execution_results