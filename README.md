# 📘 Multi-Agent Exploratory Data Analysis (EDA) System
A fully autonomous multi-agent architecture for end-to-end exploratory data analysis with report generation, visualization, data cleaning, and insight synthesis.

## ⭐ Table of Contents
1. Introduction
2. System Features
3. Architecture Overview
4. High-Level Architecture Diagram
5. Pipeline Sequence Diagram
6. Multi-Agent System Design
7. Detailed Agent Documentation
   - JSON/XML Normalizer Agent
   - Data Quality Agent
   - Data Exploration Agent
   - Relationship Discovery Agent
   - Data Storytelling Agent
   - LLM Orchestrator Agent
8. Agent Class Diagrams
9. Deployment Architecture
10. Container / Kubernetes Deployment Diagram
11. Installation
12. API Usage
13. Repository Structure
14. Testing
15. Extensibility
16. Citations

# 🧠 Introduction
This repository provides a complete implementation of a multi-agent system for Exploratory Data Analysis (EDA). The system enables automated:
- Dataset ingestion and normalization
- Data quality assessment and cleaning
- Statistical exploration and clustering
- Relationship discovery and correlation analysis
- AI-powered insight generation
- Interactive dashboard and report creation

It combines statistical analysis, machine learning, visualization libraries, LLM-powered insights, and multi-agent orchestration to produce a robust, extensible, and scalable EDA pipeline.

# 🚀 System Features
✔ Fully autonomous multi-agent workflow
✔ LLM-powered orchestration with dynamic planning
✔ JSON, XML, CSV, Excel dataset ingestion with auto-detection
✔ Comprehensive data quality assessment with Great Expectations
✔ Automated missing-value handling and outlier detection
✔ Smart imputation and duplicate removal
✔ Advanced clustering (K-means with auto cluster selection)
✔ Correlation analysis (Pearson, Spearman, Mutual Information)
✔ Regression analysis with visualizations
✔ AI-generated narrative insights
✔ Interactive Dash dashboards
✔ Professional HTML report generation
✔ REST API powered by FastAPI
✔ Containerized (Docker + Docker Compose)
✔ Supports cloud deployment

# 🏗 Architecture Overview
The system follows a hierarchical multi-agent architecture, coordinated by the LLM Orchestrator Agent.
- Each agent is autonomous with specialized functions
- Agents produce artifacts consumed by other agents
- LLM Orchestrator creates dynamic workflow plans
- A central SharedMemory object facilitates communication

# 🗺 High-Level Architecture Diagram
```mermaid
flowchart TD
    A[User Input: Dataset] --> B[JSON/XML Normalizer Agent]
    B --> C[LLM Orchestrator Agent]
    C --> D[Data Quality Agent]
    C --> E[Data Exploration Agent]
    C --> F[Relationship Discovery Agent]
    C --> G[Data Storytelling Agent]
    D --> I[Shared Memory]
    E --> I
    F --> I
    G --> I
    I --> H[Interactive Dashboard + HTML Report]
```

# 🔁 Pipeline Sequence Diagram
```mermaid
sequenceDiagram
    participant User
    participant Normalizer as JSON/XML Normalizer
    participant Orchestrator as LLM Orchestrator
    participant Quality as Data Quality
    participant Explore as Data Exploration
    participant Relationship as Relationship Discovery
    participant Storytelling as Data Storytelling
    
    User->>Normalizer: Upload dataset (JSON/XML/CSV)
    Normalizer->>Normalizer: auto_detect_and_normalize()
    Normalizer->>Orchestrator: Normalized DataFrame + Metadata
    Orchestrator->>Orchestrator: analyze_dataset_requirements()
    Orchestrator->>Orchestrator: create_workflow_plan()
    Orchestrator->>Quality: Execute quality assessment
    Quality->>Quality: assess_data_quality()
    Quality->>Quality: handle_missing_values()
    Quality->>Quality: detect_outliers()
    Quality->>Orchestrator: Quality Report + Cleaned Data
    Orchestrator->>Explore: Statistical exploration
    Explore->>Explore: generate_summary_statistics()
    Explore->>Explore: perform_clustering()
    Explore->>Explore: plot_distributions()
    Explore->>Orchestrator: Statistics + Visualizations
    Orchestrator->>Relationship: Relationship analysis
    Relationship->>Relationship: compute_correlations()
    Relationship->>Relationship: perform_regression_analysis()
    Relationship->>Orchestrator: Correlation matrices + Plots
    Orchestrator->>Storytelling: Generate insights
    Storytelling->>Storytelling: generate_narrative_summary()
    Storytelling->>Storytelling: create_interactive_dashboard()
    Storytelling->>User: Dashboard + HTML Report
```

# 🤖 Multi-Agent System Design
The architecture uses six specialized agents coordinated by an LLM Orchestrator Agent.

🔹 Agents:

| Agent | Purpose | Key Functions |
|-------|---------|---------------|
| JSON/XML Normalizer Agent | Format detection and data normalization | auto_detect_and_normalize(), flatten_nested_columns() |
| Data Quality Agent | Quality assessment, cleaning, validation | assess_data_quality(), handle_missing_values(), detect_outliers() |
| Data Exploration Agent | Statistical analysis and clustering | generate_summary_statistics(), perform_clustering(), plot_distributions() |
| Relationship Discovery Agent | Correlation and relationship analysis | compute_correlations(), perform_regression_analysis(), compute_mutual_information() |
| Data Storytelling Agent | AI-powered insights and reporting | generate_narrative_summary(), create_interactive_dashboard() |
| LLM Orchestrator Agent | Workflow planning and coordination | create_workflow_plan(), execute_workflow() |

# 📚 Detailed Agent Documentation

## 📄 JSON/XML Normalizer Agent
**Purpose:**
- Auto-detect data format (JSON, XML, CSV, Excel)
- Convert structured data to normalized DataFrame
- Flatten nested structures
- Extract metadata

**Functions:**
- `normalize_json()` - Convert JSON to DataFrame
- `normalize_xml()` - Convert XML to DataFrame
- `auto_detect_and_normalize()` - Auto-detect format and normalize
- `flatten_nested_columns()` - Flatten nested data structures

**Diagram:**
```mermaid
flowchart TD
    A[Input File] --> B{auto_detect_and_normalize}
    B -->|JSON| C[normalize_json]
    B -->|XML| D[normalize_xml]
    B -->|CSV/Excel| E[Direct Load]
    C --> F[flatten_nested_columns]
    D --> F
    E --> F
    F --> G[Return Normalized DataFrame + Metadata]
```

## ✅ Data Quality Agent
**Purpose:**
- Comprehensive data quality assessment
- Missing value imputation
- Duplicate detection and removal
- Outlier detection using statistical methods
- Feature encoding and normalization
- Data validation with Great Expectations

**Functions:**
- `assess_data_quality()` - Comprehensive quality report
- `identify_data_types()` - Categorize column types
- `handle_missing_values()` - Smart imputation strategies
- `handle_duplicates()` - Remove duplicate records
- `detect_outliers()` - Z-score & IQR methods
- `encode_categorical_features()` - Label/One-hot encoding
- `normalize_data()` - StandardScaler normalization
- `validate_with_great_expectations()` - Data validation framework

**Diagram:**
```mermaid
flowchart TD
    A[Input DataFrame] --> B[assess_data_quality]
    B --> C[identify_data_types]
    C --> D[handle_missing_values]
    D --> E[handle_duplicates]
    E --> F[detect_outliers]
    F --> G[encode_categorical_features]
    G --> H[normalize_data]
    H --> I[validate_with_great_expectations]
    I --> J[Quality Report + Cleaned DataFrame]
```

## 🔍 Data Exploration Agent
**Purpose:**
- Generate comprehensive statistical summaries
- Perform clustering analysis with auto cluster selection
- Create distribution visualizations
- Identify outliers with detailed analysis
- Generate categorical frequency plots

**Functions:**
- `generate_summary_statistics()` - Comprehensive statistical summary
- `identify_outliers()` - Outlier detection with detailed reports
- `plot_distributions()` - Histogram visualizations
- `plot_boxplots()` - Box plot analysis
- `plot_countplots()` - Categorical frequency plots
- `perform_clustering()` - K-means with automatic cluster selection
- `create_scatter_matrix()` - Pairwise scatter plot matrix

**Diagram:**
```mermaid
flowchart TD
    A[Cleaned DataFrame] --> B[generate_summary_statistics]
    A --> C[identify_outliers]
    A --> D[perform_clustering]
    B --> E[plot_distributions]
    C --> F[plot_boxplots]
    C --> G[plot_countplots]
    D --> H[create_scatter_matrix]
    E --> I[Statistical Artifacts]
    F --> I
    G --> I
    H --> I
```

## 🔗 Relationship Discovery Agent
**Purpose:**
- Compute multiple correlation metrics
- Identify linear and non-linear relationships
- Perform regression analysis
- Generate relationship visualizations
- Find strong feature relationships

**Functions:**
- `compute_correlations()` - Pearson & Spearman correlations
- `plot_correlation_heatmap()` - Annotated correlation heatmap
- `plot_pairplot()` - Seaborn pairplot for relationships
- `plot_categorical_vs_numerical()` - Box plots by category
- `plot_boxplots_categorical_vs_numerical()` - Multiple category combinations
- `compute_mutual_information()` - Non-linear relationship detection
- `perform_regression_analysis()` - Linear regression with visualizations
- `identify_strong_relationships()` - Find high correlation pairs

**Diagram:**
```mermaid
flowchart TD
    A[Cleaned DataFrame] --> B[compute_correlations]
    A --> C[compute_mutual_information]
    B --> D[plot_correlation_heatmap]
    B --> E[identify_strong_relationships]
    C --> F[plot_pairplot]
    A --> G[perform_regression_analysis]
    A --> H[plot_categorical_vs_numerical]
    D --> I[Relationship Artifacts]
    E --> I
    F --> I
    G --> I
    H --> I
```

## 📊 Data Storytelling Agent
**Purpose:**
- Generate AI-powered narrative insights
- Create interactive dashboards
- Produce professional HTML reports
- Synthesize findings from all agents
- Provide fallback narratives without LLM

**Functions:**
- `generate_narrative_summary()` - AI-powered insights using LLM
- `generate_summary_insights()` - Extract key findings
- `create_interactive_dashboard()` - Full Plotly Dash dashboard
- `create_summary_report()` - Professional HTML report generation
- `_generate_fallback_narrative()` - Non-LLM backup narrative

**Diagram:**
```mermaid
flowchart TD
    A[All Agent Artifacts] --> B[generate_summary_insights]
    B --> C{LLM Available?}
    C -->|Yes| D[generate_narrative_summary]
    C -->|No| E[_generate_fallback_narrative]
    D --> F[create_interactive_dashboard]
    E --> F
    F --> G[create_summary_report]
    G --> H[Interactive Dashboard + HTML Report]
```

## 🎯 LLM Orchestrator Agent
**Purpose:**
- Initialize and coordinate all agents
- Analyze dataset requirements dynamically
- Create optimal workflow plans using LLM or rules
- Execute complete EDA workflow
- Manage shared memory and artifacts

**Functions:**
- `initialize_agents()` - Setup and initialize all agents
- `analyze_dataset_requirements()` - Analyze dataset characteristics
- `create_workflow_plan()` - LLM-based or rule-based workflow planning
- `execute_workflow()` - Complete EDA pipeline execution

**Diagram:**
```mermaid
flowchart TD
    A[Dataset Input] --> B[initialize_agents]
    B --> C[analyze_dataset_requirements]
    C --> D{create_workflow_plan}
    D -->|LLM-based| E[Dynamic Plan]
    D -->|Rule-based| F[Static Plan]
    E --> G[execute_workflow]
    F --> G
    G --> H[Normalizer Agent]
    G --> I[Quality Agent]
    G --> J[Exploration Agent]
    G --> K[Relationship Agent]
    G --> L[Storytelling Agent]
    H --> M[Final Outputs]
    I --> M
    J --> M
    K --> M
    L --> M
```

# 🧩 Agent Class Diagrams
```mermaid
classDiagram
    class LLMOrchestrator {
        +initialize_agents()
        +analyze_dataset_requirements()
        +create_workflow_plan()
        +execute_workflow()
        +shared_memory: SharedMemory
    }
    
    class JSONXMLNormalizerAgent {
        +normalize_json()
        +normalize_xml()
        +auto_detect_and_normalize()
        +flatten_nested_columns()
    }
    
    class DataQualityAgent {
        +assess_data_quality()
        +identify_data_types()
        +handle_missing_values()
        +handle_duplicates()
        +detect_outliers()
        +encode_categorical_features()
        +normalize_data()
        +validate_with_great_expectations()
    }
    
    class DataExplorationAgent {
        +generate_summary_statistics()
        +identify_outliers()
        +plot_distributions()
        +plot_boxplots()
        +plot_countplots()
        +perform_clustering()
        +create_scatter_matrix()
    }
    
    class RelationshipDiscoveryAgent {
        +compute_correlations()
        +plot_correlation_heatmap()
        +plot_pairplot()
        +plot_categorical_vs_numerical()
        +compute_mutual_information()
        +perform_regression_analysis()
        +identify_strong_relationships()
    }
    
    class DataStorytellingAgent {
        +generate_narrative_summary()
        +generate_summary_insights()
        +create_interactive_dashboard()
        +create_summary_report()
        +_generate_fallback_narrative()
    }
    
    LLMOrchestrator --> JSONXMLNormalizerAgent
    LLMOrchestrator --> DataQualityAgent
    LLMOrchestrator --> DataExplorationAgent
    LLMOrchestrator --> RelationshipDiscoveryAgent
    LLMOrchestrator --> DataStorytellingAgent
```

# ☁ Deployment Architecture
```mermaid
flowchart LR
    Client[User / Notebook / UI] --> API[FastAPI REST Service]
    API --> ORCH[LLM Orchestrator Service]
    
    subgraph AGENTS["Agent Worker Pool"]
        A1[JSON/XML Normalizer Agent]
        A2[Data Quality Agent]
        A3[Data Exploration Agent]
        A4[Relationship Discovery Agent]
        A5[Data Storytelling Agent]
    end
    
    ORCH --> A1
    ORCH --> A2
    ORCH --> A3
    ORCH --> A4
    ORCH --> A5
    
    A1 --> STORAGE[(Object Storage)]
    A2 --> STORAGE
    A3 --> STORAGE
    A4 --> STORAGE
    A5 --> STORAGE
    
    subgraph EXTERNAL["External Services"]
        LLM[LLM API - Gemini/GPT]
        GE[Great Expectations]
    end
    
    A2 --> GE
    A5 --> LLM
    ORCH --> LLM
```

# 🐳 Container / Kubernetes Deployment Diagram
```mermaid
flowchart TD
    subgraph Cluster["Kubernetes Cluster"]
        subgraph API["Deployment: API"]
            Pod1["api-pod"]
        end
        
        subgraph Orchestrators["Deployment: LLM Orchestrators"]
            Pod2["orch-1"]
            Pod3["orch-2"]
        end
        
        subgraph Agents["Deployment: Agent Pool"]
            Pod4["normalizer"]
            Pod5["quality"]
            Pod6["exploration"]
            Pod7["relationship"]
            Pod8["storytelling"]
        end
        
        subgraph External["External Services"]
            LLM["LLM Service"]
            GE["Great Expectations"]
        end
        
        SVC_API["Service: API"] --> Pod1
        SVC_ORCH["Service: Orchestrator"] --> Pod2
        SVC_ORCH --> Pod3
        SVC_AGENTS["Service: Agents"] --> Pod4
        SVC_AGENTS --> Pod5
        SVC_AGENTS --> Pod6
        SVC_AGENTS --> Pod7
        SVC_AGENTS --> Pod8
        
        Pod2 --> LLM
        Pod3 --> LLM
        Pod5 --> GE
        Pod8 --> LLM
    end
```

# 🛠 Installation
Clone repository:
```bash
git clone https://github.com/eaamankwah/multi-agent-eda-system
cd multi-agent-eda-system
```

Install requirements:
```bash
pip install -r requirements.txt
```

Set up environment variables:
```bash
# Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env
echo "GREAT_EXPECTATIONS_HOME=./ge_configs" >> .env
```

# 🌐 Run API Server
```bash
uvicorn src.app:app --reload
```

# 🔌 API Usage
Run EDA via cURL:
```bash
curl -X POST -F "file=@dataset.csv" http://127.0.0.1:8000/eda/run
```

Run with JSON data:
```bash
curl -X POST -F "file=@data.json" http://127.0.0.1:8000/eda/run
```

Run with XML data:
```bash
curl -X POST -F "file=@data.xml" http://127.0.0.1:8000/eda/run
```

# 📁 Repository Structure

```
multi-agent-eda-system/
multi-agent-eda-system/
│
├── deployment files/
│   ├── app.py                                      # FastAPI application
│   ├── cloudbuild.yaml                             # Cloud Build configuration
│   ├── DEPLOYMENT_GUIDE.md                         # Deployment documentation
│   ├── Dockerfile                                  # Container definition
│   ├── eda_results.json                            # Sample EDA results
│   ├── eda_summary_report.html                     # Sample HTML report
│   ├── evaluation_report.txt                       # Evaluation metrics
│   └── requirements.txt                            # Python dependencies
│
├── docker-compose.yml                              # Docker Compose configuration
├── Dockerfile                                      # Main Dockerfile
│
├── docs/
│   ├── architecture.md                             # System architecture
│   ├── deployment.md                               # Deployment guide
│   └── usage_guide.md                              # User guide
│
├── images/
│   ├── agent system gallery.png                    # Agent system overview
│   ├── agent system gallery2.png                   # Agent interactions
│   ├── agent system gallery3.png                   # Workflow visualization
│   ├── agent system gallery4.png                   # Results dashboard
│   ├── architecture diagram.png                    # Architecture diagram
│   ├── deployment architecture.png                 # Deployment architecture
│   ├── Gemini_Generated_multi-agent eda system.png # Gemini generated diagram
│   ├── llm orchestrator agent.png                  # Orchestrator agent diagram
│   ├── pipeline sequence diagram.png               # Pipeline sequence flow
│   └── repo structure.png                          # Repository structure
│   └── multi-agent system test coverage.jpeg       # Test system coverage
│
├── Multi_Agent_Exploratory_Data_Analysis_System.html  # Main documentation
├── Multi_Agent_Exploratory_Data_Analysis_System.ipynb # Jupyter notebook
├── Multi_Agent_Exploratory_Data_Analysis_System.py    # Python script version
│
├── README.md                                       # This file
├── requirements.txt                                # Python dependencies
│
├── src/
│   ├── agents/
│   │   ├── exploration_agent.py                    # Data Exploration Agent
│   │   ├── normalizer_agent.py                     # JSON/XML Normalizer Agent
│   │   ├── quality_agent.py                        # Data Quality Agent
│   │   ├── relationship_agent.py                   # Relationship Discovery Agent
│   │   └── storytelling_agent.py                   # Data Storytelling Agent
│   │
│   ├── app.py                                      # FastAPI application entry
│   │
│   ├── orchestrator/
│   │   └── llm_orchestrator.py                     # LLM Orchestrator Agent
│   │
│   └── utils/
│       ├── config.py                               # Configuration settings
│       └── shared_memory.py                        # Shared memory manager
│
└── tests/
    └── test_agents.py                              # Agent unit tests
    └── test_normalizer.py                          
    └── test_quality.py                              
    └── test_exploration.py  
    └── test_relationship.py  
    └── test_storytelling.py                                                       
```

# 🧪 Testing
![Multi-Agent System Test Coverage](images/multi_agent_system_test_coverage.jpeg)
Run all tests:
```bash
pytest tests/ -v
```

Run specific multi-agent system test coverage:
```bash
pytest tests/test_normalizer.py -v
pytest tests/test_quality.py -v
pytest tests/test_exploration.py -v
pytest tests/test_relationship.py -v
pytest tests/test_storytelling.py -v
```

# 🧩 Extensibility
You may extend the system by adding:

🔹 **Additional Agents**
- Feature Engineering Agent - Advanced feature creation and selection
- Time Series Analysis Agent - Temporal pattern detection
- Model Training Agent - AutoML integration
- Data Drift Agent - Monitor data distribution changes
- Anomaly Detection Agent - Advanced outlier detection

🔹 **Frontend Dashboard**
- React dashboard for interactive exploration
- Streamlit interface for quick prototyping
- Jupyter notebook integration

🔹 **Enhanced LLM Integration**
- Multi-model support (GPT-4, Claude, Gemini)
- Fine-tuned models for domain-specific insights
- Multi-turn reasoning for deeper analysis
- Chain-of-thought prompting for complex queries
- Model Context Protocol for improved performance

🔹 **Advanced Features**
- Real-time streaming data support
- Distributed processing with Dask
- GPU acceleration for large datasets
- MLflow integration for experiment tracking

# 📚 Citations

- Provost, Fawcett — Data Science for Business
- McKinney — Python for Data Analysis
- Scikit-Learn Documentation
- Pandas Documentation
- Plotly & Dash Documentation
- Great Expectations Documentation
- FastAPI Documentation
- Seaborn & Matplotlib Documentation
- Google DeepMind Gemini Papers
- OpenAI GPT Documentation