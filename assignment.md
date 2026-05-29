## Objective
Build a RAG or Agentic AI system that allows users to query UK public procurement data using natural language.
The system must use the FTS dataset from the Open Contracting Data Registry:
https://data.open-contracting.org/en/publication/41

## Data Source (Mandatory)
Use the dataset from:
- Open Contracting Partnership Data Registry
- Publication: UK Find a Tender Service
 
## Important Notes for Candidates
- Data follows OCDS schema
- Expect:
  - Deeply nested JSON
  - Multiple “releases” per contract
  - Fields like:
    - tender
    - awards
    - contracts
    - parties
- Not all fields are consistently populated

### 1. Data Ingestion & Normalisation
- Parse OCDS JSON structure
- Flatten or model key entities:
  - Contracts
  - Buyers
  - Suppliers
  - Categories (CPV)
- Handle:
  - Missing fields
  - Duplicate releases
  - Long descriptions

Strong candidates will:
- Create a clean “contract-level” dataset
- Design a schema for retrieval

### 2. Retrieval System
RAG
- Chunk contract text
- Generate embeddings
- Retrieve top-k relevant contracts

Agentic AI
- Build an agent that:
- Interprets query intent
- Applies filters (region, category, etc.)
- Queries data iteratively

### 3. Natural Language Interface
Support queries like:
- “Find framework agreements for IT services in the NHS”
- “Contracts awarded in London for construction”
- “Multi-supplier frameworks for cloud services”

System should:
- Understand intent (semantic + structured)
- Map to relevant contracts

### 4. Output Requirements
For each query:

Contract Results:
- Notice ID
- Title
- Summary (LLM-generated or extracted)
- Relevance score

Explanation:
- Why it matched:
- Semantic similarity
- Metadata filters
- Keyword overlap

### 5. Evaluation
Explain:
- How you evaluate retrieval quality
- Ranking strategy
- Failure cases

Optional:
- Precision@k
- Manual evaluation set
- LLM-as-judge