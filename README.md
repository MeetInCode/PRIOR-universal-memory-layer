# CORTEX - Knowledge Graph Memory System


Here is a gist of entire project architecture and data flow -

## [1] INPUT & CONTEXT LAYER
<img width="1602" height="512" alt="image" src="https://github.com/user-attachments/assets/df665c31-5c9e-4444-87f1-0db9e738bac8" />

## [2] INTENT & EXTRACTION [3] HYBRID SEARCH ENGINE [4] GRAPH MUTATION ENGINE
<img width="1452" height="877" alt="image" src="https://github.com/user-attachments/assets/a434081a-55bd-4f55-9178-eb9d8dfd6c74" />

## [5] SYNTHESIS & OUTPUT
<img width="1318" height="265" alt="image" src="https://github.com/user-attachments/assets/4105e421-4578-458c-b837-cfdebb6cafa9" />

## Data Orchestrator
<img width="783" height="819" alt="image" src="https://github.com/user-attachments/assets/fc98127a-6d07-499a-8de2-b00398e002f2" />


## Architecture
<img width="3074" height="1945" alt="image" src="https://github.com/user-attachments/assets/8ddcef94-0f3b-4755-a0e1-25f20f1c78c3" />

## just a demo mcp simulation with neo4j running in docker
https://drive.google.com/file/d/16S8A5jCClMnLZV2_-wZ_pFYyA3gEOq9h/view?usp=sharing

# reference - https://arxiv.org/pdf/2504.19413
## 🚀 Quick Start



```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment variables in .env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
GROQ_API_KEY=your_groq_api_key

# 3. Run the app
streamlit run cortex_app.py
```

## 📁 Project Structure

```
knowledgegraph/
├── cortex_app.py        # Streamlit UI
├── cortex_memory.py     # Core memory layer (Neo4j + Embeddings)
├── graph_rag.py         # GraphRAG utilities (optional)
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables
└── README.md           # This file
```

## 🧠 How It Works

### Architecture Flow

1. **User Message** → Extract keywords using LLM + schema awareness
2. **Keyword Search** → Query Neo4j for relevant nodes/edges
3. **Context Retrieval** → Build context from graph relationships
4. **Response Generation** → LLM generates response with context
5. **Action Classification** → Decide: NO-OP, INGEST, UPDATE, DELETE
6. **Graph Operations** → Execute with embeddings stored per node

### Key Features

- **Schema-Aware**: Uses live graph schema for keyword extraction
- **Keyword-Enhanced Nodes**: Each node has 5+ keywords for better retrieval
- **Soft Deletes**: Deleted nodes are marked, not removed
- **Embeddings**: Vector embeddings stored on each node for semantic search
- **Session Memory**: Chat history maintained per session

## 💬 Example Interactions

```
User: Hi, I'm John and I work at TechCorp as a software engineer
CORTEX: Nice to meet you, John! I've noted that you work at TechCorp as a software engineer.
[Created Person/john, Linked User->WORKS_AT->Person/john]

User: My birthday is March 15th
CORTEX: Got it! I'll remember your birthday is on March 15th.
[Created Event/birthday-march-15, Linked User->HAS_EVENT->Event/birthday-march-15]

User: What do you know about me?
CORTEX: I know you're John, a software engineer at TechCorp, and your birthday is March 15th.
```

## 🔧 Configuration

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `NEO4J_URI` | Neo4j connection URI (e.g., `bolt://localhost:7687`) |
| `NEO4J_USERNAME` | Neo4j username |
| `NEO4J_PASSWORD` | Neo4j password |
| `GROQ_API_KEY` | Groq API key for LLM |

### Optional Configuration

Edit `cortex_memory.py` to change:
- `GROQ_MODEL`: Default is `llama-3.3-70b-versatile`
- Embedding model: Default is `all-MiniLM-L6-v2`

## 📊 Graph Schema

The system automatically creates and uses these node types:

- **User**: Core user node (id = email)
- **Person**: People mentioned
- **Location**: Places
- **Event**: Dates, birthdays, events
- **Skill**: Skills, technologies
- **Organization**: Companies, institutions
- **Memory**: Raw message chunks with embeddings

Relationship types are auto-generated in CAPS_SNAKE_CASE (e.g., `WORKS_AT`, `LIVES_IN`, `KNOWS`).

## 🛠 Development

### Running Tests

```bash
python -m pytest tests/
```

### Checking Database Connection

```bash
python check_db_connection.py
```





## 📝 License

MIT License
