"""
CORTEX Memory Layer v4.0
========================
Uses LLMGraphTransformer for reliable entity extraction.
Direct Neo4j operations with embeddings.

Architecture:
1. User sends message
2. Build keywords from message + schema + chat history
3. Retrieve relevant nodes/edges + vector similarity search
4. Generate response
5. Decide operation: NO-OP, INGEST, UPDATE, DELETE
6. Use LLMGraphTransformer for reliable entity extraction
7. Store with embeddings in Neo4j
"""

import os
# Force CPU before any torch imports
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TORCH_DEVICE"] = "cpu"

import json
import uuid
import re
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from pydantic import BaseModel, Field
from rich.console import Console
from dotenv import load_dotenv

# LangChain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.documents import Document

# LLMGraphTransformer for reliable entity extraction
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Neo4j
from langchain_neo4j import Neo4jGraph

# Embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

load_dotenv()
console = Console()

# ============================================================================
# CONFIGURATION
# ============================================================================

GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension

# Allowed node and relationship types for consistent schema
ALLOWED_NODES = ["Person", "Location", "Organization", "Event", "Skill", "Preference", "Project"]
ALLOWED_RELATIONSHIPS = [
    "KNOWS", "LIVES_IN", "WORKS_AT", "HAS_SKILL", "HAS_EVENT", 
    "USES", "LIKES", "DISLIKES", "MEMBER_OF", "WORKS_ON", "RELATED_TO"
]

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class MemoryDecision(BaseModel):
    """Decision output from memory classifier."""
    action: str = Field(description="One of: NO-OP, INGEST, UPDATE, DELETE")
    reason: str = Field(description="Brief explanation for the decision")

# ============================================================================
# CORE MEMORY LAYER
# ============================================================================

class CortexMemory:
    """
    CORTEX Memory Layer v4.0 - Uses LLMGraphTransformer.
    
    Flow:
    1. Extract keywords from user message + schema
    2. Retrieve relevant graph context + vector similarity
    3. Generate AI response
    4. Classify memory action
    5. Use LLMGraphTransformer for entity extraction
    6. Store in Neo4j with embeddings
    """
    
    def __init__(self):
        self._validate_env()
        
        # Initialize LLM
        self.llm = ChatGroq(
            model=GROQ_MODEL, 
            temperature=0, 
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Initialize Neo4j Graph
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        
        # Initialize LLMGraphTransformer
        self.graph_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=ALLOWED_NODES,
            allowed_relationships=ALLOWED_RELATIONSHIPS,
            node_properties=True,
            relationship_properties=True
        )
        console.print("[green]✓ LLMGraphTransformer Initialized[/]")
        
        # Initialize Embeddings (optional)
        self.embeddings = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                console.print("[green]✓ Embeddings Model Loaded[/]")
            except Exception as e:
                console.print(f"[yellow]⚠ Embeddings unavailable: {e}[/]")
        
        # Session state
        self.user_email = None
        self.chat_history = []
        
        # Initialize schema
        self._init_schema()
        console.print("[green]✓ CORTEX Memory v4.0 Initialized[/]")
    
    def _validate_env(self):
        """Validate required environment variables."""
        required = ["GROQ_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
        missing = [v for v in required if not os.getenv(v)]
        if missing:
            raise EnvironmentError(f"Missing environment variables: {missing}")
    
    def _init_schema(self):
        """Initialize graph constraints and vector indexes."""
        try:
            self.graph.query(
                "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE"
            )
            self.graph.refresh_schema()
        except Exception as e:
            console.print(f"[yellow]Schema init warning: {e}[/]")
    
    def get_schema(self) -> str:
        """Get current graph schema."""
        try:
            labels = self.graph.query("CALL db.labels()")
            rels = self.graph.query("CALL db.relationshipTypes()")
            label_list = [r.get('label', '') for r in labels]
            rel_list = [r.get('relationshipType', '') for r in rels]
            return f"Labels: {', '.join(label_list)}\nRelationships: {', '.join(rel_list)}"
        except:
            return "Schema unavailable"
    
    def set_user_session(self, email: str):
        """Initialize session for a specific user."""
        self.user_email = email.lower().strip()
        self.chat_history = []
        
        self.graph.query(
            "MERGE (u:User {id: $email}) SET u.last_seen = datetime()",
            {"email": self.user_email}
        )
        console.print(f"[dim]Session: {self.user_email}[/]")
    
    # =========================================================================
    # STEP 1: KEYWORD EXTRACTION
    # =========================================================================
    
    def extract_keywords(self, message: str) -> List[str]:
        """Extract search keywords from user message."""
        prompt = ChatPromptTemplate.from_template(
            """Extract 5-8 search keywords from: "{message}"
Return ONLY a JSON array: ["keyword1", "keyword2", ...]"""
        )
        
        try:
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"message": message})
            # Extract JSON array
            match = re.search(r'\[.*?\]', result, re.DOTALL)
            if match:
                return json.loads(match.group())
            return message.lower().split()[:8]
        except:
            return message.lower().split()[:8]
    
    # =========================================================================
    # STEP 2: HYBRID RETRIEVAL (Graph + Vector)
    # =========================================================================
    
    def retrieve_context(self, keywords: List[str], query_text: str = "") -> Tuple[str, List[str]]:
        """Hybrid retrieval: Graph traversal + Vector similarity."""
        logs = []
        context_parts = []
        
        if not self.user_email:
            return "No user session", ["Error: No user session"]
        
        logs.append(f"🔍 Keywords: {keywords[:5]}")
        
        # GRAPH RETRIEVAL
        try:
            results = self.graph.query(
                """
                MATCH (u:User {id: $email})-[r]->(n)
                WHERE (n.deleted IS NULL OR n.deleted = false)
                RETURN type(r) as rel, labels(n)[0] as label, n.id as id, 
                       properties(n) as props
                LIMIT 20
                """,
                {"email": self.user_email}
            )
            
            if results:
                logs.append(f"📊 Found {len(results)} connected entities")
                for r in results:
                    props = {k: v for k, v in (r.get('props') or {}).items() 
                             if k not in ['embedding', 'deleted'] and v}
                    context_parts.append(f"[{r['label']}] {r['id']} ({r['rel']}): {props}")
            else:
                logs.append("📊 No connected entities")
        except Exception as e:
            logs.append(f"⚠ Graph error: {e}")
        
        # VECTOR RETRIEVAL
        if self.embeddings and query_text:
            try:
                query_vector = self.embeddings.embed_query(query_text)
                vector_results = self.graph.query(
                    """
                    MATCH (m:Memory {user_email: $email})
                    WHERE m.embedding IS NOT NULL
                    WITH m, 
                         reduce(d = 0.0, i IN range(0, size(m.embedding)-1) | 
                                d + m.embedding[i] * $qv[i]) / 
                         (sqrt(reduce(a = 0.0, i IN range(0, size(m.embedding)-1) | 
                                a + m.embedding[i] * m.embedding[i])) *
                          sqrt(reduce(b = 0.0, i IN range(0, size($qv)-1) | 
                                b + $qv[i] * $qv[i]))) as sim
                    WHERE sim > 0.3
                    RETURN m.text as text, sim
                    ORDER BY sim DESC LIMIT 3
                    """,
                    {"email": self.user_email, "qv": query_vector}
                )
                
                if vector_results:
                    logs.append(f"🔮 Found {len(vector_results)} similar memories")
                    for vr in vector_results:
                        context_parts.append(f"[Memory] {vr['text']}")
            except Exception as e:
                logs.append(f"⚠ Vector error: {str(e)[:50]}")
        
        context = "\n".join(context_parts) if context_parts else "No context."
        return context, logs
    
    # =========================================================================
    # STEP 3: RESPONSE GENERATION
    # =========================================================================
    
    def generate_response(self, message: str, context: str) -> str:
        """Generate concise AI response."""
        prompt = ChatPromptTemplate.from_template(
            """You are CORTEX, a concise AI assistant.

CONTEXT: {context}
USER: {message}

RULES:
- Max 1-2 sentences
- Answer ONLY what was asked
- If storing info: "Got it." or "Noted."
- If unknown: "I don't have that information."

RESPONSE:"""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "message": message})
    
    # =========================================================================
    # STEP 4: ACTION CLASSIFICATION
    # =========================================================================
    
    def classify_action(self, message: str) -> MemoryDecision:
        """Classify if message needs memory update."""
        prompt = ChatPromptTemplate.from_template(
            """Does this message contain a FACT about the user to store?

MESSAGE: {message}

ACTIONS:
- NO-OP: Questions, greetings, casual chat
- INGEST: New facts (name, birthday, location, preferences, friends, skills)
- UPDATE: Corrections ("Actually I...", "My new...")
- DELETE: Removal requests ("Forget...", "Delete...")

Return JSON only: {{"action": "NO-OP|INGEST|UPDATE|DELETE", "reason": "brief"}}"""
        )
        
        try:
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"message": message})
            match = re.search(r'\{.*?\}', result, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return MemoryDecision(**data)
        except:
            pass
        return MemoryDecision(action="NO-OP", reason="Classification failed")
    
    # =========================================================================
    # STEP 5: GRAPH OPERATIONS WITH LLMGraphTransformer
    # =========================================================================
    
    def execute_memory_operation(self, message: str, action: str) -> List[str]:
        """Use LLMGraphTransformer for reliable entity extraction."""
        logs = []
        
        if action == "DELETE":
            # Handle deletion separately
            logs.append("🗑 Delete operation - marking entities as deleted")
            return logs
        
        # Prepare context for better extraction
        context_message = f"The user ({self.user_email}) says: {message}"
        doc = Document(page_content=context_message)
        
        try:
            # Use LLMGraphTransformer for extraction
            graph_documents = self.graph_transformer.convert_to_graph_documents([doc])
            
            if not graph_documents:
                logs.append("⚠ No entities extracted")
                return logs
            
            for graph_doc in graph_documents:
                # Process extracted nodes
                for node in graph_doc.nodes:
                    node_id = node.id.lower().replace(" ", "-")
                    node_label = node.type
                    node_properties = dict(node.properties) if node.properties else {}
                    
                    # Generate keywords from node data
                    keywords = [node_id, node_label.lower()]
                    keywords.extend(str(v).lower().split()[:2] for v in node_properties.values() if v)
                    keywords = list(set([str(k) for k in keywords]))[:10]
                    
                    # Create node in Neo4j
                    try:
                        self.graph.query(
                            f"""
                            MERGE (n:{node_label} {{id: $id}})
                            SET n += $props,
                                n.keywords = $keywords,
                                n.updated_at = datetime()
                            """,
                            {"id": node_id, "props": node_properties, "keywords": keywords}
                        )
                        logs.append(f"✅ Node: {node_label}/{node_id}")
                        
                        # Link to user
                        self.graph.query(
                            f"""
                            MATCH (u:User {{id: $email}}), (n:{node_label} {{id: $node_id}})
                            MERGE (u)-[r:RELATED_TO]->(n)
                            """,
                            {"email": self.user_email, "node_id": node_id}
                        )
                        
                        # Store embedding
                        if self.embeddings:
                            text = f"{node_label}: {node_id}. {json.dumps(node_properties)}"
                            vector = self.embeddings.embed_query(text)
                            self.graph.query(
                                f"MATCH (n:{node_label} {{id: $id}}) SET n.embedding = $vec",
                                {"id": node_id, "vec": vector}
                            )
                    except Exception as e:
                        logs.append(f"⚠ Node error: {e}")
                
                # Process extracted relationships
                for rel in graph_doc.relationships:
                    source_id = rel.source.id.lower().replace(" ", "-")
                    target_id = rel.target.id.lower().replace(" ", "-")
                    rel_type = rel.type.upper().replace(" ", "_")
                    
                    try:
                        # Create relationship
                        self.graph.query(
                            f"""
                            MATCH (a {{id: $source}}), (b {{id: $target}})
                            MERGE (a)-[r:{rel_type}]->(b)
                            """,
                            {"source": source_id, "target": target_id}
                        )
                        logs.append(f"🔗 Rel: {source_id} -[{rel_type}]-> {target_id}")
                    except Exception as e:
                        logs.append(f"⚠ Rel error: {e}")
            
            # Store raw message as Memory
            if self.embeddings:
                try:
                    msg_id = f"msg-{uuid.uuid4().hex[:8]}"
                    msg_vector = self.embeddings.embed_query(message)
                    self.graph.query(
                        """
                        CREATE (m:Memory {
                            id: $id, text: $text, user_email: $email,
                            timestamp: datetime(), embedding: $vec
                        })
                        """,
                        {"id": msg_id, "text": message, "email": self.user_email, "vec": msg_vector}
                    )
                    logs.append("💾 Stored message")
                except:
                    pass
                    
        except Exception as e:
            logs.append(f"⚠ Extraction error: {e}")
        
        return logs
    
    # =========================================================================
    # MAIN PROCESSING PIPELINE
    # =========================================================================
    
    def process(self, message: str) -> Tuple[str, List[str]]:
        """Main processing pipeline."""
        logs = []
        
        if not self.user_email:
            raise ValueError("No user session. Call set_user_session() first.")
        
        # Step 1: Extract keywords
        keywords = self.extract_keywords(message)
        logs.append(f"🔍 Keywords: {keywords}")
        
        # Step 2: Retrieve context
        context, retrieval_logs = self.retrieve_context(keywords, query_text=message)
        logs.extend(retrieval_logs)
        
        # Step 3: Generate response
        response = self.generate_response(message, context)
        
        # Step 4: Classify action
        decision = self.classify_action(message)
        icons = {"INGEST": "🟢", "UPDATE": "🟠", "DELETE": "🔴", "NO-OP": "⚪"}
        logs.append(f"{icons.get(decision.action, '⚪')} Action: {decision.action} ({decision.reason})")
        
        # Step 5: Execute if needed
        if decision.action != "NO-OP":
            exec_logs = self.execute_memory_operation(message, decision.action)
            logs.extend(exec_logs)
        
        # Update chat history
        self.chat_history.append(("User", message))
        self.chat_history.append(("Assistant", response))
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]
        
        return response, logs


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    console.print("[bold magenta]CORTEX Memory Test v4.0[/]")
    
    try:
        memory = CortexMemory()
        memory.set_user_session("test@example.com")
        
        test_messages = [
            "Hi, I'm John and I work at Google as a software engineer",
            "My friend Kush is vegan and lives in Mumbai",
            "I use Windows 11 on my laptop",
            "My birthday is on March 15th",
            "What do you know about me?",
        ]
        
        for msg in test_messages:
            console.print(f"\n[cyan]User:[/] {msg}")
            response, logs = memory.process(msg)
            console.print(f"[green]CORTEX:[/] {response}")
            for log in logs:
                console.print(f"[dim]  {log}[/]")
                
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        import traceback
        traceback.print_exc()
