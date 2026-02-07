import logging
import os
import sys
from typing import List, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv

# Suppress Neo4j driver warnings
logging.getLogger("neo4j").setLevel(logging.ERROR)
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Console setup
console = Console()

# --- Configuration & Setup ---

def setup_environment():
    """Load and verify environment variables."""
    load_dotenv()
    required = ["GROQ_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        console.print(f"[bold red]Missing environment variables:[/] {', '.join(missing)}")
        sys.exit(1)

# --- Models ---

class Entities(BaseModel):
    """Identifying information about entities."""
    names: List[str] = Field(
        ...,
        description="All the person, organization, location, or business entities that appear in the text",
    )

class GraphNode(BaseModel):
    id: str = Field(..., description="Unique identifier for the node (e.g., 'Meet', 'Stanford')")
    type: str = Field(..., description="Type of the node (e.g., 'Person', 'Organization')")
    properties: Optional[dict] = Field({}, description="Additional attributes of the node")

class GraphEdge(BaseModel):
    source: str = Field(..., description="ID of the source node")
    target: str = Field(..., description="ID of the target node")
    type: str = Field(..., description="Type of the relationship (e.g., 'STUDIES_AT', 'INTERESTED_IN')")

# --- Memory Tools ---

class AddKnowledge(BaseModel):
    """Add new entities and relationships to the graph."""
    nodes: List[GraphNode] = Field(..., description="List of new nodes to add")
    edges: List[GraphEdge] = Field(..., description="List of new relationships to add")

class UpdateKnowledge(BaseModel):
    """Update existing specific nodes with new properties or fix contradictions."""
    entity_id: str = Field(..., description="The ID of the entity to update")
    properties: dict = Field(..., description="New properties to merge or overwrite (e.g., {'status': 'active'})")
    reason: str = Field(..., description="Why this update is needed")

class DeleteEdge(BaseModel):
    source: str = Field(..., description="ID of the source node")
    target: Optional[str] = Field(None, description="ID of the target node. Leave None/Null to delete ALL relationships of this type from the source.")
    type: str = Field(..., description="Type of the relationship (e.g., 'STUDIES_AT', 'INTERESTED_IN')")

class DeleteKnowledge(BaseModel):
    """Remove obsolete, incorrect, or duplicate entities or relationships."""
    entity_ids: Optional[List[str]] = Field(None, description="List of entity IDs to remove")
    relationships: Optional[List[DeleteEdge]] = Field(None, description="List of specific relationships to remove")
    document_ids: Optional[List[str]] = Field(None, description="List of Document IDs containing obsolete information to remove")
    reason: str = Field(..., description="Why these elements should be deleted")

class NoOp(BaseModel):
    """Do nothing if the information is already present, irrelevant, or redundant."""
    reason: str = Field(..., description="Why no action is taken")

# --- Core Logic ---

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: str = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
        enum=["graph_vector", "general_llm"]
    )

class GraphRAGSystem:
    def __init__(self):
        setup_environment()
        
        self.console = Console()
        
        # 1. Initialize LLM (Groq)
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # 2. Local Embeddings
        self.console.print("[dim]Loading embeddings model...[/]")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 3. Neo4j Graph
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        # Fetch Schema
        try:
            self.graph.refresh_schema()
            self.schema = self.graph.schema
            self.console.print(f"[dim]Graph schema loaded: {self.schema[:100]}...[/]")
        except Exception as e:
            self.console.print(f"[yellow]Could not fetch schema: {e}[/]")
            self.schema = ""

        # 4. Initialize Indices (Fix for Missing Index Error)
        self._init_indices()

        # 5. Vector Store
        self.vector_index = None

    def _init_indices(self):
        """Create necessary indices and ensure data compatibility."""
        self.console.print("[dim]Checking graph indices...[/]")
        try:
            # 1. Create Layout for Entities (using a common label '__Entity__')
            # This index is required by _structured_retriever
            self.graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (n:__Entity__) ON EACH [n.id]")
            
            # 2. Backfill: Ensure existing non-document nodes have the __Entity__ label
            # This ensures they are indexed by the fulltext index we just created/verified
            self.graph.query("MATCH (n) WHERE n.id IS NOT NULL AND NOT 'Document' IN labels(n) SET n:__Entity__")
            
            self.console.print("[dim]Ensured 'entity' fulltext index exists and applied labels.[/]")
        except Exception as e:
            self.console.print(f"[red]Index initialization warning: {e}[/]")

    def _retrieve_documents_with_ids(self, query: str) -> List[dict]:
        """Search for existing documents and return their content AND IDs for deletion."""
        if not self.vector_index: return []
        try:
            docs = self.vector_index.similarity_search(query, k=3)
            results = []
            for doc in docs:
                text_content = doc.page_content
                # Fetch ID for this text from Graph
                res = self.graph.query("MATCH (d:Document {text: $text}) RETURN d.id as id LIMIT 1", {"text": text_content})
                if res:
                   results.append({"id": res[0]['id'], "text": text_content[:200] + "..."})
            return results
        except Exception:
            return []

    def _get_document_text(self, doc_id: str) -> str:
        """Retrieve text content of a specific document by ID."""
        try:
            query = "MATCH (d:Document) WHERE d.id = $id RETURN d.text as text"
            res = self.graph.query(query, {"id": doc_id})
            return res[0]['text'] if res else ""
        except Exception:
            return ""

    def ingest_text(self, text: str):
        """Intelligent Ingestion: Chunk -> Retrieve Context -> Tool Call (Add/Upd/Del)"""
        
        from langchain_text_splitters import TokenTextSplitter
        
        # 1. Split Text
        self.console.print("[dim]Splitting text...[/]")
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        
        # 2. Bind Tools
        llm_with_tools = self.llm.bind_tools([AddKnowledge, UpdateKnowledge, DeleteKnowledge, NoOp])
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            
            task = progress.add_task("Processing chunks...", total=len(chunks))
            
            for i, chunk in enumerate(chunks):
                progress.update(task, description=f"Processing chunk {i+1}/{len(chunks)}...")
                
                # A. Retrieve Existing Context
                existing_graph_context = self._structured_retriever(chunk)
                existing_docs_with_ids = self._retrieve_documents_with_ids(chunk)
                
                existing_docs_str = "\n".join([f"[DocID: {d['id']}] {d['text']}" for d in existing_docs_with_ids])
                
                if not existing_graph_context: existing_graph_context = "No related graph knowledge."
                if not existing_docs_str: existing_docs_str = "No related documents."
                
                # B. Decide Action
                safe_schema = self.schema.replace("{", "{{").replace("}", "}}") if self.schema else ""
                
                system_prompt = f"""You are an Expert Knowledge Graph Architect.
                
                Current Graph Schema: {safe_schema}
                
                Existing Graph Knowledge (to prevent duplicates):
                {existing_graph_context}
                
                Existing Documents (Unstructured):
                {existing_docs_str}
                
                Task: content extraction and graph modeling from 'New Input'.
                
                **Strict Modeling Rules:**
                1. **Entity Resolution (CRITICAL)**:
                   - **Standardization**: ALWAYS generalize to Title Case for IDs (e.g. use "John" not "john", "Python" not "python").
                   - **De-duplication**: Check 'Existing Graph Knowledge'. If an entity exists (e.g. "John Doe"), and input says "John", use "John Doe" as ID.
                
                2. **Graph Topology & Schema**:
                   - **Node Labels**: Use singular, PascalCase (e.g., 'Person', 'Organization', 'Skill'). Reuse Schema labels if applicable.
                   - **Relationship Types**: Use SCREAMING_SNAKE_CASE (e.g., 'WORKS_AT', 'HAS_SKILL').
                   - **Direction**: (Source)-[RELATIONSHIP]->(Target). Ensure logical direction.
                
                3. **Data Integrity & Updates (Context Preservation)**:
                   - **Merge Logic**: If 'New Input' updates an existing fact (e.g. "Meet moved to Bangalore" but graph has "Lived in Mumbai"):
                     - **Graph**: Update the node/edge to the *current* state.
                     - **Documents**: You MUST use `DeleteKnowledge` to delete the *outdated document* (by ID). This triggers a merge process where you will be able to preserve history.
                   - **New Data**: Use `AddKnowledge` for new nodes/edges.
                   - **Property Updates**: If a node exists, use `AddKnowledge` (it merges properties) or `UpdateKnowledge`.
                
                4. **Conservative Deletion Rules (CRITICAL)**:
                   - **Relationships**: If a relationship ends (e.g. "Meet stopped learning Neo4j"), DELETE ONLY the relationship edge. DO NOT DELETE THE ENTITIES (Meet, Neo4j).
                   - **Garbage Collection**: If an entity (e.g. 'Neo4j') becomes isolated (no other connections) after you delete the relationship, the system's background process WILL AUTOMATICALLY DELETE IT. You do not need to do this explicitly. Only delete the edge.
                   - **Entities**: Delete an entity (node) ONLY if it is explicitly stated to be destroyed, non-existent, or an error.

                5. **Properties**:
                   - Extract attributes (e.g., 'age', 'start_date') into Node/Edge properties.
                
                Refine the graph structure to be clean and connected.
                """
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "New Input: {input}"),
                ])
                
                chain = prompt | llm_with_tools
                
                # 1. Invoke
                try:
                    response = chain.invoke({"input": chunk})
                except Exception as e:
                    self.console.print(f"[red]Error processing chunk {i}: {e}[/]")
                    continue

                # 2. Determine Final Text Payload (Merge Logic)
                text_payload = chunk
                doc_ids_to_del = []
                
                # Pre-scan for DeleteKnowledge to handle Document Merging
                if response.tool_calls:
                    for tool_call in response.tool_calls:
                        if tool_call['name'] == "DeleteKnowledge":
                            doc_ids_to_del.extend(tool_call['args'].get('document_ids') or [])
                
                if doc_ids_to_del:
                    self.console.print(f"[cyan]Detected document deletion. Merging context...[/]")
                    old_texts = [self._get_document_text(did) for did in doc_ids_to_del]
                    full_old_text = "\n".join([t for t in old_texts if t])
                    
                    if full_old_text:
                        merge_template = """You are updating a document. Merge the 'Old Content' with the 'New Update'.
                        
                        **Objective**: Create a single cohesive text that reflects the *current* state while explicitly preserving *past context*.
                        
                        **Instructions**:
                        1. **Current State**: The 'New Update' is the source of truth for the current state.
                        2. **History**: Do NOT simply delete conflicting 'Old Content'. instead, rephrase it as past context (e.g. "Meet now lives in Bangalore (previously lived in Mumbai).").
                        3. **Mentions**: Explicitly mention "previous" or "formerly" for outdated facts.
                        4. **Redundancy**: Remove exact duplicates.
                        5. Return ONLY the final merged text.
                        
                        Old Content:
                        {old_text}
                        
                        New Update:
                        {new_update}"""
                        
                        merge_prompt = ChatPromptTemplate.from_template(merge_template)
                        merger = merge_prompt | self.llm | StrOutputParser()
                        try:
                            text_payload = merger.invoke({"old_text": full_old_text, "new_update": chunk})
                            self.console.print(f"[cyan]Merged {len(doc_ids_to_del)} documents into new version.[/]")
                        except Exception as e:
                            self.console.print(f"[red]Merge failed: {e}. Using new chunk only.[/]")

                # 3. Execute Tools
                try:
                    if response.tool_calls:
                        for tool_call in response.tool_calls:
                            args = tool_call['args']
                            name = tool_call['name']
                            
                            if name == "AddKnowledge":
                                self._handle_add(args)
                            elif name == "UpdateKnowledge":
                                self._handle_update(args)
                            elif name == "DeleteKnowledge":
                                self._handle_delete(args)
                            elif name == "NoOp":
                                self.console.print(f"[dim]NoOp: {args.get('reason')}[/]")
                except Exception as e:
                    self.console.print(f"[red]Error executing tools: {e}[/]")

                # 4. Cleanup Orphans
                self._prune_orphaned_nodes()

                # 5. Add Document Node (New or Merged)
                # We add this AFTER deleting the old ones (via tools), effectively replacing them.
                self._add_document_node(text_payload)
            
            # Refresh schema at the end
            self.graph.refresh_schema()
            self.schema = self.graph.schema
            
        self.console.print(f"[bold green]Ingestion Complete![/] Processed {len(chunks)} chunks.")

    def _handle_add(self, args):
        """Execute AddKnowledge"""
        nodes = args.get('nodes', [])
        edges = args.get('edges', [])
        
        timestamp = datetime.now().isoformat()
        
        for node in nodes:
            # Create/Merge Node
            props = node.get('properties', {})
            props['id'] = node['id'] 
            # FIX: Add __Entity__ label so it's picked up by the fulltext index
            query = f"MERGE (n:`{node['type']}` {{id: $id}}) SET n:__Entity__, n += $props, n.last_updated = $timestamp"
            self.graph.query(query, {"id": node['id'], "props": props, "timestamp": timestamp})
            
        for edge in edges:
            query = f"""
            MATCH (s {{id: $source}}), (t {{id: $target}})
            MERGE (s)-[r:`{edge['type']}`]->(t)
            SET r.last_updated = $timestamp
            """
            self.graph.query(query, {"source": edge['source'], "target": edge['target'], "timestamp": timestamp})
        
        self.console.print(f"[green]Added {len(nodes)} nodes, {len(edges)} edges.[/]")

    def _handle_update(self, args):
        """Execute UpdateKnowledge"""
        eid = args['entity_id']
        props = args['properties']
        timestamp = datetime.now().isoformat()
        
        query = "MATCH (n) WHERE n.id = $id SET n += $props, n.last_updated = $timestamp RETURN n"
        self.graph.query(query, {"id": eid, "props": props, "timestamp": timestamp})
        self.console.print(f"[yellow]Updated entity {eid}: {props}[/]")

    def _handle_delete(self, args):
        """Execute DeleteKnowledge"""
        eids = args.get('entity_ids') or []
        rels = args.get('relationships') or []
        doc_ids = args.get('document_ids') or []
        
        if eids:
            query = "MATCH (n) WHERE n.id IN $ids DETACH DELETE n"
            self.graph.query(query, {"ids": eids})
            self.console.print(f"[red]Deleted entities: {eids}[/]")
        
        if doc_ids:
            query = "MATCH (d:Document) WHERE d.id IN $ids DETACH DELETE d"
            self.graph.query(query, {"ids": doc_ids})
            self.console.print(f"[red]Deleted documents: {doc_ids}[/]")
            
        if rels:
            for rel in rels:
                source = rel['source']
                target = rel.get('target')
                rel_type = rel['type']
                
                if target:
                    # Specific deletion
                    query = f"""
                    MATCH (s {{id: $source}})-[r:`{rel_type}`]-(t {{id: $target}})
                    DELETE r
                    """
                    self.graph.query(query, {"source": source, "target": target})
                    self.console.print(f"[red]Deleted relationship: ({source})-[{rel_type}]->({target})[/]")
                else:
                    # Wildcard deletion (by type from source)
                    query = f"""
                    MATCH (s {{id: $source}})-[r:`{rel_type}`]-()
                    DELETE r
                    """
                    self.graph.query(query, {"source": source})
                    self.console.print(f"[red]Deleted ALL '{rel_type}' relationships from ({source})[/]")

                    self.console.print(f"[red]Deleted ALL '{rel_type}' relationships from ({source})[/]")

    def _prune_orphaned_nodes(self):
        """Remove entities that have no relationships (orphans)."""
        # We only delete nodes labeled __Entity__ that have no relationships.
        query = """
        MATCH (n:__Entity__)
        WHERE NOT (n)--()
        DELETE n
        RETURN count(n) as deleted_count
        """
        try:
            res = self.graph.query(query)
            if res and res[0]['deleted_count'] > 0:
                self.console.print(f"[dim red]Pruned {res[0]['deleted_count']} orphaned nodes.[/]")
        except Exception as e:
            self.console.print(f"[dim red]Error pruning orphans: {e}[/]")

    def _add_document_node(self, text):
        """Add a Document node for vector indexing compatibility"""
        doc_id = str(hash(text))
        timestamp = datetime.now().isoformat()
        query = "MERGE (d:Document {id: $id}) SET d.text = $text, d.created_at = $timestamp"
        self.graph.query(query, {"id": doc_id, "text": text, "timestamp": timestamp})
        
        # Ensure vector index exists
        if not self.vector_index:
             try:
                 self.vector_index = Neo4jVector.from_existing_graph(
                    self.embeddings,
                    search_type="hybrid",
                    node_label="Document",
                    text_node_properties=["text"],
                    embedding_node_property="embedding",
                    url=os.getenv("NEO4J_URI"),
                    username=os.getenv("NEO4J_USERNAME"),
                    password=os.getenv("NEO4J_PASSWORD")
                )
             except: pass 
    
    def _generate_full_text_query(self, input_str: str) -> str:
        """
        Generate a full-text search query for a given input string.
        Allowing for some fuzziness (~2).
        """
        words = [el for el in input_str.split() if el]
        if not words:
            return ""
        full_text_query = ""
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    def _structured_retriever(self, question: str) -> str:
        """
        Graph Retrieval:
        1. Extract entities from question.
        2. Search entities in Neo4j (Fulltext).
        3. Get 1-hop neighborhood.
        """
        # 1. Extract Entities
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are extracting organization, person, location, and other key entities from the text."),
            ("human", "Use the given format to extract information from the following input: {question}"),
        ])
        
        entity_chain = prompt | self.llm.with_structured_output(Entities)
        
        try:
            entities = entity_chain.invoke({"question": question})
        except Exception as e:
            self.console.print(f"[yellow]Entity extraction failed: {e}. Falling back to keyword search.[/]")
            return ""

        if not entities or not entities.names:
            return ""

        # 2. Query Neighbors
        result_text = ""
        for entity in entities.names:
            query_str = self._generate_full_text_query(entity)
            if not query_str:
                continue
            
            # FIX: Updated CALL syntax to avoid deprecation warning
            # Also returning 'last_updated' or other properties from relationship
            response = self.graph.query(
                """
                CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node, score
                CALL () { 
                  MATCH (node)-[r:!MENTIONS]->(neighbor)
                  RETURN node.id + ' - ' + type(r) + ' (updated: ' + coalesce(r.last_updated, 'n/a') + ') -> ' + neighbor.id AS output
                  UNION
                  MATCH (node)<-[r:!MENTIONS]-(neighbor)
                  RETURN neighbor.id + ' - ' + type(r) + ' (updated: ' + coalesce(r.last_updated, 'n/a') + ') -> ' + node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": query_str}
            )
            result_text += "\n".join([el['output'] for el in response])
            
        return result_text

    def _retriever(self, question: str) -> str:
        """
        Hybrid Retriever: Combines Structured (Graph) + Unstructured (Vector)
        """
        self.console.print(f"[dim]Retrieving specific context for: {question}...[/]")
        
        # Structured
        structured_data = self._structured_retriever(question)
        
        # Unstructured (Vector Search)
        # We need to ensure vector_index is initialized.
        if not self.vector_index:
             # Try to initialize if it exists in DB
             try:
                 self.vector_index = Neo4jVector.from_existing_graph(
                    self.embeddings,
                    search_type="hybrid",
                    node_label="Document",
                    text_node_properties=["text"],
                    embedding_node_property="embedding",
                    url=os.getenv("NEO4J_URI"),
                    username=os.getenv("NEO4J_USERNAME"),
                    password=os.getenv("NEO4J_PASSWORD")
                )
             except Exception:
                 return "Vector index not found. Please ingest data first."
        
        unstructured_docs = self.vector_index.similarity_search(question)
        unstructured_data = [el.page_content for el in unstructured_docs]
        
        final_context = f"""Structured Context (Graph):\n{structured_data}\n\nUnstructured Context (Text):\n{"#Document ".join(unstructured_data)}"""
        return final_context

    def query_rag(self, question: str):
        """Full RAG Pipeline with Intelligent Routing"""
        
        # 1. Routing Step
        if getattr(self, "schema", None):
            # FIX: Escape braces so LangChain doesn't treat them as variables
            safe_schema = self.schema.replace("{", "{{").replace("}", "}}")
            schema_context = f"The User's Knowledge Base Schema: {safe_schema}" 
        else:
             # Fallback if schema failed to load
            schema_context = "The User's Knowledge Base contains custom entities and relationships from their documents."

        system_prompt = f"""You are a query router. You must decide whether to route a user question to a 'graph_vector' (Graph Knowledge Base) or 'general_llm' (General Chat).

{schema_context}

Routing Rules:
1. **Graph Knowledge Base ('graph_vector')**: 
   - Choose this if the question is about specific people, projects, concepts, or entities that assume local/custom knowledge (e.g. "What is Meet interested in?", "Who worked on Project X?", "What is the relationship between A and B?").
   - If the question mentions Proper Nouns (names of people, places, things) that look like they belong in the schema (e.g. a Person node, an Organization node), ALWAYS choose 'graph_vector'.
   - If the question asks about "interests", "skills", "relationships" corresponding to the schema details.

2. **General Chat ('general_llm')**: 
   - Choose this ONLY for generic small talk (e.g. "Hi", "How are you?") or broad general knowledge questions unrelated to the specific entities in the schema (e.g. "What is the capital of France?", "Explain quantum physics").

**CRITICAL**: If you are unsure, PREFER 'graph_vector' to check the database first.
"""
        
        route_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])
        
        router = route_prompt | self.llm.with_structured_output(RouteQuery)
        
        self.console.print(f"[dim]Schema available: {bool(self.schema)}[/]")
        self.console.print("[dim]Deciding route...[/]")
        
        try:
            route_decision = router.invoke({"question": question})
            decision = route_decision.datasource
        except Exception as e:
            self.console.print(f"[dim]Routing failed: {e}. Defaulting to Graph...[/]")
            decision = "graph_vector"

        if decision == "general_llm":
            self.console.print(f"[bold blue]Routing to General LLM[/] (Reason: Question deemed generic)")
            template = """Answer the question based on your general knowledge.
            Question: {question}
            Helpful Answer:"""
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | StrOutputParser()
            return chain.invoke({"question": question})
        
        else:
            self.console.print(f"[bold green]Routing to Graph Knowledge Base[/] (Reason: relevant to Schema)")
            
            # --- Retrieval ---
            context = self._retriever(question)
            
            # --- Generation ---
            template = """Answer the question based only on the following context.
            Pay attention to the 'updated' timestamps in the context if available.
            - If there is conflicting information, prioritize the information with the MOST RECENT updated timestamp check for the most recent update. it should be very accurate.
            - If the user explicitly asks for "previous" or "history", mention the older information.
            - Otherwise, provide the current/latest fact.
            
            Context:
            {context}
            
            Question: {question}
            
            Helpful Answer:"""
            
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | StrOutputParser()
            # FIX: Pass 'context' to the chain!
            return chain.invoke({"context": context, "question": question})

# --- Main CLI ---

def main():
    console.print(Panel(
        "[bold magenta]GraphRAG System[/]\n"
        "Powered by [bold cyan]LangChain[/], [bold orange3]Groq[/], and [bold blue]Neo4j[/]",
        subtitle="Hybrid Retrieval (Vector + Graph)",
        border_style="magenta"
    ))
    
    rag = GraphRAGSystem()
    
    while True:
        action = Prompt.ask(
            "\nWhat would you like to do?",
            choices=["ingest", "query", "exit"],
            default="query"
        )
        
        if action == "exit":
            break
            
        if action == "ingest":
            choice = Prompt.ask("Input method", choices=["file", "paste"], default="file")
            text = ""
            if choice == "file":
                path = Prompt.ask("File path")
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                else:
                    console.print("[red]File not found[/]")
            else:
                console.print("[yellow]Paste text (double enter to finish):[/]")
                lines = []
                while True:
                    line = input()
                    if not line: break
                    lines.append(line)
                text = "\n".join(lines)
            
            if text:
                rag.ingest_text(text)
                
        elif action == "query":
            q = Prompt.ask("\n[bold cyan]Ask a question[/]")
            if q:
                response = rag.query_rag(q)
                console.print(Panel(response, title="Answer", border_style="green"))

if __name__ == "__main__":
    main()
