"""
CORTEX - Personal AI with Persistent Memory
============================================
Clean Streamlit application using direct Neo4j queries.
"""

import streamlit as st
from cortex_memory import CortexMemory
from rich.console import Console

console = Console()

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="CORTEX",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* Dark theme enhancements */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 1rem 1rem 0.25rem 1rem;
        margin: 0.5rem 0;
    }
    
    .assistant-message {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 1rem 1rem 1rem 0.25rem;
        margin: 0.5rem 0;
    }
    
    /* Processing logs */
    .log-container {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 0.5rem;
        padding: 0.75rem;
        font-family: monospace;
        font-size: 0.8rem;
        color: #888;
    }
    
    /* Action badges */
    .action-ingest { color: #4ade80; }
    .action-update { color: #fbbf24; }
    .action-delete { color: #f87171; }
    .action-noop { color: #94a3b8; }
    
    /* Sidebar */
    .sidebar-stats {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 0.5rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'memory' not in st.session_state:
    try:
        st.session_state.memory = CortexMemory()
        st.session_state.initialized = True
    except Exception as e:
        st.session_state.memory = None
        st.session_state.init_error = str(e)
        st.session_state.initialized = False

if 'user_email' not in st.session_state:
    st.session_state.user_email = None

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'logs' not in st.session_state:
    st.session_state.logs = []

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 🧠 CORTEX")
    st.caption("Personal AI with Persistent Memory")
    
    st.divider()
    
    # User login
    if not st.session_state.user_email:
        st.markdown("### 👤 Identity")
        email = st.text_input(
            "Email Address",
            placeholder="you@example.com",
            help="Your unique identifier for memory"
        )
        
        if st.button("🔓 Start Session", use_container_width=True):
            if email and "@" in email:
                st.session_state.user_email = email
                if st.session_state.memory:
                    st.session_state.memory.set_user_session(email)
                st.rerun()
            else:
                st.error("Please enter a valid email")
    else:
        st.success(f"**User:** {st.session_state.user_email}")
        
        if st.button("🚪 End Session", use_container_width=True):
            st.session_state.user_email = None
            st.session_state.messages = []
            st.session_state.logs = []
            st.rerun()
        
        st.divider()
        
        # Stats
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state.messages) // 2)
        with col2:
            st.metric("Operations", len([l for l in st.session_state.logs if "Created" in l or "Updated" in l]))
        
        # Schema info
        if st.session_state.memory:
            with st.expander("📋 Graph Schema"):
                schema = st.session_state.memory.get_schema()
                st.code(schema, language="text")
        
        st.divider()
        
        # Clear options
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.logs = []
            if st.session_state.memory:
                st.session_state.memory.chat_history = []
            st.rerun()

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown('<h1 class="main-header">🧠 CORTEX</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #888;">Personal AI with Persistent Memory</p>', unsafe_allow_html=True)

# Check initialization
if not st.session_state.initialized:
    st.error(f"Failed to initialize: {st.session_state.get('init_error', 'Unknown error')}")
    st.info("Check your .env file for NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, and GROQ_API_KEY")
    st.stop()

# Require login
if not st.session_state.user_email:
    st.info("👈 Please enter your email in the sidebar to start")
    
    # Demo info
    with st.expander("ℹ️ How CORTEX Works"):
        st.markdown("""
        **CORTEX** is an AI assistant with persistent memory stored in a knowledge graph.
        
        **What it remembers:**
        - Personal facts (name, birthday, location)
        - Preferences and interests
        - Relationships and connections
        - Skills and work history
        
        **How it works:**
        1. 🔍 Extracts keywords from your message
        2. 📊 Retrieves relevant context from the graph
        3. 💬 Generates a context-aware response
        4. 📝 Stores new facts with embeddings
        
        **Try saying:**
        - "Hi, I'm John and I work at TechCorp"
        - "My birthday is on March 15th"
        - "I live in San Francisco"
        - "What do you know about me?"
        """)
    st.stop()

# Chat display
chat_container = st.container()

with chat_container:
    for i, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        content = msg["content"]
        logs = msg.get("logs", [])
        
        with st.chat_message(role):
            st.markdown(content)
            
            # Show processing logs for assistant messages
            if role == "assistant" and logs:
                with st.expander("⚙️ Processing", expanded=False):
                    for log in logs:
                        # Color-code by type
                        if "Created" in log or "Linked" in log:
                            st.markdown(f"🟢 {log}")
                        elif "Updated" in log:
                            st.markdown(f"🟠 {log}")
                        elif "deleted" in log.lower():
                            st.markdown(f"🔴 {log}")
                        elif "Action:" in log:
                            action = log.split("Action:")[1].split()[0].strip()
                            icon = {"INGEST": "🟢", "UPDATE": "🟠", "DELETE": "🔴"}.get(action, "⚪")
                            st.markdown(f"{icon} **{log}**")
                        else:
                            st.caption(log)

# Chat input
if prompt := st.chat_input("Message CORTEX..."):
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process and respond
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response, logs = st.session_state.memory.process(prompt)
                
                st.markdown(response)
                
                # Show logs
                if logs:
                    with st.expander("⚙️ Processing", expanded=False):
                        for log in logs:
                            if "Created" in log or "Linked" in log:
                                st.markdown(f"🟢 {log}")
                            elif "Updated" in log:
                                st.markdown(f"🟠 {log}")
                            elif "deleted" in log.lower():
                                st.markdown(f"🔴 {log}")
                            elif "Action:" in log:
                                action = log.split("Action:")[1].split()[0].strip()
                                icon = {"INGEST": "🟢", "UPDATE": "🟠", "DELETE": "🔴"}.get(action, "⚪")
                                st.markdown(f"{icon} **{log}**")
                            else:
                                st.caption(log)
                
                # Store assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "logs": logs
                })
                st.session_state.logs.extend(logs)
                
            except Exception as e:
                st.error(f"Error: {e}")
                console.print_exception()
