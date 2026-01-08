"""
Streamlit App with LangGraph Workflow Integration
Streaming Chatbot - Following app_v1.py structure
"""

import streamlit as st
import json
import uuid
import sqlite3
import sys
import os
import time
from pathlib import Path

# Add parent directory to path to import from langgraph folder
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from langgraph_custom.langgraph_workflow import build_workflow, chat_node_stream
from dotenv import load_dotenv
import re

load_dotenv()

# Cache the workflow to avoid rebuilding on every page load
@st.cache_resource
def get_workflow():
    """Build and cache the LangGraph workflow for reuse across sessions"""
    return build_workflow()

# Cache database queries to reduce I/O on page loads
@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_cached_conversations():
    """Get list of conversation IDs from database with caching"""
    return get_all_conversations()

# Database functions (same as app_v1)
DB_FILE = str(parent_dir / "data" / "chat_history.db")

def get_db_connection():
    # Ensure the database directory exists
    db_dir = os.path.dirname(DB_FILE)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Create base table first
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        )
    ''')
    
    # Check if new columns exist and add them if they don't
    c.execute("PRAGMA table_info(chat_messages)")
    columns = [row[1] for row in c.fetchall()]
    
    if 'message_type' not in columns:
        c.execute('ALTER TABLE chat_messages ADD COLUMN message_type TEXT DEFAULT "text"')
    
    if 'metadata' not in columns:
        c.execute('ALTER TABLE chat_messages ADD COLUMN metadata TEXT DEFAULT NULL')
        
    if 'timestamp' not in columns:
        # Add timestamp column without default first
        c.execute('ALTER TABLE chat_messages ADD COLUMN timestamp TIMESTAMP')
        # Then update existing rows to have a timestamp
        c.execute('UPDATE chat_messages SET timestamp = datetime("now") WHERE timestamp IS NULL')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS extraction_states (
            conversation_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT (datetime('now'))
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            stored_filepath TEXT NOT NULL,
            file_size INTEGER,
            upload_timestamp TIMESTAMP DEFAULT (datetime('now')),
            FOREIGN KEY (conversation_id) REFERENCES extraction_states (conversation_id)
        )
    ''')
    conn.commit()
    return conn

def save_message_to_db(conversation_id, role, content, message_type="text", metadata=None):
    import json
    conn = get_db_connection()
    c = conn.cursor()
    metadata_json = json.dumps(metadata) if metadata else None
    # Use datetime("now") for SQLite timestamp
    c.execute("""INSERT INTO chat_messages 
                 (conversation_id, role, content, message_type, metadata, timestamp) 
                 VALUES (?, ?, ?, ?, ?, datetime('now'))""", 
              (conversation_id, role, content, message_type, metadata_json))
    conn.commit()
    conn.close()

def load_messages_from_db(conversation_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT role, content FROM chat_messages WHERE conversation_id = ? ORDER BY id", 
              (conversation_id,))
    messages = [{"role": row[0], "content": row[1]} for row in c.fetchall()]
    conn.close()
    return messages

def save_extraction_state(conversation_id, state):
    """Save extraction state to database"""
    import json
    conn = get_db_connection()
    c = conn.cursor()
    # Only save serializable state data (excluding full_text for storage efficiency)
    state_to_save = {
        "parsed_json": state.get("parsed_json", {}),
        "parser_only_json": state.get("parser_only_json", {}),
        "data_to_summarize": state.get("data_to_summarize", {}),
        "confidence_score": state.get("confidence_score", 0.0),
        "completeness_score": state.get("completeness_score", 0.0),
        "missing_fields": state.get("missing_fields", []),
        "nct_id": state.get("nct_id", ""),
        "input_type": state.get("input_type", "unknown"),
        "input_url": state.get("input_url", ""),  # Save for re-extraction attempts
        "used_llm_fallback": state.get("used_llm_fallback", False),
        "extraction_cost_estimate": state.get("extraction_cost_estimate", {}),
    }
    state_json = json.dumps(state_to_save)
    c.execute("""INSERT OR REPLACE INTO extraction_states 
                 (conversation_id, state_json, updated_at) 
                 VALUES (?, ?, datetime('now'))""", 
              (conversation_id, state_json))
    conn.commit()
    conn.close()

def save_uploaded_file(conversation_id, original_filename, stored_filepath, file_size):
    """Save uploaded file information to database"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO uploaded_files (conversation_id, original_filename, stored_filepath, file_size) VALUES (?, ?, ?, ?)", 
              (conversation_id, original_filename, stored_filepath, file_size))
    conn.commit()
    conn.close()

def get_uploaded_file_path(conversation_id):
    """Get stored file path for a conversation"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT stored_filepath, original_filename FROM uploaded_files WHERE conversation_id = ? ORDER BY upload_timestamp DESC LIMIT 1", 
              (conversation_id,))
    row = c.fetchone()
    conn.close()
    return row if row else None

def load_extraction_state(conversation_id):
    """Load extraction state from database"""
    import json
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT state_json FROM extraction_states WHERE conversation_id = ?", 
              (conversation_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None

def get_all_conversations():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT DISTINCT conversation_id FROM chat_messages ORDER BY id DESC")
    conversations = [row[0] for row in c.fetchall()]
    conn.close()
    return conversations

def get_database_statistics():
    """Get statistics about the database including record counts and sizes"""
    conn = get_db_connection()
    c = conn.cursor()
    
    stats = {}
    
    try:
        # Count total messages
        c.execute("SELECT COUNT(*) FROM chat_messages")
        stats["total_messages"] = c.fetchone()[0]
        
        # Count conversations
        c.execute("SELECT COUNT(DISTINCT conversation_id) FROM chat_messages")
        stats["total_conversations"] = c.fetchone()[0]
        
        # Count extraction states
        c.execute("SELECT COUNT(*) FROM extraction_states")
        stats["total_extractions"] = c.fetchone()[0]
        
        # Count uploaded files
        c.execute("SELECT COUNT(*) FROM uploaded_files")
        stats["total_files_uploaded"] = c.fetchone()[0]
        
        # Get total file size (in MB)
        c.execute("SELECT COALESCE(SUM(file_size), 0) FROM uploaded_files")
        total_bytes = c.fetchone()[0]
        stats["total_file_size_mb"] = round(total_bytes / (1024 * 1024), 2)
        
        # Count messages by role
        c.execute("SELECT role, COUNT(*) FROM chat_messages GROUP BY role")
        role_counts = c.fetchall()
        stats["messages_by_role"] = {role: count for role, count in role_counts}
        
        # Count message types
        c.execute("SELECT message_type, COUNT(*) FROM chat_messages WHERE message_type IS NOT NULL GROUP BY message_type")
        type_counts = c.fetchall()
        stats["messages_by_type"] = {msg_type: count for msg_type, count in type_counts}
        
        # Get database file size
        if os.path.exists(DB_FILE):
            db_size_bytes = os.path.getsize(DB_FILE)
            stats["database_file_size_mb"] = round(db_size_bytes / (1024 * 1024), 2)
        else:
            stats["database_file_size_mb"] = 0
        
        # Get average messages per conversation
        if stats["total_conversations"] > 0:
            c.execute("SELECT AVG(msg_count) FROM (SELECT COUNT(*) as msg_count FROM chat_messages GROUP BY conversation_id)")
            avg = c.fetchone()[0]
            stats["avg_messages_per_conversation"] = round(avg, 1) if avg else 0
        else:
            stats["avg_messages_per_conversation"] = 0
        
        # Get extraction success rate
        c.execute("SELECT COUNT(*) FROM extraction_states WHERE state_json LIKE '%\"nct_id\"%'")
        successful_extractions = c.fetchone()[0]
        if stats["total_extractions"] > 0:
            stats["extraction_success_rate"] = round((successful_extractions / stats["total_extractions"]) * 100, 1)
        else:
            stats["extraction_success_rate"] = 0
        
        # Count unique NCT IDs
        c.execute("SELECT COUNT(DISTINCT JSON_EXTRACT(state_json, '$.nct_id')) FROM extraction_states WHERE JSON_EXTRACT(state_json, '$.nct_id') IS NOT NULL AND JSON_EXTRACT(state_json, '$.nct_id') != ''")
        result = c.fetchone()
        stats["unique_nct_ids"] = result[0] if result and result[0] else 0
            
    except Exception as e:
        st.error(f"Error retrieving database statistics: {str(e)}")
    finally:
        conn.close()
    
    return stats

def display_database_statistics():
    """Display database statistics in the sidebar"""
    with st.sidebar:
        if st.button("📊 Database Statistics", use_container_width=True):
            stats = get_database_statistics()
            
            with st.expander("📊 Database Statistics", expanded=True):
                # Main statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("💬 Total Conversations", stats.get("total_conversations", 0))
                    st.metric("📄 Total Messages", stats.get("total_messages", 0))
                with col2:
                    st.metric("🔍 Total Extractions", stats.get("total_extractions", 0))
                    st.metric("📁 Files Uploaded", stats.get("total_files_uploaded", 0))
                
                # Unique NCT IDs
                st.metric("🔬 Unique Clinical Trials (NCT IDs)", stats.get("unique_nct_ids", 0))
                
                st.divider()
                
                # File and database sizes
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("💾 Uploaded Files Size", f"{stats.get('total_file_size_mb', 0)} MB")
                with col2:
                    st.metric("📦 Database Size", f"{stats.get('database_file_size_mb', 0)} MB")
                
                st.divider()
                
                # Average metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📊 Avg Messages/Conversation", stats.get("avg_messages_per_conversation", 0))
                with col2:
                    st.metric("✅ Extraction Success Rate", f"{stats.get('extraction_success_rate', 0)}%")
                
                st.divider()
                
                # Messages by role breakdown
                st.subheader("💬 Messages by Role")
                role_data = stats.get("messages_by_role", {})
                if role_data:
                    for role, count in role_data.items():
                        st.write(f"• **{role.capitalize()}**: {count} messages")
                else:
                    st.write("No message data available")
                
                # Messages by type breakdown
                st.subheader("🏷️ Messages by Type")
                type_data = stats.get("messages_by_type", {})
                if type_data:
                    for msg_type, count in type_data.items():
                        st.write(f"• **{msg_type}**: {count} messages")
                else:
                    st.write("No type data available")

def new_chat_click():
    """Callback for starting a new chat - sets flag for main flow to handle rerun"""
    st.session_state.messages = []
    st.session_state.current_convo_id = str(uuid.uuid4())
    st.session_state.url_key = str(uuid.uuid4())
    st.session_state.current_state = None
    st.session_state.cached_full_text = None
    st.session_state.approved_fields = {}
    st.session_state.refinement_requests = {}
    # Set flag to trigger rerun in main flow (st.rerun() doesn't work in callbacks)
    st.session_state.new_chat_requested = True

def create_extraction_results_tabs(state):
    """Create tabbed extraction results with Metrics and View JSON tabs"""
    confidence = state.get("confidence_score", 0)
    completeness = state.get("completeness_score", 0)
    missing_fields = state.get("missing_fields", [])
    nct_id = state.get("nct_id", "Unknown")
    input_type = state.get("input_type", "unknown")
    parsed_json = state.get("parsed_json", {})
    used_llm_fallback = state.get("used_llm_fallback", False)
    parser_json = state.get("parser_only_json", {})
    
    # Count filled fields
    total_fields = 9
    filled_fields = total_fields - len(missing_fields)
    
    # Initialize active tab in session state if not exists
    if "active_extraction_tab" not in st.session_state.ui_state:
        st.session_state.ui_state["active_extraction_tab"] = 0  # Default to first tab
    
    # Callback function to update tab state
    def update_tab_selection():
        if "extraction_tab_selector" in st.session_state:
            st.session_state.ui_state["active_extraction_tab"] = st.session_state.extraction_tab_selector
    
    # Create tabs with selection state and callback
    tab_names = ["📊 Extraction Metrics", "📄 View JSON"]
    selected_tab = st.radio(
        "Select View", 
        options=list(range(len(tab_names))),
        format_func=lambda x: tab_names[x],
        index=st.session_state.ui_state["active_extraction_tab"],
        horizontal=True,
        key="extraction_tab_selector",
        label_visibility="collapsed",
        on_change=update_tab_selection
    )
    
    # Update session state with current selection (fallback)
    st.session_state.ui_state["active_extraction_tab"] = selected_tab
    
    # Add some spacing
    st.markdown("")
    
    # Render content based on selected tab
    if selected_tab == 0:  # Metrics tab
        # Study Info
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown(f"**Study ID:** {nct_id}")
            st.markdown(f"**Input Type:** {input_type.upper()}")
        with col_b:
            st.markdown(f"**Extraction Method:**")
            st.markdown(f"{'🤖 Parser + LLM (Multi-turn)' if used_llm_fallback else '📄 Parser Only'}")
        
        st.divider()
        
        # Show metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Confidence", 
                f"{confidence:.1%}",
                delta="High" if confidence > 0.7 else "Low"
            )
        with col2:
            st.metric(
                "Completeness", 
                f"{completeness:.1%}",
                delta="Excellent" if completeness >= 0.9 else "Good" if completeness >= 0.6 else "Poor"
            )
        with col3:
            st.metric(
                "Fields Extracted",
                f"{filled_fields}/{total_fields}"
            )
        
        # Show missing fields if any
        if missing_fields:
            st.warning(f"**Missing {len(missing_fields)} fields:**")
            for field in missing_fields[:5]:
                st.write(f"• {field.replace('_', ' ').title()}")
        else:
            st.success("✅ All fields extracted successfully!")
        
        # Show LLM extraction details if used
        if used_llm_fallback:
            cost_est = state.get("extraction_cost_estimate", {})
            if cost_est:
                st.divider()
                st.markdown("**💰 Multi-turn Extraction Details:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cost", f"${cost_est.get('total_cost', 0):.3f}")
                with col2:
                    st.metric("Time", f"{cost_est.get('estimated_time_minutes', 0):.1f} min")
                with col3:
                    st.metric("Tokens", f"{cost_est.get('total_tokens', 0):,}")
    
    elif selected_tab == 1:  # View JSON tab
        # CRITICAL: Always use the most current state from session_state for JSON view
        current_state = st.session_state.get('current_state', state)
        
        # Force refresh check for debugging
        ui_state = st.session_state.get("ui_state", {})
        if ui_state.get("force_json_refresh"):
            st.info("🔄 **Status:** Displaying refreshed JSON data after re-extraction")
        
        create_json_view_tabs(current_state)
        
        # Add regenerate summary button at the bottom
        st.divider()
        st.markdown("### 🔄 Regenerate Summary")
        st.caption("After reviewing and refining the extracted fields, click below to regenerate the summary with updated data")
        
        if st.button("🔄 Regenerate Summary", key="regenerate_summary_btn", use_container_width=True, type="primary"):
            # Use the most current state from session_state if available
            current_state = st.session_state.get('current_state', state)
            _regenerate_summary(current_state)

def create_json_view_tabs(state):
    """Create tabs for viewing JSON data with collapsible sections"""
    parsed_json = state.get("parsed_json", {})
    parser_json = state.get("parser_only_json", {})
    used_llm = state.get("used_llm_fallback", False)
    nct_id = state.get("nct_id", "study")
    
    # Debug info for re-extractions
    ui_state = st.session_state.get("ui_state", {})
    if ui_state.get("force_json_refresh"):
        last_field = ui_state.get("last_reextraction_field", "unknown")
        last_time = ui_state.get("last_reextraction_time", 0)
        st.info(f"🔄 **Debug:** JSON view refreshed after re-extracting '{last_field}' at {time.strftime('%H:%M:%S', time.localtime(last_time))}")
        # Clear the flag after showing the debug info
        st.session_state.ui_state["force_json_refresh"] = False
    
    # Define field metadata
    field_info = {
        "study_overview": {"icon": "📋", "title": "Study Overview"},
        "brief_description": {"icon": "📝", "title": "Brief Description"},
        "primary_secondary_objectives": {"icon": "🎯", "title": "Primary & Secondary Objectives"},
        "treatment_arms_interventions": {"icon": "💊", "title": "Treatment Arms & Interventions"},
        "eligibility_criteria": {"icon": "✅", "title": "Eligibility Criteria"},
        "enrollment_participant_flow": {"icon": "👥", "title": "Enrollment & Participant Flow"},
        "adverse_events_profile": {"icon": "⚠️", "title": "Adverse Events Profile"},
        "study_locations": {"icon": "📍", "title": "Study Locations"},
        "sponsor_information": {"icon": "🏢", "title": "Sponsor Information"}
    }
    
    if used_llm and parser_json:
        # Create tabs for parser vs LLM comparison
        tab1, tab2 = st.tabs(["🤖 Final JSON (LLM Enhanced)", "📄 Parser Output"])
        
        with tab1:
            st.markdown("**LLM-Enhanced Extraction Results**")
            _render_json_sections(parsed_json, field_info, "llm")
            
            # Download button
            json_str = json.dumps(parsed_json, indent=2)
            st.download_button(
                label="📥 Download LLM JSON",
                data=json_str,
                file_name=f"llm_enhanced_{nct_id}.json",
                mime="application/json",
                key="download_llm_json",
                use_container_width=True
            )
        
        with tab2:
            st.markdown("**Initial Parser Extraction (Before LLM)")
            _render_json_sections(parser_json, field_info, "parser")
            
            # Download button
            parser_json_str = json.dumps(parser_json, indent=2)
            st.download_button(
                label="📥 Download Parser JSON",
                data=parser_json_str,
                file_name=f"parser_only_{nct_id}.json",
                mime="application/json",
                key="download_parser_json",
                use_container_width=True
            )
    else:
        # Single tab for parser-only results
        st.markdown("**📊 Extracted Data (JSON Format)**")
        _render_json_sections(parsed_json, field_info, "parsed")
        
        # Download button
        json_str = json.dumps(parsed_json, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"extracted_data_{nct_id}.json",
            mime="application/json",
            key="download_json",
            use_container_width=True
        )

def _render_json_sections(json_data, field_info, prefix):
    """Render collapsible sections for each JSON field with human-in-the-loop review"""
    if not json_data:
        st.info("No data extracted")
        return
    
    for field, value in json_data.items():
        info = field_info.get(field, {"icon": "📄", "title": field.replace('_', ' ').title()})
        icon = info["icon"]
        title = info["title"]
        
        # Handle both old string format and new dict format with page references
        if isinstance(value, dict):
            content = value.get("content", "")
            page_refs = value.get("page_references", [])
        else:
            content = str(value) if value else ""
            page_refs = []
        
        # Ensure content is always a string
        if not isinstance(content, str):
            content = str(content) if content else ""
        
        # Check if field has data
        has_data = content and content.strip() and len(content.strip()) > 10
        status = "✅" if has_data else "❌"
        
        # Create expander with enhanced page reference info
        if page_refs:
            page_info = f" 📄 Pages: {', '.join(map(str, page_refs))}"
            page_color = "🟢" if len(page_refs) > 1 else "🔵"
        else:
            page_info = " ❓ No page refs"
            page_color = "⚪"
        
        with st.expander(f"{icon} {title} {status}{page_info}", expanded=False):
            if has_data:
                # Enhanced metadata section
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    # Safely handle content split - ensure it's a string
                    word_count = len(str(content).split()) if content else 0
                    st.caption(f"📊 **{word_count:,} words**")
                with col2:
                    if page_refs:
                        page_display = ", ".join(map(str, page_refs))
                        st.caption(f"📄 **Pages: {page_display}**")
                    else:
                        st.caption("📄 **No page references**")
                with col3:
                    # Show data quality indicator
                    if len(page_refs) > 1:
                        st.caption("✨ **Multi-page extraction**")
                    elif len(page_refs) == 1:
                        st.caption("📍 **Single-page source**")
                    else:
                        st.caption("⚠️ **Missing page info**")
                
                # Show content in a text area for easy copying
                st.text_area(
                    label="Content",
                    value=content,
                    height=250,  # Increased height for better readability
                    key=f"{prefix}_{field}_content",
                    label_visibility="collapsed"
                )
                
                # Add content analysis
                if page_refs:
                    with st.expander("🔍 **Content Analysis**", expanded=False):
                        st.markdown(f"**Source Quality:** {'High' if len(page_refs) > 1 else 'Medium' if len(page_refs) == 1 else 'Low'}")
                        st.markdown(f"**Page Coverage:** {len(page_refs)} page(s) referenced")
                        st.markdown(f"**Content Length:** {word_count:,} words ({len(str(content)):,} characters)")
                        
                        if len(page_refs) > 1:
                            st.success("✅ Multi-page extraction suggests comprehensive coverage")
                        elif len(page_refs) == 1:
                            st.info("ℹ️ Single-page source - may be concise or incomplete")
                        else:
                            st.warning("⚠️ No page tracking - extraction quality uncertain")
                
                # Human-in-the-loop review section
                st.markdown("---")
                st.markdown("**🔍 Review Extraction**")
                
                col_approve, col_refine = st.columns([1, 2])
                
                with col_approve:
                    approve_btn = st.button(
                        "✅ Approve",
                        key=f"approve_{prefix}_{field}",
                        help="Mark this extraction as correct",
                        use_container_width=True
                    )
                    
                    if approve_btn:
                        st.success("Approved!")
                        # Store approval in session state
                        if "approved_fields" not in st.session_state:
                            st.session_state.approved_fields = {}
                        st.session_state.approved_fields[field] = True
                
                with col_refine:
                    refine_btn = st.button(
                        "🔧 Request Refinement",
                        key=f"refine_{prefix}_{field}",
                        help="Provide feedback to improve this extraction",
                        use_container_width=True
                    )
                
                # Preserve the active tab when refinement button is clicked
                if refine_btn:
                    if "ui_state" not in st.session_state:
                        st.session_state.ui_state = {}
                    st.session_state.ui_state["active_extraction_tab"] = 1  # Stay on View JSON tab
                
                # Show refinement interface if button clicked or if recently re-extracted
                show_refine_ui = (refine_btn or 
                                st.session_state.get(f"show_refine_{field}", False) or 
                                st.session_state.get(f"reextracted_{field}", False))
                
                if show_refine_ui:
                    st.session_state[f"show_refine_{field}"] = True
                    
                    # Show success message if recently re-extracted
                    if st.session_state.get(f"reextracted_{field}", False):
                        st.success(f"✅ {field.replace('_', ' ').title()} was successfully re-extracted!")
                        # Clear the re-extraction flag after showing message
                        del st.session_state[f"reextracted_{field}"]
                    
                    feedback = st.text_area(
                        "What's wrong with this extraction? Provide specific feedback:",
                        placeholder="e.g., 'Missing secondary endpoints', 'Incorrect page numbers', 'Primary and secondary objectives are mixed up'",
                        key=f"feedback_{prefix}_{field}",
                        height=100
                    )
                    
                    col_submit, col_cancel = st.columns([1, 1])
                    with col_submit:
                        if st.button("🔄 Re-extract with Feedback", key=f"reextract_{prefix}_{field}", use_container_width=True, type="primary"):
                            if feedback and feedback.strip():
                                # Store refinement request in session state
                                if "refinement_requests" not in st.session_state:
                                    st.session_state.refinement_requests = {}
                                
                                st.session_state.refinement_requests[field] = feedback
                                st.info(f"Refinement requested for {title}. Re-running extraction...")
                                
                                # Trigger re-extraction for this specific field
                                _reextract_field_with_feedback(field, feedback)
                            else:
                                st.warning("Please provide specific feedback for refinement")
                    
                    with col_cancel:
                        if st.button("❌ Cancel", key=f"cancel_refine_{prefix}_{field}", use_container_width=True):
                            # Clear the refinement UI
                            if f"show_refine_{field}" in st.session_state:
                                del st.session_state[f"show_refine_{field}"]
                            # Preserve the active tab state (View JSON tab = 1)
                            if "ui_state" not in st.session_state:
                                st.session_state.ui_state = {}
                            st.session_state.ui_state["active_extraction_tab"] = 1
                            st.rerun()
                
            else:
                st.warning("No data extracted for this field")
                
                # Option to manually trigger extraction for empty fields
                if st.button(f"🔄 Retry Extraction", key=f"retry_{prefix}_{field}"):
                    st.info(f"Retrying extraction for {title}...")
                    _reextract_field_with_feedback(field, "Field was empty, please try to extract again with more focus")


def _reextract_field_with_feedback(field_name: str, feedback: str):
    """Re-extract a specific field with user feedback"""
    if not st.session_state.get("current_state"):
        st.error("No active extraction to refine")
        return
    
    try:
        from langgraph_custom.multi_turn_extractor import MultiTurnExtractor
        from langgraph_custom.enhanced_parser import EnhancedClinicalTrialParser
        
        # Get the current state
        state = st.session_state.current_state
        
        # Get full text - check cache first, then state, then try to re-extract
        full_text = None
        
        # Priority 1: Check session cache (most reliable)
        if "cached_full_text" in st.session_state and st.session_state.cached_full_text:
            full_text = st.session_state.cached_full_text
        # Priority 2: Check state
        elif "full_text" in state and state["full_text"]:
            full_text = state["full_text"]
            st.session_state.cached_full_text = full_text  # Cache it
        # Priority 3: For URL input type, fetch from URL
        elif state.get("input_type") == "url":
            st.error("Cannot re-extract from URL. Full text not cached. Please start a new extraction.")
            return
        # Priority 4: Try to get file from database and re-extract
        elif state.get("input_type") == "pdf":
            # First try the original file path
            file_path = state["input_url"]
            
            # If original path doesn't exist, check database for stored file
            import os
            if not os.path.exists(file_path):
                file_info = get_uploaded_file_path(st.session_state.current_convo_id)
                if file_info:
                    file_path, original_filename = file_info
                    st.info(f"📁 Using stored file: {original_filename}")
                else:
                    st.error("⚠️ PDF file no longer available and not found in database.")
                    st.info("💡 To re-extract this field, please upload the PDF document again in a new chat session.")
                    return
            
            # Check if the file exists now
            if not os.path.exists(file_path):
                st.error(f"⚠️ PDF file not found at: {file_path}")
                st.info("💡 The file may have been moved or deleted. Please upload the document again.")
                return
            
            # Extract and cache full text
            parser = EnhancedClinicalTrialParser()
            try:
                full_text, _ = parser.extract_text_multimethod(file_path)
                if full_text:
                    st.session_state.cached_full_text = full_text
                    state["full_text"] = full_text
                else:
                    st.error("Failed to extract text from PDF file.")
                    return
            except Exception as parse_error:
                st.error(f"Error parsing PDF: {str(parse_error)}")
                return
        else:
            st.error("Cannot determine input type for re-extraction.")
            return
        
        if not full_text or len(full_text.strip()) < 100:
            st.error("Insufficient text available for re-extraction. Please start a new extraction.")
            return
        
        # Initialize extractor with shared LLM instance
        try:
            from langgraph_custom.langgraph_workflow import get_shared_llm_instance
            shared_llm = get_shared_llm_instance()
            
            extractor = MultiTurnExtractor(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens_per_call=180000,
                delay_between_calls=2.0,
                llm_instance=shared_llm  # Use the shared LLM instance
            )
            
        except Exception as e:
            st.error(f"Failed to initialize extractor: {str(e)}")
            return
            
        with st.spinner(f"Re-extracting {field_name.replace('_', ' ').title()} with your feedback..."):
            # Re-extract this specific field with feedback
            refined_result = extractor.extract_field_with_feedback(
                full_text,
                field_name,
                feedback
            )
            
            if refined_result:
                # Update the state with new extraction
                st.session_state.current_state["parsed_json"][field_name] = refined_result
                
                # Also update data_to_summarize
                field_mapping = {
                    "study_overview": "Study Overview",
                    "brief_description": "Brief Description",
                    "primary_secondary_objectives": "Primary and Secondary Objectives",
                    "treatment_arms_interventions": "Treatment Arms and Interventions",
                    "eligibility_criteria": "Eligibility Criteria",
                    "enrollment_participant_flow": "Enrollment and Participant Flow",
                    "adverse_events_profile": "Adverse Events Profile",
                    "study_locations": "Study Locations",
                    "sponsor_information": "Sponsor Information"
                }
                
                display_key = field_mapping.get(field_name, field_name)
                content = refined_result.get("content", "") if isinstance(refined_result, dict) else str(refined_result) if refined_result else ""
                # Ensure content is a string
                if not isinstance(content, str):
                    content = str(content) if content else ""
                st.session_state.current_state["data_to_summarize"][display_key] = content
                
                # CRITICAL: Force UI state cache invalidation to ensure JSON view updates
                if "ui_state" not in st.session_state:
                    st.session_state.ui_state = {}
                st.session_state.ui_state["force_json_refresh"] = True
                st.session_state.ui_state["last_reextraction_field"] = field_name
                st.session_state.ui_state["last_reextraction_time"] = time.time()
                
                # Save updated state to database
                save_extraction_state(st.session_state.current_convo_id, st.session_state.current_state)
                
                # Save re-extraction to conversation history with detailed metadata
                reextraction_msg = f"🔄 Re-extracted {field_name.replace('_', ' ').title()} with feedback: {feedback}"
                metadata = {
                    "action": "re-extraction",
                    "field": field_name,
                    "feedback": feedback,
                    "success": True,
                    "extraction_method": "multi_turn_llm"
                }
                st.session_state.messages.append({"role": "system", "content": reextraction_msg})
                save_message_to_db(st.session_state.current_convo_id, "system", reextraction_msg, "re-extraction", metadata)
                
                # Also show the actual re-extracted content in chat
                content = refined_result.get("content", "") if isinstance(refined_result, dict) else str(refined_result) if refined_result else ""
                if content:
                    reextracted_content_msg = f"**📋 Updated {field_name.replace('_', ' ').title()}:**\n\n{content[:500]}{'...' if len(content) > 500 else ''}"
                    st.session_state.messages.append({"role": "assistant", "content": reextracted_content_msg, "type": "re-extracted_content"})
                    save_message_to_db(st.session_state.current_convo_id, "assistant", reextracted_content_msg, "re-extracted_content", {
                        "field": field_name,
                        "original_feedback": feedback,
                        "content_length": len(content)
                    })
                
                # Don't clear refinement UI state - let user see the success message in context
                # Keep the tab on View JSON to show updated results
                if "ui_state" not in st.session_state:
                    st.session_state.ui_state = {}
                st.session_state.ui_state["active_extraction_tab"] = 1  # View JSON tab
                
                # Set flag to update UI without full state loss
                st.session_state[f"reextracted_{field_name}"] = True
                
                # Clear any refinement UI state to close the feedback form
                if f"show_refine_{field_name}" in st.session_state:
                    del st.session_state[f"show_refine_{field_name}"]
                
                # Show success without full page rerun - let the UI update naturally
                st.success(f"✅ Re-extraction completed! Field '{field_name}' has been updated.")
                st.info(f"🔍 **Debug Info:** Updated content length: {len(content)} characters")
                st.info("💡 The updated results are now visible in the View JSON tab below.")
                
                # Use rerun to update the display
                st.rerun()
            else:
                st.error("Re-extraction failed. Please try again with different feedback.")
                # Save failed re-extraction attempt
                failure_metadata = {
                    "action": "re-extraction",
                    "field": field_name,
                    "feedback": feedback,
                    "success": False,
                    "error": "extraction_failed"
                }
                save_message_to_db(st.session_state.current_convo_id, "system", 
                                 f"❌ Re-extraction failed for {field_name.replace('_', ' ').title()}", 
                                 "re-extraction", failure_metadata)
        
    except Exception as e:
        st.error(f"Error during re-extraction: {str(e)}")
        # Save error to conversation history
        error_metadata = {
            "action": "re-extraction",
            "field": field_name,
            "feedback": feedback,
            "success": False,
            "error": str(e)
        }
        save_message_to_db(st.session_state.current_convo_id, "system", 
                         f"⚠️ Re-extraction error for {field_name.replace('_', ' ').title()}: {str(e)}", 
                         "error", error_metadata)

def _auto_reextract_empty_fields():
    """Automatically re-extract all empty or minimal fields"""
    if not st.session_state.get("current_state"):
        st.error("No active extraction to process")
        return
    
    state = st.session_state.current_state
    parsed_data = state.get("parsed_json", {})
    empty_fields = []
    
    # Find empty or minimal fields
    for field, value in parsed_data.items():
        if isinstance(value, dict):
            content = value.get("content", "")
        else:
            content = str(value) if value else ""
        
        # Check if field is empty or has minimal content
        if not content or len(content.strip()) <= 10:
            empty_fields.append(field)
    
    if not empty_fields:
        st.info("🎉 All fields already have substantial content!")
        return
    
    # Show which fields will be re-extracted
    st.info(f"📋 Found {len(empty_fields)} fields to re-extract: {', '.join([f.replace('_', ' ').title() for f in empty_fields])}")
    
    # Process each empty field individually
    success_count = 0
    failed_fields = []
    
    for i, field in enumerate(empty_fields):
        with st.spinner(f"🔄 Re-extracting field {i+1}/{len(empty_fields)}: {field.replace('_', ' ').title()}..."):
            try:
                from langgraph_custom.multi_turn_extractor import MultiTurnExtractor
                
                # Get full text
                full_text = st.session_state.get("cached_full_text") or state.get("full_text")
                if not full_text:
                    failed_fields.append(field)
                    continue
                
                # Initialize extractor with shared LLM instance
                try:
                    from langgraph_custom.langgraph_workflow import get_shared_llm_instance
                    shared_llm = get_shared_llm_instance()
                    
                    extractor = MultiTurnExtractor(
                        model="gpt-4o-mini",
                        temperature=0.1,
                        max_tokens_per_call=180000,
                        delay_between_calls=2.0,
                        llm_instance=shared_llm  # Use shared LLM
                    )
                    
                except Exception as init_error:
                    st.error(f"Failed to initialize extractor for {field}: {str(init_error)}")
                    failed_fields.append(field)
                    continue
                
                # Re-extract this field
                refined_result = extractor.extract_field_with_feedback(
                    full_text,
                    field,
                    "Field was empty, please try to extract again with more focus and detail"
                )
                
                if refined_result:
                    # Update the state
                    state["parsed_json"][field] = refined_result
                    
                    # Update data_to_summarize
                    field_mapping = {
                        "study_overview": "Study Overview",
                        "brief_description": "Brief Description",
                        "primary_secondary_objectives": "Primary and Secondary Objectives",
                        "treatment_arms_interventions": "Treatment Arms and Interventions",
                        "eligibility_criteria": "Eligibility Criteria",
                        "enrollment_participant_flow": "Enrollment and Participant Flow",
                        "adverse_events_profile": "Adverse Events Profile",
                        "study_locations": "Study Locations",
                        "sponsor_information": "Sponsor Information"
                    }
                    
                    display_key = field_mapping.get(field, field)
                    content = refined_result.get("content", "") if isinstance(refined_result, dict) else str(refined_result) if refined_result else ""
                    if content:
                        state["data_to_summarize"][display_key] = content
                    
                    success_count += 1
                    st.success(f"✅ {field.replace('_', ' ').title()}")
                    
                    # Show the extracted content in chat for this field
                    content = refined_result.get("content", "") if isinstance(refined_result, dict) else str(refined_result) if refined_result else ""
                    if content:
                        field_content_msg = f"**📋 Auto-extracted {field.replace('_', ' ').title()}:**\n\n{content[:300]}{'...' if len(content) > 300 else ''}"
                        st.session_state.messages.append({"role": "assistant", "content": field_content_msg, "type": "auto_extracted_content"})
                        save_message_to_db(st.session_state.current_convo_id, "assistant", field_content_msg, "auto_extracted_content", {
                            "field": field,
                            "content_length": len(content),
                            "auto_extraction": True
                        })
                else:
                    failed_fields.append(field)
                    
            except Exception as e:
                failed_fields.append(field)
                st.error(f"❌ {field.replace('_', ' ').title()}: {str(e)}")
    
    # Save updated state
    if success_count > 0:
        save_extraction_state(st.session_state.current_convo_id, state)
        
        # Log the auto re-extraction activity
        auto_reextract_msg = f"🔄 Auto re-extracted {success_count} empty fields"
        metadata = {
            "action": "auto_re_extraction",
            "success_count": success_count,
            "failed_count": len(failed_fields),
            "processed_fields": empty_fields
        }
        st.session_state.messages.append({"role": "system", "content": auto_reextract_msg})
        save_message_to_db(st.session_state.current_convo_id, "system", auto_reextract_msg, "auto_re_extraction", metadata)
        
        # Add summary of what was extracted
        if success_count > 0:
            successful_fields = [f for f in empty_fields if f not in failed_fields]
            summary_msg = f"✅ **Auto-extraction complete!** Successfully extracted content for: {', '.join([f.replace('_', ' ').title() for f in successful_fields])}"
            if failed_fields:
                summary_msg += f"\n\n⚠️ Failed to extract: {', '.join([f.replace('_', ' ').title() for f in failed_fields])}"
            st.session_state.messages.append({"role": "assistant", "content": summary_msg, "type": "auto_extraction_summary"})
            save_message_to_db(st.session_state.current_convo_id, "assistant", summary_msg, "auto_extraction_summary", metadata)
    
    # Show final results
    if success_count > 0:
        st.success(f"✅ Successfully re-extracted {success_count} out of {len(empty_fields)} fields!")
        if failed_fields:
            st.warning(f"⚠️ Failed to re-extract: {', '.join([f.replace('_', ' ').title() for f in failed_fields])}")
        st.info("💡 Check the View JSON tab to see the updated results.")
        
        # Switch to JSON view to show results
        st.session_state.ui_state["active_extraction_tab"] = 1
        time.sleep(1)
        st.rerun()
    else:
        st.error("❌ Failed to re-extract any empty fields. Please try manual re-extraction with specific feedback.")

def _regenerate_summary(state):
    """Regenerate summary with updated extracted data"""
    try:
        from langgraph_custom.langgraph_workflow import chat_node_stream
        
        # Ensure we have valid state and data to work with
        if not state or not state.get("parsed_json"):
            st.error("No extraction data available to generate summary from.")
            return
        
        with st.spinner("🔄 Regenerating summary with updated data..."):
            # Update data_to_summarize with current parsed_json
            field_mapping = {
                "study_overview": "Study Overview",
                "brief_description": "Brief Description", 
                "primary_secondary_objectives": "Primary and Secondary Objectives",
                "treatment_arms_interventions": "Treatment Arms and Interventions",
                "eligibility_criteria": "Eligibility Criteria",
                "enrollment_participant_flow": "Enrollment and Participant Flow",
                "adverse_events_profile": "Adverse Events Profile",
                "study_locations": "Study Locations",
                "sponsor_information": "Sponsor Information"
            }
            
            # Rebuild data_to_summarize from current parsed_json
            state["data_to_summarize"] = {}
            fields_with_content = 0
            
            for field, value in state.get("parsed_json", {}).items():
                if isinstance(value, dict):
                    content = value.get("content", "")
                else:
                    content = str(value) if value else ""
                
                # Ensure content is always a string
                if not isinstance(content, str):
                    content = str(content) if content else ""
                
                if content and content.strip():
                    display_key = field_mapping.get(field, field)
                    state["data_to_summarize"][display_key] = content
                    fields_with_content += 1
            
            # Check if we have enough data to generate a meaningful summary
            if fields_with_content < 3:
                st.warning(f"Only {fields_with_content} fields have content. Consider re-extracting empty fields before regenerating summary.")
                if not st.button("Continue anyway", key="continue_regenerate"):
                    return
            
            # Generate new summary
            with st.chat_message("assistant"):
                st.markdown("### 📝 Updated Summary")
                message_placeholder = st.empty()
                full_response = ""
                
                # Stream the new summary
                for chunk in chat_node_stream(state):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Add PDF download button
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    try:
                        pdf_data = create_summary_pdf(full_response, state.get('nct_id', 'study'))
                        st.download_button(
                            label="📄 Download PDF",
                            data=pdf_data,
                            file_name=f"summary_{state.get('nct_id', 'study')}_updated_{int(time.time())}.pdf",
                            mime="application/pdf",
                            key=f"regenerated_pdf_download_{int(time.time())}"
                        )
                    except Exception as e:
                        st.error("PDF error")
                st.markdown("---")
            
            # Save to messages and database with metadata to indicate regenerated summary
            summary_metadata = {
                "action": "regenerate_summary",
                "timestamp": time.time(),
                "fields_count": len([k for k, v in state.get('parsed_json', {}).items() if v]),
                "nct_id": state.get('nct_id', 'study'),
                "fields_with_content": fields_with_content
            }
            
            # Add a brief regeneration notice before the full summary
            regen_notice = f"🔄 **Summary Regenerated** using {fields_with_content} fields from {state.get('nct_id', 'the study')}"
            st.session_state.messages.append({"role": "system", "content": regen_notice, "type": "regeneration_notice"})
            save_message_to_db(st.session_state.current_convo_id, "system", regen_notice, "regeneration_notice", summary_metadata)
            
            # Add the full regenerated summary
            st.session_state.messages.append({"role": "assistant", "content": full_response, "type": "regenerated_summary"})
            save_message_to_db(st.session_state.current_convo_id, "assistant", full_response, "regenerated_summary", summary_metadata)
            
            # Update the current state with the new summary info
            state["last_summary_regenerated"] = time.time()
            save_extraction_state(st.session_state.current_convo_id, state)
            
            st.success("✅ Summary regenerated successfully! You can regenerate again anytime as you continue refining fields.")
            
    except Exception as e:
        st.error(f"Error regenerating summary: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def create_summary_pdf(summary_text, nct_id):
    """Create PDF from summary text"""
    try:
        from fpdf import FPDF
        import unicodedata
        
        def clean_text_for_pdf(text):
            if not text:
                return ""
            try:
                cleaned = text.encode('ascii', 'ignore').decode('ascii')
                return cleaned
            except:
                normalized = unicodedata.normalize('NFKD', text)
                ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')
                return ascii_text
        
        class CustomPDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 14)
                self.set_text_color(0, 51, 102)
                self.cell(0, 10, f'Clinical Trial Summary: {nct_id}', 0, 1, 'C')
                self.ln(3)
            
            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.set_text_color(128, 128, 128)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
        pdf = CustomPDF()
        pdf.add_page()
        pdf.set_margins(15, 25, 15)
        
        # Add URL link
        pdf.set_font("Arial", 'U', 10)
        url_text = f"https://clinicaltrials.gov/study/{nct_id}"
        pdf.set_text_color(0, 0, 255)
        pdf.cell(0, 8, url_text, 0, 1, 'C', link=url_text)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(8)

        clean_summary = clean_text_for_pdf(summary_text)
        lines = clean_summary.split('\n')
        
        for line in lines:
            try:
                line = line.strip()
                if not line:
                    pdf.ln(3)
                    continue
                
                if line.startswith('# '):
                    pdf.ln(5)
                    pdf.set_font("Arial", 'B', 16)
                    pdf.set_text_color(0, 51, 102)
                    pdf.multi_cell(0, 8, line.replace('# ', ''))
                    pdf.set_text_color(0, 0, 0)
                    pdf.ln(3)
                elif line.startswith('## '):
                    pdf.ln(6)
                    pdf.set_font("Arial", 'B', 14)
                    pdf.multi_cell(0, 7, line.replace('## ', ''))
                    pdf.ln(2)
                elif line.startswith('### '):
                    pdf.ln(4)
                    pdf.set_font("Arial", 'B', 12)
                    pdf.multi_cell(0, 6, line.replace('### ', ''))
                    pdf.ln(2)
                elif '**' in line:
                    pdf.set_font("Arial", 'B', 11)
                    pdf.multi_cell(0, 6, line.replace('**', ''))
                elif line.startswith('- ') or line.startswith('• '):
                    pdf.set_font("Arial", '', 10)
                    pdf.cell(8, 6, '', 0, 0)
                    pdf.multi_cell(0, 6, line)
                else:
                    pdf.set_font("Arial", '', 10)
                    pdf.multi_cell(0, 6, line)
            except:
                continue

        return pdf.output(dest='S').encode('latin1', 'ignore')
        
    except Exception as e:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "PDF Generation Error", ln=True)
        pdf.cell(0, 10, f"NCT ID: {nct_id}", ln=True)
        return pdf.output(dest='S').encode('latin1', 'ignore')

# Page configuration
st.set_page_config(
    page_title="ClinicalIQ – Clinical Protocol Intelligence & Q&A",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if "workflow_app" not in st.session_state:
    # Use cached workflow instead of rebuilding every time
    st.session_state.workflow_app = get_workflow()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_convo_id" not in st.session_state:
    st.session_state.current_convo_id = str(uuid.uuid4())
    st.session_state.url_key = str(uuid.uuid4())

if "current_state" not in st.session_state:
    st.session_state.current_state = None

# Initialize UI state persistence
if "ui_state" not in st.session_state:
    st.session_state.ui_state = {
        "active_extraction_tab": 0,
        "expanded_fields": {},
        "refinement_ui_open": {}
    }

# Reduce top padding
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 0rem;
        }
    </style>
""", unsafe_allow_html=True)

# Clean header with logo
logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "ctis-2024.png")
if os.path.exists(logo_path):
    st.markdown("""
        <div style='display: flex; align-items: center; justify-content: center; gap: 25px; margin: 0; padding: 15px 0; background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);'>
            <img src='data:image/png;base64,{}' width='110' style='flex-shrink: 0;'/>
            <div style='text-align: center;'>
                <h1 style='margin: 0; padding: 0; font-size: 3.5em; font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-weight: 700; letter-spacing: -1.5px;'>
                    <span style='color: #C1272D;'>Cli</span><span style='color: #1E293B;'>nicalI</span><span style='color: #C1272D;'>Q</span>
                </h1>
                <p style='color: #64748B; margin: 8px 0 0 0; padding: 0; font-size: 1.2em; font-weight: 500; font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>AI-Powered Clinical Protocol Intelligence Platform</p>
            </div>
        </div>
    """.format(__import__('base64').b64encode(open(logo_path, 'rb').read()).decode()), unsafe_allow_html=True)
else:
    st.markdown("""
        <div style='text-align: center; margin: 0; padding: 15px 0; background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);'>
            <h1 style='margin: 0; padding: 0; font-size: 3.5em; font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-weight: 700; letter-spacing: -1.5px;'>
                <span style='color: #C1272D;'>Cli</span><span style='color: #1E293B;'>nicalI</span><span style='color: #C1272D;'>Q</span>
            </h1>
            <p style='color: #64748B; margin: 8px 0 0 0; padding: 0; font-size: 1.2em; font-weight: 500; font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>AI-Powered Clinical Protocol Intelligence Platform</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar
st.sidebar.button("➕ Start New Chat", key="new_chat_button", on_click=new_chat_click, use_container_width=True)

st.sidebar.divider()

# Display database statistics button
display_database_statistics()

st.sidebar.divider()

# Use cached conversation list for faster sidebar rendering
conversations = get_cached_conversations()
if conversations:
    st.sidebar.subheader("💬 Past Chats")
    
    # Create dropdown options - show "Current" for active conversation
    options = ["Select a conversation..."] + conversations
    default_index = 0
    
    # Find current conversation in list
    if st.session_state.current_convo_id in conversations:
        default_index = conversations.index(st.session_state.current_convo_id) + 1
    
    selected_convo = st.sidebar.selectbox(
        "Load previous conversation",
        options=options,
        index=default_index,
        key="convo_selector",
        label_visibility="collapsed"
    )
    
    # Load selected conversation if changed
    if selected_convo != "Select a conversation..." and selected_convo != st.session_state.current_convo_id:
        st.session_state.messages = load_messages_from_db(selected_convo)
        st.session_state.current_convo_id = selected_convo
        
        # Restore extraction state if available
        saved_state = load_extraction_state(selected_convo)
        if saved_state:
            st.session_state.current_state = saved_state
        else:
            st.session_state.current_state = None
        
        # Restore state if available (legacy support)
        for msg in st.session_state.messages:
            if msg["role"] == "assistant" and "Clinical Trial Summary" in msg["content"]:
                nct_match = re.search(r"NCT\d{8}", msg["content"])
                if nct_match:
                    st.session_state.current_summary = msg["content"]
                break
        st.rerun()

# Welcome message if no messages
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("""👋 Welcome to **ClinicalIQ** - Your AI-powered clinical protocol intelligence platform!

📄 **Upload Protocols**
Start with a clinical trial PDF or a ClinicalTrials.gov URL.

🔍 **Automated Data Extraction**
Key fields—study overview, objectives, treatments, eligibility, and more—are extracted into structured JSON, and you can review, refine, or approve them.

📊 **Summary Generation**
Create clear study summaries based on the extracted data.

💬 **Ask Anything**
Query your study data using natural language for fast insights.

🔎 **Find Related Studies**
Search for clinical trials that use a specific drug or intervention, or find studies similar to an uploaded study/trial.

Get started by uploading a protocol or entering a ClinicalTrials.gov URL.""")
    st.markdown("")

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Check if this is the extraction results marker
        if message["content"] == "EXTRACTION_RESULTS_MARKER" and st.session_state.current_state:
            st.markdown("### 📊 Extraction Results")
            create_extraction_results_tabs(st.session_state.current_state)
        else:
            st.markdown(message["content"])

# File uploader - clean minimal design with hidden label
if not st.session_state.messages:
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key="pdf_uploader",
        label_visibility="hidden"
    )
else:
    uploaded_file = None

# Initialize variables
url_input = None
nct_match = None

# Handle initial URL input
if url_input and nct_match and not st.session_state.messages:
    nct_number = nct_match.group(0)
    st.info(f"Found NCT number: **{nct_number}**. Processing through LangGraph workflow...")
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": f"URL: {url_input}"})
    with st.chat_message("user"):
        st.markdown(f"URL: {url_input}")
    save_message_to_db(st.session_state.current_convo_id, "user", f"URL: {url_input}")
    
    # Run workflow
    with st.spinner("Extracting data from ClinicalTrials.gov..."):
        try:
            initial_state = {
                "input_url": url_input,
                "input_type": "unknown",
                "raw_data": {},
                "parsed_json": {},
                "data_to_summarize": {},
                "confidence_score": 0.0,
                "completeness_score": 0.0,
                "missing_fields": [],
                "nct_id": "",
                "chat_query": "generate_summary",
                "chat_response": "",
                "stream_response": None,
                "error": "",
                "used_llm_fallback": False
            }
            
            # Process through workflow (non-streaming for initial extraction)
            result = st.session_state.workflow_app.invoke(initial_state)
            
            if result.get("error"):
                st.error(f"Error: {result['error']}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {result['error']}"})
                save_message_to_db(st.session_state.current_convo_id, "assistant", f"Error: {result['error']}")
            else:
                st.session_state.current_state = result
                
                # Cache full_text if available to prevent file access issues later
                if "full_text" in result and result["full_text"]:
                    st.session_state.cached_full_text = result["full_text"]
                
                # Show extraction results in tabs
                with st.chat_message("assistant"):
                    st.markdown("### 📊 Extraction Results")
                    create_extraction_results_tabs(result)
                
                # Save extraction results marker to chat history for persistence
                st.session_state.messages.append({"role": "assistant", "content": "EXTRACTION_RESULTS_MARKER"})
                save_message_to_db(st.session_state.current_convo_id, "assistant", "EXTRACTION_RESULTS_MARKER")
                save_extraction_state(st.session_state.current_convo_id, result)
                
                # Stream the summary
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Use streaming function
                    for chunk in chat_node_stream(result):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                    # Add PDF download button right after summary
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col1:
                        try:
                            pdf_data = create_summary_pdf(full_response, result.get('nct_id', 'study'))
                            st.download_button(
                                label="📄 Download PDF",
                                data=pdf_data,
                                file_name=f"summary_{result.get('nct_id', 'study')}.pdf",
                                mime="application/pdf",
                                key="url_initial_pdf_download"
                            )
                        except Exception as e:
                            st.error("PDF error")
                    st.markdown("---")
                
                # Save to database
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                save_message_to_db(st.session_state.current_convo_id, "assistant", full_response)
                
                st.balloons()
                
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            save_message_to_db(st.session_state.current_convo_id, "assistant", error_msg)

# Handle PDF upload
if uploaded_file is not None and not st.session_state.messages:
    # Create uploads directory if it doesn't exist
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    # Generate unique filename to avoid conflicts
    import time
    timestamp = str(int(time.time()))
    safe_filename = f"{timestamp}_{uploaded_file.name}"
    permanent_path = uploads_dir / safe_filename
    
    # Save file permanently
    with open(permanent_path, 'wb') as f:
        f.write(uploaded_file.getvalue())
    
    # Save file info to database
    save_uploaded_file(
        st.session_state.current_convo_id, 
        uploaded_file.name, 
        str(permanent_path), 
        len(uploaded_file.getvalue())
    )
    
    # Save upload event to conversation history
    upload_msg = f"📄 Uploaded PDF: {uploaded_file.name} ({len(uploaded_file.getvalue())} bytes)"
    upload_metadata = {
        "action": "file_upload",
        "filename": uploaded_file.name,
        "file_size": len(uploaded_file.getvalue()),
        "stored_path": str(permanent_path)
    }
    save_message_to_db(st.session_state.current_convo_id, "system", upload_msg, "file_upload", upload_metadata)
    
    # Add user message - show parsing status
    with st.chat_message("assistant"):
        st.markdown(f"🔍 Parsing uploaded PDF: **{uploaded_file.name}**")
    
    # Run workflow with progress tracking for multi-turn extraction
    try:
        # Create a single status message that updates in place
        status_msg = st.empty()
        
        initial_state = {
            "input_url": str(permanent_path),
            "input_type": "unknown",
            "raw_data": {},
            "parsed_json": {},
            "parser_only_json": {},
            "data_to_summarize": {},
            "confidence_score": 0.0,
            "completeness_score": 0.0,
            "missing_fields": [],
            "nct_id": "",
            "chat_query": "generate_summary",
            "chat_response": "",
            "stream_response": None,
            "error": "",
            "used_llm_fallback": False,
            "extraction_progress": {},
            "extraction_cost_estimate": {},
            "progress_log": []
        }
        
        # Show initial parsing step
        status_msg.info("🔍 Step 1/3: Analyzing PDF with enhanced parser...")
        
        # Stream workflow execution to show real-time updates
        result = None
        last_progress_count = 0
        
        for chunk in st.session_state.workflow_app.stream(initial_state):
            # Get the latest state from the chunk
            for node_name, node_state in chunk.items():
                if isinstance(node_state, dict):
                    result = node_state
                    # Check for NEW progress log updates
                    progress_log = node_state.get("progress_log", [])
                    if progress_log and len(progress_log) > last_progress_count:
                        # Show only the new messages
                        for i in range(last_progress_count, len(progress_log)):
                            status_msg.info(f"⚙️ {progress_log[i]}")
                        last_progress_count = len(progress_log)
        
        # Use the final result
        if result is None:
            result = st.session_state.workflow_app.invoke(initial_state)
        
        # Show final status with cost if LLM was used
        if result.get("used_llm_fallback"):
            cost_est = result.get("extraction_cost_estimate", {})
            if cost_est:
                status_msg.success(f"✅ Multi-turn extraction complete | Cost: ${cost_est.get('total_cost', 0):.3f} | Time: {cost_est.get('estimated_time_minutes', 0):.1f} min")
                import time
                time.sleep(1)
        
        # Clear progress indicator
        status_msg.empty()
        
        if result.get("error"):
            st.error(f"Error: {result['error']}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {result['error']}"})
            save_message_to_db(st.session_state.current_convo_id, "assistant", f"Error: {result['error']}")
        else:
            # Store result in session state
            st.session_state.current_state = result
            
            # Cache full_text if available to prevent file access issues later
            if "full_text" in result and result["full_text"]:
                st.session_state.cached_full_text = result["full_text"]
            
            # Show extraction results in tabs
            with st.chat_message("assistant"):
                st.markdown("### 📊 Extraction Results")
                create_extraction_results_tabs(result)
            
            # Save extraction results message to chat history
            st.session_state.messages.append({"role": "assistant", "content": "EXTRACTION_RESULTS_MARKER"})
            save_message_to_db(st.session_state.current_convo_id, "assistant", "EXTRACTION_RESULTS_MARKER")
            save_extraction_state(st.session_state.current_convo_id, result)
            
            # Stream the summary
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                for chunk in chat_node_stream(result):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Add PDF download button right after summary
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    try:
                        pdf_data = create_summary_pdf(full_response, result.get('nct_id', 'study'))
                        st.download_button(
                            label="📄 Download PDF",
                            data=pdf_data,
                            file_name=f"summary_{result.get('nct_id', 'study')}.pdf",
                            mime="application/pdf",
                            key="pdf_chat_pdf_download"
                        )
                    except Exception as e:
                        st.error("PDF error")
                st.markdown("---")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            save_message_to_db(st.session_state.current_convo_id, "assistant", full_response)
            
            st.balloons()
            
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        save_message_to_db(st.session_state.current_convo_id, "assistant", error_msg)

# Handle chat input (URLs, questions, with file upload support and RAG suggestions)
if prompt := st.chat_input("Ask a question, search for clinical trials (e.g., 'Find studies using Nivolumab'), or paste a ClinicalTrials.gov URL..."):
    # Check if it's a URL
    nct_match = re.search(r'NCT\d{8}', prompt)
    is_url = nct_match is not None or 'clinicaltrials.gov' in prompt.lower()
    
    if is_url and not st.session_state.messages:
        # Handle URL input
        url_input = prompt
        nct_number = nct_match.group(0) if nct_match else "Unknown"
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": f"URL: {url_input}"})
        with st.chat_message("user"):
            st.markdown(f"URL: {url_input}")
        save_message_to_db(st.session_state.current_convo_id, "user", f"URL: {url_input}")
        
        # Run workflow
        with st.spinner("Extracting data from ClinicalTrials.gov..."):
            try:
                initial_state = {
                    "input_url": url_input,
                    "input_type": "unknown",
                    "raw_data": {},
                    "parsed_json": {},
                    "data_to_summarize": {},
                    "confidence_score": 0.0,
                    "completeness_score": 0.0,
                    "missing_fields": [],
                    "nct_id": "",
                    "chat_query": "generate_summary",
                    "chat_response": "",
                    "stream_response": None,
                    "error": "",
                    "used_llm_fallback": False
                }
                
                # Process through workflow
                result = st.session_state.workflow_app.invoke(initial_state)
                
                if result.get("error"):
                    st.error(f"Error: {result['error']}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {result['error']}"})
                    save_message_to_db(st.session_state.current_convo_id, "assistant", f"Error: {result['error']}")
                else:
                    # Store result in session state for persistence
                    st.session_state.current_state = result
                    
                    # Cache full_text if available to prevent file access issues later
                    if "full_text" in result and result["full_text"]:
                        st.session_state.cached_full_text = result["full_text"]
                    
                    # Show extraction results in tabs
                    with st.chat_message("assistant"):
                        st.markdown("### 📊 Extraction Results")
                        create_extraction_results_tabs(result)
                    
                    # Save extraction results marker to database for persistence
                    st.session_state.messages.append({"role": "assistant", "content": "EXTRACTION_RESULTS_MARKER"})
                    save_message_to_db(st.session_state.current_convo_id, "assistant", "EXTRACTION_RESULTS_MARKER")
                    save_extraction_state(st.session_state.current_convo_id, result)
                    
                    # Stream summary
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        full_response = ""
                        
                        for chunk in chat_node_stream(result):
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                        
                        message_placeholder.markdown(full_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    save_message_to_db(st.session_state.current_convo_id, "assistant", full_response)
                    
                    # Add PDF download button in chat
                    with st.chat_message("assistant"):
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col1:
                            try:
                                pdf_data = create_summary_pdf(full_response, result.get('nct_id', 'study'))
                                st.download_button(
                                    label="📄 Download PDF",
                                    data=pdf_data,
                                    file_name=f"summary_{result.get('nct_id', 'study')}.pdf",
                                    mime="application/pdf",
                                    key="chat_input_pdf_download"
                                )
                            except Exception as e:
                                st.error("PDF error")
                        st.markdown("---")
                    
                    st.rerun()
                    
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                save_message_to_db(st.session_state.current_convo_id, "assistant", error_msg)
    else:
        # Handle regular chat questions
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        save_message_to_db(st.session_state.current_convo_id, "user", prompt)
        
        # Generate response with streaming
        with st.chat_message("assistant"):
            # Check if this might be a RAG query (even without existing state)
            from langgraph_custom.langgraph_workflow import should_use_rag_tool
            
            if st.session_state.current_state:
                # Update state with new query
                chat_state = st.session_state.current_state.copy()
                chat_state["chat_query"] = prompt
                
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    # Stream response
                    for chunk in chat_node_stream(chat_state):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                    # Save to session and database
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    save_message_to_db(st.session_state.current_convo_id, "assistant", full_response)
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    save_message_to_db(st.session_state.current_convo_id, "assistant", error_msg)
            elif should_use_rag_tool(prompt):
                # Handle RAG-only queries without requiring document processing
                try:
                    # Create a minimal state for RAG-only processing
                    rag_state = {
                        "input_url": "",
                        "input_type": "rag_only",
                        "raw_data": {},
                        "parsed_json": {},
                        "data_to_summarize": {},
                        "confidence_score": 0.0,
                        "completeness_score": 0.0,
                        "missing_fields": [],
                        "nct_id": "",
                        "chat_query": prompt,
                        "chat_response": "",
                        "stream_response": None,
                        "error": "",
                        "used_llm_fallback": False,
                        "extraction_progress": {},
                        "extraction_cost_estimate": {},
                        "progress_log": [],
                        "use_rag_tool": False,
                        "rag_tool_results": ""
                    }
                    
                    # Process through workflow for RAG search
                    result = st.session_state.workflow_app.invoke(rag_state)
                    
                    if result.get("error"):
                        st.error(f"Error: {result['error']}")
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {result['error']}"})
                        save_message_to_db(st.session_state.current_convo_id, "assistant", f"Error: {result['error']}")
                    else:
                        # Stream the RAG response
                        message_placeholder = st.empty()
                        full_response = ""
                        
                        for chunk in chat_node_stream(result):
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                        
                        message_placeholder.markdown(full_response)
                        
                        # Save to session and database
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        save_message_to_db(st.session_state.current_convo_id, "assistant", full_response)
                        
                except Exception as e:
                    error_msg = f"Error processing RAG query: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    save_message_to_db(st.session_state.current_convo_id, "assistant", error_msg)
            else:
                no_data_msg = "Please process a document first before asking questions, or try searching for clinical trials (e.g., 'Find studies that use Nivolumab')."
                st.info(no_data_msg)
                st.session_state.messages.append({"role": "assistant", "content": no_data_msg})
                save_message_to_db(st.session_state.current_convo_id, "assistant", no_data_msg)
