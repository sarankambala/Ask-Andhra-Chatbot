import streamlit as st

# Set page config
st.set_page_config(
    page_title="Ask Andhra RAG Intelligence Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Smart Chat"

def main():
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        
        # App selection buttons at the top
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧠 Smart Chat", 
                        use_container_width=True,
                        type="primary" if st.session_state.current_page == "Enhanced RAG Chat" else "secondary"):
                st.session_state.current_page = "Smart Chat"
                st.rerun()
        
        with col2:
            if st.button("💡 AI Insights", 
                        use_container_width=True,
                        type="primary" if st.session_state.current_page == "AI Insights" else "secondary"):
                st.session_state.current_page = "AI Insights"
                st.rerun()
        
        st.markdown("---")
        
        # Display current page info
        st.info(f"**Current App:** {st.session_state.current_page}")
    
    # Load the appropriate page based on selection
    if st.session_state.current_page == "Smart Chat":
        from multiragchat import main as rag_main
        rag_main()
    else:
        from insights_ai import main as insights_main
        insights_main()

if __name__ == "__main__":
    main()