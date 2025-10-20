import streamlit as st
import weaviate
from weaviate.classes.query import Filter
import pdfplumber
from sentence_transformers import SentenceTransformer
from groq import Groq
import time
import os
import json
import random

# Configuration
WEAVIATE_URL = "m8l0s1ucsfgj0pcvv4k4ra.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "d3pqMVVjeFZuNk9ZeXFTM19PYXdsVlBZdGQ3OXIzTGtvZHJQV3J6MHA2czZjdWcwcmJtQ3B1ZUd2VHhJPV92MjAw"
GROQ_API_KEY = "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"
MODEL_NAME = "llama-3.1-8b-instant"

# Initialize clients
@st.cache_resource
def init_weaviate_client():
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY)
    )

@st.cache_resource
def init_embedding_model():
    return SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)

@st.cache_resource
def init_groq_client():
    return Groq(api_key=GROQ_API_KEY)

def get_collection_samples(collection_name, sample_size=10):
    """Get sample documents from a collection to generate insights"""
    try:
        client = init_weaviate_client()
        collection = client.collections.get(collection_name)
        
        response = collection.query.fetch_objects(
            limit=sample_size,
            return_properties=["content", "page", "chunk_id"]
        )
        
        samples = []
        if response.objects:
            for obj in response.objects:
                samples.append({
                    'content': obj.properties['content'],
                    'page': obj.properties.get('page', 'N/A'),
                })
        
        return samples
    except Exception as e:
        st.error(f"Error fetching samples from {collection_name}: {e}")
        return []

def generate_ai_insights(collection_name, user_role, num_insights=4):
    """Generate AI-driven insights from collection content based on user role"""
    
    # Get sample content from the collection
    samples = get_collection_samples(collection_name)
    
    if not samples:
        return []
    
    # Prepare context from samples
    context = "\n\n".join([f"Document excerpt (Page {sample['page']}): {sample['content'][:500]}..." for sample in samples])
    
    # Role-based system prompts
    role_prompts = {
        "public": """You are creating insights for general public users. Focus on:
- Public benefits and community impact
- General progress and timelines
- Environmental and social considerations
- Accessible, non-technical information
- Overall project goals and public value""",
        
        "govt": """You are creating insights for government officials and technical experts. Focus on:
- Budget analysis and financial details
- Technical specifications and engineering details
- Governance structures and compliance
- Risk assessment and mitigation strategies
- Performance metrics and KPIs
- Strategic recommendations and decision support"""
    }
    
    system_prompt = f"""You are an expert document analyst. Your task is to generate insightful and engaging topics/questions that would help users explore and understand the document content.

{role_prompts.get(user_role, role_prompts['public'])}

INSIGHT TEMPLATE (generate exactly this format for each insight):
{{
  "heading": "Creative and engaging title",
  "description": "Brief description of what this insight explores (1-2 sentences)",
  "query": "Specific question that when asked will reveal comprehensive information about this topic"
}}

CRITERIA:
- Create insights that cover different aspects of the document
- Ensure queries are specific and will yield comprehensive answers
- Cover technical, strategic, economic, and implementation aspects
- Make insights mutually exclusive but collectively exhaustive
- Tailor the complexity and focus based on the user role
"""

    user_message = f"""Document Collection: {collection_name}
User Role: {user_role}

Document Content Samples:
{context}

Generate exactly {num_insights} insights following the template above. Return ONLY a JSON array without any additional text.

Example format:
[
  {{
    "heading": "Project Scope & Vision",
    "description": "Explore the main objectives and long-term vision of the project",
    "query": "What are the primary objectives, scope, and long-term vision of this project?"
  }},
  {{
    "heading": "Technical Specifications", 
    "description": "Detailed technical features and construction specifications",
    "query": "What are the key technical specifications, design features, and construction methodologies used?"
  }}
]

Now generate {num_insights} insights for {collection_name} tailored for {user_role} users:"""

    try:
        groq_client = init_groq_client()
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model=MODEL_NAME,
            temperature=0.8,
            max_tokens=1500,
        )
        
        response_text = chat_completion.choices[0].message.content
        # Clean the response to extract JSON
        response_text = response_text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        insights = json.loads(response_text)
        
        # Add collection and unique ID to each insight
        for i, insight in enumerate(insights):
            insight['id'] = f"{collection_name.lower()}_{user_role}_{i+1}"
            insight['collection'] = collection_name
            insight['role'] = user_role
        
        return insights
        
    except Exception as e:
        st.error(f"Error generating insights: {e}")
        # Return default insights if AI generation fails
        return get_default_insights(collection_name, user_role, num_insights)

def get_default_insights(collection_name, user_role, num_insights):
    """Provide default insights if AI generation fails"""
    default_templates = {
        "public": {
            "PolavaramProject": [
                {
                    "heading": "Project Overview",
                    "description": "Key objectives and public benefits of the Polavaram project",
                    "query": "What are the main objectives and how will the Polavaram project benefit local communities?"
                },
                {
                    "heading": "Project Progress", 
                    "description": "Current status and expected completion timeline",
                    "query": "What is the current progress and expected completion date for the Polavaram project?"
                }
            ],
            "AmaravathiCapitalCity": [
                {
                    "heading": "Development Vision",
                    "description": "Master plan and development approach for the capital city",
                    "query": "What is the master plan and vision for Amaravathi capital city development?"
                },
                {
                    "heading": "Public Benefits",
                    "description": "How the capital city will benefit residents and the region",
                    "query": "What are the main benefits and advantages of developing Amaravathi as the capital city?"
                }
            ]
        },
        "govt": {
            "PolavaramProject": [
                {
                    "heading": "Budget Analysis",
                    "description": "Detailed financial breakdown and funding allocation",
                    "query": "What is the detailed budget breakdown, funding sources, and expenditure analysis for Polavaram project?"
                },
                {
                    "heading": "Technical Specifications",
                    "description": "Engineering details and construction methodologies",
                    "query": "What are the comprehensive technical specifications, engineering standards, and construction methodologies used in Polavaram project?"
                }
            ],
            "AmaravathiCapitalCity": [
                {
                    "heading": "Financial Planning",
                    "description": "Budget allocation and investment strategy",
                    "query": "What is the detailed budget allocation, funding strategy, and financial planning for Amaravathi capital city development?"
                },
                {
                    "heading": "Infrastructure Details",
                    "description": "Technical specifications and urban planning standards",
                    "query": "What are the technical specifications, infrastructure standards, and urban planning details for Amaravathi capital city?"
                }
            ]
        }
    }
    
    role_templates = default_templates.get(user_role, default_templates['public'])
    insights = role_templates.get(collection_name, [])[:num_insights]
    for i, insight in enumerate(insights):
        insight['id'] = f"{collection_name.lower()}_{user_role}_{i+1}"
        insight['collection'] = collection_name
        insight['role'] = user_role
    
    return insights

def retrieve_context(query, collection_name, top_k=5):
    """Retrieve relevant context from Weaviate with source information"""
    try:
        client = init_weaviate_client()
        collection = client.collections.get(collection_name)
        
        # Generate query embedding
        model = init_embedding_model()
        query_embedding = model.encode(query)
        
        # Perform vector search
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=["distance", "score"]
        )
        
        retrieved_docs = []
        if response.objects:
            for obj in response.objects:
                retrieved_docs.append({
                    'content': obj.properties['content'],
                    'page': obj.properties.get('page', 'N/A'),
                    'chunk_id': obj.properties.get('chunk_id', 'N/A'),
                    'distance': obj.metadata.distance if obj.metadata else 'N/A',
                    'collection': collection_name
                })
        
        return retrieved_docs
    except Exception as e:
        st.error(f"Error retrieving context: {e}")
        return []

def generate_answer(query, collection_name, user_role):
    """Generate answer using RAG with source tracking and role-based responses"""
    # Retrieve relevant context
    retrieved_docs = retrieve_context(query, collection_name, top_k=5)
    
    if not retrieved_docs:
        return "I couldn't find relevant information in the document to answer your question.", []
    
    # Prepare context and sources
    context = "\n\n".join([doc['content'] for doc in retrieved_docs])
    sources = retrieved_docs
    
    # Role-based system prompts
    role_prompts = {
        "public": """You are an assistant for general public users. Focus on:
- Clear, accessible language without technical jargon
- Public benefits and community impact
- General timelines and progress updates
- Environmental and social considerations
- Overall project value to society""",
        
        "govt": """You are an assistant for government officials and technical experts. Focus on:
- Detailed technical specifications and data
- Budget analysis and financial details
- Governance structures and compliance requirements
- Risk assessment and mitigation strategies
- Performance metrics and KPIs
- Strategic recommendations and decision support
- Quantitative data and specific figures"""
    }
    
    system_prompt = f"""You are an expert assistant.
{role_prompts.get(user_role, role_prompts['public'])}

Answer questions accurately and comprehensively based on the provided context.
ALWAYS format your answer as clear, numbered points (minimum 2 points, maximum 15 points).
Each point should be self-contained, informative, and address different aspects of the query.
Ensure the points collectively provide a complete answer to the user's question.
If the context doesn't contain enough information for at least 2 meaningful points, respond: "The requested information is not sufficiently available in the provided document to provide a comprehensive answer."
Make each point concise yet detailed enough to be valuable.
Maintain a professional, objective tone throughout."""

    user_message = f"""Context from the document:
{context}

Question: {query}

Please provide a comprehensive answer formatted as numbered points (2-15 points) that directly address the question.
Ensure each point is clear, valuable, and based on the context provided."""

    try:
        groq_client = init_groq_client()
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model=MODEL_NAME,
            temperature=0.7,
            max_tokens=1800,
        )
        
        return chat_completion.choices[0].message.content, sources
    except Exception as e:
        return f"Error generating answer: {e}", sources

def main():
    
    # Initialize session state
    if 'user_role' not in st.session_state:
        st.session_state.user_role = "public"
    if 'current_collection' not in st.session_state:
        st.session_state.current_collection = None
    if 'current_insight' not in st.session_state:
        st.session_state.current_insight = None
    if 'insights' not in st.session_state:
        st.session_state.insights = {}
    if 'current_messages' not in st.session_state:
        st.session_state.current_messages = []
    if 'sources' not in st.session_state:
        st.session_state.sources = []
    
    # Role selection in sidebar (without navigation buttons)
    with st.sidebar:
        st.title("AI-Driven Insights")
        
        # Role selection only
        st.subheader("Select Your Role")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("General Public", use_container_width=True,
                        type="primary" if st.session_state.user_role == "public" else "secondary",
                        key="insights_public_btn"):
                old_role = st.session_state.user_role
                st.session_state.user_role = "public"
                if old_role != st.session_state.user_role:
                    st.session_state.insights = {}
                    st.session_state.current_collection = None
                    st.session_state.current_insight = None
                    st.session_state.current_messages = []
                st.rerun()
        with col2:
            if st.button("Govt Official", use_container_width=True,
                        type="primary" if st.session_state.user_role == "govt" else "secondary",
                        key="insights_govt_btn"):
                old_role = st.session_state.user_role
                st.session_state.user_role = "govt"
                if old_role != st.session_state.user_role:
                    st.session_state.insights = {}
                    st.session_state.current_collection = None
                    st.session_state.current_insight = None
                    st.session_state.current_messages = []
                st.rerun()
        
        st.info(f"Current Role: {'General Public' if st.session_state.user_role == 'public' else 'Government Official'}")
        st.markdown("---")
        

        collections = ["PolavaramProject", "AmaravathiCapitalCity"]
        
        for collection in collections:
            st.subheader(collection)
            
            # Generate or load insights for this collection and role
            insight_key = f"{collection}_{st.session_state.user_role}"
            if insight_key not in st.session_state.insights:
                with st.spinner(f"Generating insights for {collection}..."):
                    st.session_state.insights[insight_key] = generate_ai_insights(
                        collection, st.session_state.user_role, num_insights=4
                    )
            
            insights = st.session_state.insights.get(insight_key, [])
            
            # Display insights as clickable cards
            for insight in insights:
                if st.button(
                    f"**{insight['heading']}**\n\n{insight['description']}",
                    key=insight['id'],
                    use_container_width=True
                ):
                    st.session_state.current_collection = collection
                    st.session_state.current_insight = insight
                    st.session_state.current_messages = []
                    
                    # Auto-ask the insight question
                    with st.spinner("Exploring this insight..."):
                        answer, sources = generate_answer(
                            insight['query'], 
                            collection, 
                            st.session_state.user_role
                        )
                    
                    st.session_state.current_messages = [
                        {"role": "user", "content": insight['query']},
                        {"role": "assistant", "content": answer, "sources": sources}
                    ]
                    st.rerun()
            
            # "More Insights" button
            if st.button(f"More Insights for {collection}", key=f"more_{insight_key}"):
                with st.spinner("Generating fresh insights..."):
                    st.session_state.insights[insight_key] = generate_ai_insights(
                        collection, st.session_state.user_role, num_insights=4
                    )
                st.rerun()
            
            st.markdown("---")
        
        st.markdown("*Insights are AI-generated based on document content*")
    
    # Main content area
    st.title("Andhra Pradesh AI Insights Chatbot")
    
    # Auto-initialize with a random insight if none selected
    if not st.session_state.current_insight:
        collections = ["PolavaramProject", "AmaravathiCapitalCity"]
        # Try to find any available insights
        for collection in collections:
            insight_key = f"{collection}_{st.session_state.user_role}"
            insights = st.session_state.insights.get(insight_key, [])
            if insights:
                # Select a random insight
                random_insight = random.choice(insights)
                st.session_state.current_collection = collection
                st.session_state.current_insight = random_insight
                st.session_state.current_messages = []
                
                # Auto-ask the insight question
                with st.spinner("Exploring a random insight..."):
                    answer, sources = generate_answer(
                        random_insight['query'], 
                        collection, 
                        st.session_state.user_role
                    )
                
                st.session_state.current_messages = [
                    {"role": "user", "content": random_insight['query']},
                    {"role": "assistant", "content": answer, "sources": sources}
                ]
                st.rerun()
                break
    
    # Display current insight chat
    if st.session_state.current_insight and st.session_state.current_collection:
        current_insight = st.session_state.current_insight
        current_collection = st.session_state.current_collection
        
        # Display current topic info
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"Exploring: {current_insight['heading']}")
            st.caption(f"Collection: {current_collection} • Role: {'General Public' if st.session_state.user_role == 'public' else 'Government Official'}")
            st.caption(current_insight['description'])
        with col2:
            if st.button("New Chat", use_container_width=True):
                st.session_state.current_messages = []
                st.rerun()
        
        st.markdown("---")
        
        # Display chat messages
        for message in st.session_state.current_messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message["content"])
                    
                    # Show sources in expander for assistant messages
                    if message.get("sources"):
                        with st.expander("View Retrieved Sources"):
                            for i, source in enumerate(message["sources"]):
                                st.markdown(f"**Source {i+1}:**")
                                st.markdown(f"**Page:** {source.get('page', 'N/A')}")
                                st.markdown(f"**Content Preview:** {source['content'][:200]}...")
                                st.markdown(f"**Collection:** {source.get('collection', 'N/A')}")
                                st.markdown("---")
        
        # Chat input for follow-up questions
        if prompt := st.chat_input("Ask a follow-up question about this insight..."):
            # Add user message
            st.session_state.current_messages.append({
                "role": "user", 
                "content": prompt
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching documents..."):
                    answer, sources = generate_answer(
                        prompt, 
                        current_collection, 
                        st.session_state.user_role
                    )
                
                st.markdown(answer)
                
                # Display sources in expander
                if sources:
                    with st.expander("View Retrieved Sources"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Source {i+1}:**")
                            st.markdown(f"**Page:** {source.get('page', 'N/A')}")
                            st.markdown(f"**Chunk ID:** {source.get('chunk_id', 'N/A')}")
                            st.markdown(f"**Collection:** {source.get('collection', 'N/A')}")
                            st.markdown(f"**Content Preview:** {source['content'][:200]}...")
                            st.markdown("---")
            
            # Add assistant message
            st.session_state.current_messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources
            })
            
            st.rerun()

if __name__ == "__main__":
    main()