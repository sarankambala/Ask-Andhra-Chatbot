import streamlit as st
import weaviate
from weaviate.classes.query import Filter
import pdfplumber
from sentence_transformers import SentenceTransformer
from groq import Groq
import time
import os
import json
import requests
from typing import List, Dict, Tuple, Optional
import uuid
from datetime import datetime

# Configuration
WEAVIATE_URL = "m8l0s1ucsfgj0pcvv4k4ra.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "d3pqMVVjeFZuNk9ZeXFTM19PYXdsVlBZdGQ3OXIzTGtvZHJQV3J6MHA2czZjdWcwcmJtQ3B1ZUd2VHhJPV92MjAw"
GROQ_API_KEY = "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"
MODEL_NAME = "llama-3.1-8b-instant"

# JSONBin Configuration
JSONBIN_MASTER_KEY = "$2a$10$Zvs9GnQFn58NfRB2viPmW.BTGobRxxEI6xuQYDFPxSe21lfZN9Gg."
JSONBIN_PUBLIC_CHATS_ID = "68f54d2bae596e708f1d8a9a"
JSONBIN_GOVT_CHATS_ID = "68f54f14ae596e708f1d8eec"
JSONBIN_BASE_URL = "https://api.jsonbin.io/v3/b"

# Initialize clients
@st.cache_resource
def init_weaviate_client():
    """Initialize Weaviate client with error handling"""
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY)
        )
        # Test connection
        client.collections.list_all()
        return client
    except Exception as e:
        st.error(f"Failed to connect to Weaviate: {e}")
        return None

@st.cache_resource
def init_embedding_model():
    """Initialize embedding model"""
    try:
        model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

@st.cache_resource
def init_groq_client():
    """Initialize Groq client"""
    try:
        return Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        return None

class ChatStorage:
    def __init__(self):
        self.master_key = JSONBIN_MASTER_KEY
        self.public_bin_id = JSONBIN_PUBLIC_CHATS_ID
        self.govt_bin_id = JSONBIN_GOVT_CHATS_ID
        self.base_url = JSONBIN_BASE_URL
        
    def _get_bin_id(self, user_role: str) -> str:
        """Get the appropriate bin ID based on user role"""
        return self.public_bin_id if user_role == "public" else self.govt_bin_id
    
    def _get_headers(self) -> Dict:
        """Get headers for JSONBin API"""
        return {
            "Content-Type": "application/json",
            "X-Master-Key": self.master_key
        }
    
    def load_chats(self, user_role: str) -> List[Dict]:
        """Load all chats for a specific role"""
        try:
            bin_id = self._get_bin_id(user_role)
            response = requests.get(f"{self.base_url}/{bin_id}/latest", headers=self._get_headers())
            
            if response.status_code == 200:
                data = response.json()
                chats = data.get('record', {}).get('chats', [])
                # Sort chats by updated_at in descending order (newest first)
                chats.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
                return chats
            else:
                st.error(f"Failed to load chats: {response.status_code}")
                return []
                
        except Exception as e:
            st.error(f"Error loading chats: {e}")
            return []
    
    def save_chats(self, user_role: str, chats: List[Dict]) -> bool:
        """Save chats for a specific role"""
        try:
            bin_id = self._get_bin_id(user_role)
            data = {"chats": chats}
            
            response = requests.put(f"{self.base_url}/{bin_id}", 
                                  headers=self._get_headers(), 
                                  json=data)
            
            if response.status_code == 200:
                return True
            else:
                st.error(f"Failed to save chats: {response.status_code}")
                return False
                
        except Exception as e:
            st.error(f"Error saving chats: {e}")
            return False
    
    def create_chat(self, user_role: str, chat_data: Dict) -> str:
        """Create a new chat and return its ID"""
        chats = self.load_chats(user_role)
        chat_id = str(uuid.uuid4())[:8]
        
        chat_data.update({
            "id": chat_id,
            "role": user_role,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": []
        })
        
        # Add new chat at the beginning of the list
        chats.insert(0, chat_data)
        if self.save_chats(user_role, chats):
            return chat_id
        return ""
    
    def update_chat(self, user_role: str, chat_id: str, messages: List[Dict], title: str = None) -> bool:
        """Update an existing chat with new messages"""
        chats = self.load_chats(user_role)
        
        for chat in chats:
            if chat["id"] == chat_id:
                chat["messages"] = messages
                chat["updated_at"] = datetime.now().isoformat()
                if title:
                    chat["title"] = title
                break
        
        # Re-sort chats by updated_at in descending order
        chats.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
        return self.save_chats(user_role, chats)
    
    def delete_chat(self, user_role: str, chat_id: str) -> bool:
        """Delete a chat"""
        chats = self.load_chats(user_role)
        chats = [chat for chat in chats if chat["id"] != chat_id]
        return self.save_chats(user_role, chats)
    
    def rename_chat(self, user_role: str, chat_id: str, new_title: str) -> bool:
        """Rename a chat"""
        chats = self.load_chats(user_role)
        
        for chat in chats:
            if chat["id"] == chat_id:
                chat["title"] = new_title
                chat["updated_at"] = datetime.now().isoformat()
                break
        
        # Re-sort chats by updated_at in descending order
        chats.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
        return self.save_chats(user_role, chats)

class EnhancedQueryRouter:
    def __init__(self):
        self.collections = ["PolavaramProject", "AmaravathiCapitalCity"]
        self.groq_client = init_groq_client()
        self.project_details = {
            "PolavaramProject": {
                "key_phases": ["Land Acquisition", "Dam Construction", "Canal Networks", "Rehabilitation"],
                "key_departments": ["Polavaram Project Authority", "Ministry of Jal Shakti", "Andhra Pradesh Water Resources"],
                "budget_components": ["Head Works", "Left Main Canal", "Right Main Canal", "Power House", "Rehabilitation"]
            },
            "AmaravathiCapitalCity": {
                "key_phases": ["Land Pooling", "Infrastructure Development", "Government Complex", "Urban Development"],
                "key_departments": ["CRDA", "Andhra Pradesh Capital Region Development"],
                "budget_components": ["Land Acquisition", "Infrastructure", "Government Buildings", "Utilities"]
            }
        }
        
    def keyword_based_routing(self, query: str) -> Optional[str]:
        """Route query based on keyword matching"""
        query_lower = query.lower()
        
        # Define comprehensive keywords for each collection
        keyword_mapping = {
            "PolavaramProject": [
                "polavaram", "dam", "irrigation", "godavari", "water resources", 
                "project", "reservoir", "canal", "hydropower", "andhra pradesh",
                "water supply", "construction", "engineering", "ppa", "polavaram project authority"
            ],
            "AmaravathiCapitalCity": [
                "amaravathi", "amaravati", "capital", "city", "development", 
                "infrastructure", "crda", "urban", "planning", "andhra pradesh",
                "capital city", "master plan", "construction", "development authority",
                "capital region", "amaravati capital"
            ]
        }
        
        # Calculate scores for each collection
        scores = {}
        for collection, keywords in keyword_mapping.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            scores[collection] = score
        
        # Return collection with highest score, if any score > 0
        max_score_collection = max(scores, key=scores.get)
        return max_score_collection if scores[max_score_collection] > 0 else None
    
    def model_based_routing(self, query: str) -> str:
        """Use LLM to route query to the most relevant collection"""
        if not self.groq_client:
            return self.keyword_based_routing(query) or self.collections[0]
            
        routing_prompt = f"""
        You are a query routing assistant. Classify the following user query into the most relevant document collection.
        
        Available Collections:
        1. PolavaramProject - Documents about Polavaram irrigation project, dam construction, water resources in Andhra Pradesh
        2. AmaravathiCapitalCity - Documents about Amaravathi capital city development, urban planning, infrastructure
        
        User Query: "{query}"
        
        Respond with ONLY the collection name: either "PolavaramProject" or "AmaravathiCapitalCity"
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": routing_prompt}],
                model=MODEL_NAME,
                temperature=0.1,
                max_tokens=50
            )
            
            selected_collection = response.choices[0].message.content.strip()
            
            # Validate and clean the response
            if "PolavaramProject" in selected_collection:
                return "PolavaramProject"
            elif "AmaravathiCapitalCity" in selected_collection:
                return "AmaravathiCapitalCity"
            else:
                return self.keyword_based_routing(query) or self.collections[0]
                
        except Exception as e:
            st.error(f"Model routing failed: {e}")
            return self.keyword_based_routing(query) or self.collections[0]
    
    def route_query(self, query: str) -> str:
        """Main routing function with fallback strategy"""
        # First try keyword-based routing
        collection = self.keyword_based_routing(query)
        
        # If keyword routing fails or is ambiguous, use model-based routing
        if not collection:
            collection = self.model_based_routing(query)
        
        return collection

class EnhancedQuestionSuggestor:
    def __init__(self):
        self.groq_client = init_groq_client()
    
    def generate_suggestions(self, user_question: str, assistant_response: str, current_collection: str, user_role: str) -> List[str]:
        """Generate relevant follow-up question suggestions using AI"""
        if user_role == "govt":
            return self._generate_government_suggestions(user_question, current_collection)
            
        if not self.groq_client:
            return self._get_fallback_suggestions(current_collection, user_role)
            
        suggestion_prompt = f"""
        Based on the following conversation context, generate 3-4 relevant follow-up questions that the user might ask next.
        
        User's Question: "{user_question}"
        Assistant's Response: "{assistant_response}"
        Current Topic: {current_collection}
        User Role: {user_role}
        
        Generate diverse, specific follow-up questions that:
        1. Dive deeper into the topic discussed
        2. Ask for comparisons with related topics
        3. Request appropriate details
        4. Explore related aspects
        
        Return ONLY the questions as a numbered list, nothing else.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": suggestion_prompt}],
                model=MODEL_NAME,
                temperature=0.8,
                max_tokens=300,
            )
            
            suggestions_text = response.choices[0].message.content.strip()
            # Parse the numbered list into individual questions
            suggestions = []
            for line in suggestions_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering/bullets and clean the question
                    question = line.split('.', 1)[-1].split(')', 1)[-1].strip()
                    question = question.lstrip('-• ').strip()
                    if question and len(question) > 10:  # Ensure it's a meaningful question
                        suggestions.append(question)
            
            return suggestions[:4] if suggestions else self._get_fallback_suggestions(current_collection, user_role)
            
        except Exception as e:
            return self._get_fallback_suggestions(current_collection, user_role)
    
    def _generate_government_suggestions(self, question: str, collection: str) -> List[str]:
        """Generate government-specific follow-up questions"""
        base_suggestions = [
            f"Provide detailed budget allocation for {collection}",
            f"What are the current challenges facing {collection}?",
            f"Show me the environmental compliance status for {collection}",
            f"Provide stakeholder analysis and engagement plan",
            f"What are the risk mitigation strategies in place?",
            f"Show me the procurement and contracting details",
            f"Provide the monitoring and evaluation framework",
            f"What are the key performance indicators for {collection}?",
            f"Provide detailed timeline with milestones and deadlines",
            f"What are the technical specifications and engineering standards?"
        ]
        
        # Add collection-specific suggestions
        if collection == "PolavaramProject":
            base_suggestions.extend([
                "Provide dam safety and structural integrity reports",
                "Show me the water allocation and distribution plan",
                "What is the rehabilitation and resettlement status?",
                "Provide environmental impact assessment details"
            ])
        elif collection == "AmaravathiCapitalCity":
            base_suggestions.extend([
                "Provide urban planning and zoning details",
                "Show me the infrastructure development timeline",
                "What is the land acquisition and compensation status?",
                "Provide master plan implementation progress"
            ])
        
        return base_suggestions[:4]
    
    def _get_fallback_suggestions(self, collection: str, user_role: str) -> List[str]:
        """Provide fallback suggestions if AI generation fails"""
        if collection == "PolavaramProject":
            if user_role == "govt":
                return [
                    "What is the detailed budget breakdown for Polavaram?",
                    "What are the specific technical specifications?",
                    "Provide detailed environmental impact assessment",
                    "What is the current project status with timelines?"
                ]
            else:
                return [
                    "What is the current status of Polavaram project?",
                    "How will Polavaram benefit local communities?",
                    "What are the environmental considerations?",
                    "When is the expected completion date?"
                ]
        elif collection == "AmaravathiCapitalCity":
            if user_role == "govt":
                return [
                    "What is the detailed infrastructure budget?",
                    "Provide master plan technical specifications",
                    "What are the land acquisition details?",
                    "Give detailed project timeline with milestones"
                ]
            else:
                return [
                    "What is the current status of Amaravathi development?",
                    "How will Amaravathi benefit residents?",
                    "What are the main features of the capital city?",
                    "What is the expected development timeline?"
                ]
        else:
            if user_role == "govt":
                return [
                    "Compare detailed budgets of both projects",
                    "Provide technical specifications comparison",
                    "Detailed environmental impact analysis",
                    "Project status and timeline comparison"
                ]
            else:
                return [
                    "Compare both projects in simple terms",
                    "Which project will benefit people more?",
                    "What are the main differences?",
                    "Which one is progressing faster?"
                ]

class EnhancedMultiCollectionRAG:
    def __init__(self):
        self.weaviate_client = init_weaviate_client()
        self.groq_client = init_groq_client()
        self.embedding_model = init_embedding_model()
        self.router = EnhancedQueryRouter()
        self.suggestor = EnhancedQuestionSuggestor()
        self.storage = ChatStorage()
        
    def search_single_collection(self, query: str, collection_name: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
        """Search in a single collection with enhanced source tracking"""
        if not self.weaviate_client or not self.embedding_model:
            return "", []
            
        try:
            collection = self.weaviate_client.collections.get(collection_name)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Perform vector search
            response = collection.query.near_vector(
                near_vector=query_embedding,
                limit=top_k,
                return_metadata=["distance", "score"]
            )
            
            retrieved_docs = []
            if response.objects:
                for obj in response.objects:
                    # Handle different possible field names
                    content = obj.properties.get('content') or obj.properties.get('text') or ""
                    retrieved_docs.append({
                        'content': content,
                        'page': obj.properties.get('page', 'N/A'),
                        'chunk_id': obj.properties.get('chunk_id', 'N/A'),
                        'distance': obj.metadata.distance if obj.metadata else 'N/A',
                        'confidence': 1 - (obj.metadata.distance if obj.metadata else 0.5),
                        'collection': collection_name,
                        'document_type': obj.properties.get('document_type', 'Technical Document'),
                        'source': obj.properties.get('source', 'Project Documentation')
                    })
            
            context = "\n\n".join([doc['content'] for doc in retrieved_docs])
            return context, retrieved_docs
            
        except Exception as e:
            st.error(f"Error searching collection {collection_name}: {e}")
            return "", []
    
    def search_all_collections(self, query: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
        """Search across all collections"""
        all_context = []
        all_sources = []
        
        for collection in self.router.collections:
            context, sources = self.search_single_collection(query, collection, top_k)
            if context:
                all_context.append(f"=== {collection} ===\n{context}")
                all_sources.extend(sources)
        
        return "\n\n".join(all_context), all_sources
    
    def _get_role_based_system_prompt(self, user_role: str, query_type: str = "general") -> str:
        """Enhanced system prompt based on user role"""
        if user_role == "govt":
            if query_type == "comparison":
                return """As a government technical advisor, provide COMPREHENSIVE comparisons with:

ESSENTIAL COMPARISON ELEMENTS:
1. Budget Analysis - Detailed financial breakdowns, cost comparisons, funding sources
2. Technical Specifications - Engineering details, design parameters, technical requirements
3. Timeline Analysis - Project schedules, milestones, completion status
4. Governance Structures - Oversight mechanisms, responsible agencies, accountability
5. Risk Assessment - Identified risks, mitigation strategies, contingency plans
6. Performance Metrics - KPIs, success indicators, progress measurements
7. Stakeholder Impact - Affected communities, economic impact, social considerations
8. Regulatory Compliance - Environmental clearances, legal requirements, compliance status

RESPONSE REQUIREMENTS:
- Use exact figures and quantitative data
- Provide actionable insights for decision-making
- Include specific recommendations
- Structure with clear comparative analysis
- Never state "information not available" - provide context and general standards
- Focus on strategic implications

Format with clear section headers and data-rich content."""
            elif query_type == "summary":
                return """As a government technical advisor, provide COMPREHENSIVE summaries including:

MANDATORY SECTIONS:
1. Executive Overview - Key facts, current status, strategic importance
2. Financial Analysis - Budget allocation, expenditures, funding status
3. Technical Overview - Specifications, engineering details, design parameters
4. Timeline Summary - Current progress, key milestones, critical path
5. Governance Framework - Responsible agencies, oversight mechanisms
6. Risk Profile - Major risks, mitigation measures, challenges
7. Performance Assessment - Achievements, delays, key metrics
8. Strategic Recommendations - Action items, priorities, next steps

RESPONSE REQUIREMENTS:
- Include quantitative data and specific metrics
- Provide complete information for each section
- Structure with clear headings and bullet points
- Focus on decision-support information
- Never omit sections - provide best available information

Format with professional government reporting standards."""
            else:
                return """As a government technical advisor, provide COMPLETE, DETAILED answers with:

MANDATORY SECTIONS for project queries:
1. PROJECT OVERVIEW - Purpose, scope, strategic importance, current status
2. FINANCIAL DETAILS - Complete budget breakdown, expenditures, funding sources, cost analysis
3. TECHNICAL SPECIFICATIONS - Engineering details, design parameters, technical requirements
4. TIMELINE & MILESTONES - Current progress, deadlines, delays, critical path analysis
5. STAKEHOLDERS & GOVERNANCE - Responsible agencies, oversight bodies, decision-making processes
6. ENVIRONMENTAL & SOCIAL IMPACT - Assessments, mitigation measures, compliance status
7. RISKS & CHALLENGES - Identified risks, mitigation strategies, contingency plans
8. BENEFITS & OUTCOMES - Expected benefits, performance metrics, success indicators

CRITICAL REQUIREMENTS:
- ALWAYS provide quantitative data and specific figures
- If exact data isn't in context, provide typical industry standards/ranges
- Use technical terminology appropriate for government professionals
- Include actionable recommendations and strategic insights
- Structure with clear headings and numbered points
- NEVER state "not mentioned" or "not available" - always provide context
- Focus on decision-support and actionable information

Format with professional government reporting standards."""
        else:
            if query_type == "comparison":
                return """Provide clear, accessible comparisons focusing on:
- High-level benefits and impacts
- General timelines without specific dates
- Public benefits and community impacts
- Environmental considerations in general terms
- Overall progress and general information

Focus on publicly available information. Format as clear, numbered points."""
            elif query_type == "summary":
                return """Provide general summaries suitable for public understanding. Include:
- Overall project goals and public benefits
- General progress updates
- Community impacts and benefits
- Environmental considerations
- Publicly available information

Format as clear, numbered points."""
            else:
                return """Provide accurate information focusing on public benefits and community impacts. 
- Use general timelines without specific dates
- Focus on overall progress and benefits
- Maintain transparency while being appropriate for public consumption
- Structure information clearly for general understanding

Format as clear, numbered points."""
    
    def handle_comparison_query(self, query: str, user_role: str) -> Tuple[str, List[Dict]]:
        """Handle comparison queries between collections"""
        comparison_context = ""
        all_sources = []
        
        # For government officials, search more comprehensively
        top_k = 6 if user_role == "govt" else 3
        
        # Get data from both collections
        for collection in self.router.collections:
            context, sources = self.search_single_collection(query, collection, top_k=top_k)
            if context:
                comparison_context += f"\n\n=== {collection} ===\n{context}"
                all_sources.extend(sources)
        
        if not comparison_context:
            return "Based on the available information, I'm unable to provide a comprehensive comparison at this time.", []
        
        system_prompt = self._get_role_based_system_prompt(user_role, "comparison")
        
        comparison_prompt = f"""Based on the following data from different collections, provide a comprehensive comparison addressing: {query}

Data from collections:
{comparison_context}

Please provide a detailed comparative analysis with specific data points and actionable insights."""
        
        answer = self._generate_answer_with_prompt(comparison_prompt, system_prompt)
        return answer, all_sources
    
    def handle_summary_query(self, query: str, user_role: str) -> Tuple[str, List[Dict]]:
        """Handle summary queries across all collections"""
        # For government officials, search more comprehensively
        top_k = 5 if user_role == "govt" else 3
        all_context, all_sources = self.search_all_collections(query, top_k=top_k)
        
        if not all_context:
            return "Based on the available documents, I'm unable to generate a comprehensive summary for this query.", []
        
        system_prompt = self._get_role_based_system_prompt(user_role, "summary")
        
        summary_prompt = f"""Based on all available documents, provide a comprehensive summary addressing: {query}

Available documents:
{all_context}

Please provide a thorough summary with complete information across all relevant aspects."""
        
        answer = self._generate_answer_with_prompt(summary_prompt, system_prompt)
        return answer, all_sources
    
    def _generate_answer_with_prompt(self, user_prompt: str, system_prompt: str) -> str:
        """Generate answer using custom prompt"""
        if not self.groq_client:
            return "I'm currently unable to process your request. Please try again later."
        
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=MODEL_NAME,
                temperature=0.7,
                max_tokens=2500,
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            return "I encountered an error while processing your request. Please try again."
    
    def validate_government_response(self, answer: str, sources: List[Dict]) -> Dict:
        """Validate that government responses contain sufficient detail"""
        validation_criteria = {
            "has_figures": any(char.isdigit() for char in answer),
            "has_dates": any(word in answer.lower() for word in ['202', 'date', 'timeline', 'schedule']),
            "has_specifics": len([s for s in sources if s['confidence'] > 0.7]) >= 2,
            "has_structure": answer.count('\n') >= 8,  # Good structure for government responses
            "has_technical_terms": any(term in answer.lower() for term in ['budget', 'technical', 'specification', 'timeline', 'risk', 'governance'])
        }
        
        score = sum(validation_criteria.values()) / len(validation_criteria)
        
        return {
            "score": score,
            "meets_standard": score >= 0.7,
            "details": validation_criteria
        }
    
    def generate_answer(self, query: str, context: str, sources: List[Dict], user_role: str) -> str:
        """Generate answer for single collection queries"""
        system_prompt = self._get_role_based_system_prompt(user_role, "general")
        
        user_message = f"""Context from relevant documents:
{context}

Question: {query}

Please provide a comprehensive answer that directly addresses the question with complete information."""
        
        answer = self._generate_answer_with_prompt(user_message, system_prompt)
        
        # Validate government responses for quality
        if user_role == "govt":
            validation = self.validate_government_response(answer, sources)
            if not validation["meets_standard"]:
                # Enhance the answer if it doesn't meet standards
                enhancement_prompt = f"""The following answer needs enhancement for government standards. Please improve it with more specific data, structure, and completeness:

Original Question: {query}
Original Answer: {answer}

Please provide an enhanced version with:
- More quantitative data and specific figures
- Clearer structure with sections
- More technical details
- Actionable recommendations"""
                
                enhanced_answer = self._generate_answer_with_prompt(enhancement_prompt, system_prompt)
                if len(enhanced_answer) > len(answer):
                    answer = enhanced_answer
        
        return answer
    
    def process_query(self, query: str, user_role: str) -> Tuple[str, str, List[Dict]]:
        """Enhanced main function to process user queries"""
        query_lower = query.lower()
        
        # Determine search parameters based on role
        search_top_k = 8 if user_role == "govt" else 4
        
        # Check for specific information types
        info_requirements = []
        if any(word in query_lower for word in ['budget', 'cost', 'financial', 'funding', 'allocation']):
            info_requirements.append("financial details and budget breakdown")
        if any(word in query_lower for word in ['timeline', 'schedule', 'progress', 'completion', 'deadline']):
            info_requirements.append("project timeline and current progress")
        if any(word in query_lower for word in ['technical', 'specification', 'engineering', 'design']):
            info_requirements.append("technical specifications")
        if any(word in query_lower for word in ['environment', 'impact', 'ecological', 'assessment']):
            info_requirements.append("environmental impact assessment")
        
        # Check for multi-collection operations
        if any(keyword in query_lower for keyword in [
            "compare", "comparison", "difference between", "similarities", 
            "contrast", "versus", "vs", "both", "each"
        ]):
            selected_collection = "multiple"
            answer, sources = self.handle_comparison_query(query, user_role)
            
        elif any(keyword in query_lower for keyword in [
            "summarize all", "all documents", "both collections", "overall summary",
            "everything", "all information", "complete overview", "comprehensive summary"
        ]):
            selected_collection = "all"
            answer, sources = self.handle_summary_query(query, user_role)
            
        else:
            # Route to specific collection
            selected_collection = self.router.route_query(query)
            context, sources = self.search_single_collection(query, selected_collection, top_k=search_top_k)
            answer = self.generate_answer(query, context, sources, user_role)
        
        return answer, selected_collection, sources

def main():
    st.title("Andhra Pradesh RAG Intelligence Hub")
    st.markdown("Ask Andhra – Chat with documents about **Polavaram Project** and **Amaravathi Capital City**")
    
    # Initialize RAG system
    if "rag_system" not in st.session_state:
        with st.spinner("Initializing Enhanced RAG system..."):
            st.session_state.rag_system = EnhancedMultiCollectionRAG()
    
    # Initialize session state (remove role selection and chat management from here)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_suggestions" not in st.session_state:
        st.session_state.current_suggestions = []
    if "last_user_question" not in st.session_state:
        st.session_state.last_user_question = ""
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "user_role" not in st.session_state:
        st.session_state.user_role = "public"
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "chats" not in st.session_state:
        st.session_state.chats = []
    if "chat_search" not in st.session_state:
        st.session_state.chat_search = ""
    
    # Load chats for current role
    if not st.session_state.chats:
        st.session_state.chats = st.session_state.rag_system.storage.load_chats(st.session_state.user_role)
    
    # Enhanced sidebar content for RAG chat (without navigation buttons)
    with st.sidebar:
        st.header("Select Your Role")
        
        # Role selection buttons only
        col1, col2 = st.columns(2)
        with col1:
            if st.button("General Public", use_container_width=True, 
                        type="primary" if st.session_state.user_role == "public" else "secondary",
                        key="rag_public_btn"):
                old_role = st.session_state.user_role
                st.session_state.user_role = "public"
                if old_role != st.session_state.user_role:
                    st.session_state.messages = []
                    st.session_state.current_suggestions = []
                    st.session_state.current_chat_id = None
                    st.session_state.chats = st.session_state.rag_system.storage.load_chats(st.session_state.user_role)
                st.rerun()
        with col2:
            if st.button("Govt Official", use_container_width=True,
                        type="primary" if st.session_state.user_role == "govt" else "secondary",
                        key="rag_govt_btn"):
                old_role = st.session_state.user_role
                st.session_state.user_role = "govt"
                if old_role != st.session_state.user_role:
                    st.session_state.messages = []
                    st.session_state.current_suggestions = []
                    st.session_state.current_chat_id = None
                    st.session_state.chats = st.session_state.rag_system.storage.load_chats(st.session_state.user_role)
                st.rerun()
        
        # Show current role with enhanced description
        if st.session_state.user_role == "public":
            role_display = "General Public"
            role_info = "Access general project information and public benefits"
        else:
            role_display = "Government Official"
            role_info = "Access detailed technical data, budgets, and strategic insights"
        
        st.info(f"**Current Role:** {role_display}\n\n{role_info}")
        
        st.markdown("---")
        
        # Chat management controls
        if st.button("New Chat ➕", key="rag_new_chat"):
            st.session_state.messages = []
            st.session_state.current_suggestions = []
            st.session_state.current_chat_id = None
            st.rerun()
            
        if st.button("Clear Chat 🗑️", key="rag_clear_chat"):
            st.session_state.messages = []
            st.session_state.current_suggestions = []
            st.session_state.current_chat_id = None
            st.rerun()
        
        st.markdown("---")
        st.header("Chat History")
        
        # Chat search
        st.session_state.chat_search = st.text_input("Search chats...", value=st.session_state.chat_search, key="rag_chat_search")
        
        # Filter and display chats
        filtered_chats = st.session_state.chats
        if st.session_state.chat_search:
            filtered_chats = [chat for chat in filtered_chats 
                            if st.session_state.chat_search.lower() in chat.get('title', '').lower()]
        
        if filtered_chats:
            for chat in filtered_chats:
                col1, col2 = st.columns([3, 1])
                with col1:
                    chat_title = chat.get('title', f"Chat {chat['id']}")
                    if st.button(f"{chat_title}", key=f"rag_load_{chat['id']}", use_container_width=True):
                        st.session_state.current_chat_id = chat['id']
                        st.session_state.messages = chat.get('messages', [])
                        st.session_state.current_suggestions = []
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"rag_delete_{chat['id']}"):
                        if st.session_state.rag_system.storage.delete_chat(st.session_state.user_role, chat['id']):
                            st.session_state.chats = st.session_state.rag_system.storage.load_chats(st.session_state.user_role)
                            if st.session_state.current_chat_id == chat['id']:
                                st.session_state.current_chat_id = None
                                st.session_state.messages = []
                            st.rerun()
            
            # Rename current chat
            if st.session_state.current_chat_id:
                current_chat = next((chat for chat in st.session_state.chats if chat['id'] == st.session_state.current_chat_id), None)
                if current_chat:
                    new_title = st.text_input("Rename chat:", value=current_chat.get('title', f"Chat {current_chat['id']}"), key="rag_rename")
                    if st.button("Update Name", key="rag_update_name") and new_title != current_chat.get('title'):
                        if st.session_state.rag_system.storage.rename_chat(st.session_state.user_role, current_chat['id'], new_title):
                            st.session_state.chats = st.session_state.rag_system.storage.load_chats(st.session_state.user_role)
                            st.rerun()
        else:
            st.info("No chats found. Start a new conversation to save chats.")
        
        st.markdown("---")

        

    
    # Check if there's a pending question from suggestion click
    if st.session_state.pending_question:
        prompt = st.session_state.pending_question
        st.session_state.pending_question = None
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.last_user_question = prompt
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching across collections..."):
                answer, selected_collection, sources = st.session_state.rag_system.process_query(
                    prompt, st.session_state.user_role
                )
                
                st.markdown(answer)
                st.caption(f"Source: {selected_collection}")
                
                # Display enhanced source information for government users
                if sources:
                    with st.expander("View Retrieved Sources"):
                        for i, source in enumerate(sources):
                            confidence_color = "🟢" if source.get('confidence', 0) > 0.7 else "🟡" if source.get('confidence', 0) > 0.5 else "🔴"
                            st.markdown(f"**Source {i+1}** {confidence_color} (Page {source.get('page', 'N/A')})")
                            st.markdown(f"*{source['content'][:250]}...*")
                            st.markdown(f"Collection: `{source.get('collection', 'N/A')}` | "
                                      f"Confidence: `{source.get('confidence', 0):.2f}` | "
                                      f"Type: `{source.get('document_type', 'N/A')}`")
                            st.markdown("---")
        
        # Add assistant response to chat history
        assistant_message = {
            "role": "assistant", 
            "content": answer,
            "collection": selected_collection,
            "sources": sources
        }
        st.session_state.messages.append(assistant_message)
        
        # Save chat
        if st.session_state.current_chat_id:
            # Update existing chat
            st.session_state.rag_system.storage.update_chat(
                st.session_state.user_role, 
                st.session_state.current_chat_id, 
                st.session_state.messages
            )
        else:
            # Create new chat
            chat_title = prompt[:30] + "..." if len(prompt) > 30 else prompt
            chat_id = st.session_state.rag_system.storage.create_chat(
                st.session_state.user_role,
                {
                    "title": chat_title,
                    "collection": selected_collection,
                    "question": prompt,
                    "answer": answer
                }
            )
            if chat_id:
                st.session_state.current_chat_id = chat_id
                st.session_state.rag_system.storage.update_chat(
                    st.session_state.user_role, 
                    chat_id, 
                    st.session_state.messages
                )
                st.session_state.chats = st.session_state.rag_system.storage.load_chats(st.session_state.user_role)
        
        # Generate new question suggestions
        with st.spinner("Generating follow-up questions..."):
            st.session_state.current_suggestions = st.session_state.rag_system.suggestor.generate_suggestions(
                st.session_state.last_user_question,
                answer,
                selected_collection,
                st.session_state.user_role
            )
        
        st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("collection"):
                st.caption(f"Source: {message['collection']}")
            if message.get("sources"):
                with st.expander("📂 View Sources"):
                    for i, source in enumerate(message["sources"]):
                        confidence_color = "🟢" if source.get('confidence', 0) > 0.7 else "🟡" if source.get('confidence', 0) > 0.5 else "🔴"
                        st.markdown(f"**Source {i+1}** {confidence_color} (Page {source.get('page', 'N/A')})")
                        st.markdown(f"*{source['content'][:200]}...*")
                        st.markdown(f"Collection: `{source.get('collection', 'N/A')}` | "
                                  f"Confidence: `{source.get('confidence', 0):.2f}`")
                        st.markdown("---")
    
    # Display enhanced question suggestions if available
    if st.session_state.current_suggestions:
        st.markdown("---")
        st.subheader("Suggested Follow-up Questions")
        cols = st.columns(2)
        for idx, suggestion in enumerate(st.session_state.current_suggestions):
            with cols[idx % 2]:
                if st.button(
                    suggestion,
                    key=f"suggestion_{idx}",
                    use_container_width=True,
                    help="Click to ask this follow-up question"
                ):
                    # Set the pending question and trigger processing
                    st.session_state.pending_question = suggestion
                    st.session_state.current_suggestions = []  # Clear suggestions temporarily
                    st.rerun()
    
    # Enhanced chat input with role-based placeholder
    if st.session_state.user_role == "govt":
        placeholder = "Ask detailed questions about budgets, technical specs, timelines..."
    else:
        placeholder = "Ask about project benefits, progress, or compare both projects..."
    
    if prompt := st.chat_input(placeholder):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.last_user_question = prompt
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching across collections..."):
                answer, selected_collection, sources = st.session_state.rag_system.process_query(
                    prompt, st.session_state.user_role
                )
                
                st.markdown(answer)
                st.caption(f"Source: {selected_collection}")
                
                # Display enhanced source information
                if sources:
                    with st.expander("View Retrieved Sources"):
                        for i, source in enumerate(sources):
                            confidence_color = "🟢" if source.get('confidence', 0) > 0.7 else "🟡" if source.get('confidence', 0) > 0.5 else "🔴"
                            st.markdown(f"**Source {i+1}** {confidence_color} (Page {source.get('page', 'N/A')})")
                            st.markdown(f"*{source['content'][:250]}...*")
                            st.markdown(f"Collection: `{source.get('collection', 'N/A')}` | "
                                      f"Confidence: `{source.get('confidence', 0):.2f}` | "
                                      f"Type: `{source.get('document_type', 'N/A')}`")
                            st.markdown("---")
        
        # Add assistant response to chat history
        assistant_message = {
            "role": "assistant", 
            "content": answer,
            "collection": selected_collection,
            "sources": sources
        }
        st.session_state.messages.append(assistant_message)
        
        # Save chat
        if st.session_state.current_chat_id:
            # Update existing chat
            st.session_state.rag_system.storage.update_chat(
                st.session_state.user_role, 
                st.session_state.current_chat_id, 
                st.session_state.messages
            )
        else:
            # Create new chat
            chat_title = prompt[:30] + "..." if len(prompt) > 30 else prompt
            chat_id = st.session_state.rag_system.storage.create_chat(
                st.session_state.user_role,
                {
                    "title": chat_title,
                    "collection": selected_collection,
                    "question": prompt,
                    "answer": answer
                }
            )
            if chat_id:
                st.session_state.current_chat_id = chat_id
                st.session_state.rag_system.storage.update_chat(
                    st.session_state.user_role, 
                    chat_id, 
                    st.session_state.messages
                )
                st.session_state.chats = st.session_state.rag_system.storage.load_chats(st.session_state.user_role)
        
        # Generate new question suggestions
        with st.spinner("Generating follow-up questions..."):
            st.session_state.current_suggestions = st.session_state.rag_system.suggestor.generate_suggestions(
                st.session_state.last_user_question,
                answer,
                selected_collection,
                st.session_state.user_role
            )
        
        st.rerun()

if __name__ == "__main__":
    main()