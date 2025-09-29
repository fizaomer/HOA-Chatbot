import os
import json
from typing import List, Dict, Optional
import numpy as np
from dotenv import load_dotenv
import streamlit as st

# OpenAI for AI responses
import openai

# Load config from .env file
load_dotenv()

class HOAChatbot:
    def __init__(self, data_file: str = "data/processed_documents.json"):
        self.data_file = data_file
        self.documents = self.load_documents()
        self.embedding_model = None
        self.document_embeddings = []
        
        # Set up OpenAI - our AI engine
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        print("✓ OpenAI client initialized")
            
        # Load the text embedding model for finding relevant docs
        self._load_embedding_model()
        
        # Build search index from all the document sections
        if self.embedding_model:
            self.document_embeddings = self.create_embeddings()
    
    
    def _load_embedding_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            print("Text embedding model loaded")
        except Exception as e:
            print(f"Couldn't load embedding model: {e}")
            self.embedding_model = None
    
    def load_documents(self) -> Dict:
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Data file {self.data_file} not found. Run pdf_processor.py first.")
            return {}
    
    def get_all_pdf_files(self) -> List[str]:
        """Find all the PDF files in the current folder."""
        pdf_files = []
        for file in os.listdir('.'):
            if file.endswith('.pdf'):
                pdf_files.append(file)
        return pdf_files
    
    def get_document_status(self) -> Dict:
        """Check which documents are available in the processed data."""
        status = {}
        
        # Show documents that are actually in the processed data
        for doc_name in self.documents.keys():
            doc_data = self.documents[doc_name]
            status[doc_name] = {
                'status': 'processed',
                'sections': len(doc_data['sections']),
                'total_chars': len(doc_data['cleaned_text']),
                'raw_length': len(doc_data['raw_text'])
            }
        
        return status
    
    def create_embeddings(self) -> List[Dict]:
        """Create embeddings for all document sections."""
        embeddings = []
        for doc_name, doc_data in self.documents.items():
            for i, section in enumerate(doc_data['sections']):
                try:
                    embedding = self.embedding_model.encode(section)
                    embeddings.append({
                        'document': doc_name,
                        'section_index': i,
                        'text': section,
                        'embedding': embedding
                    })
                except Exception as e:
                    print(f"Error creating embedding for {doc_name} section {i}: {e}")
        return embeddings
    
    def find_relevant_sections(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.document_embeddings or not self.embedding_model:
            fallback_sections = []
            for doc_name, doc_data in self.documents.items():
                for i, section in enumerate(doc_data['sections'][:top_k]):
                    fallback_sections.append({
                        'document': doc_name,
                        'section_index': i,
                        'text': section,
                        'embedding': None
                    })
            return fallback_sections
        try:
            query_embedding = self.embedding_model.encode(query)
            similarities = []
            for doc_embedding in self.document_embeddings:
                similarity = np.dot(query_embedding, doc_embedding['embedding']) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding['embedding'])
                )
                similarities.append((similarity, doc_embedding))
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [item[1] for item in similarities[:top_k]]
        except Exception as e:
            print(f"Semantic search failed: {e}")
            return self.find_relevant_sections(query, top_k)
    
    def generate_response_with_openai(self, query: str, relevant_sections: List[Dict]) -> str:
        if not relevant_sections:
            return "I don't have enough information to answer that question. Please try rephrasing or ask about a different topic."
        
        context = "\n\n".join([
            f"From {section['document']}:\n{section['text']}"
            for section in relevant_sections
        ])
        
        prompt = f"""You are a helpful HOA assistant. Use the information below from HOA documents to answer the user's question. If you can't find the answer in what's provided, just say so.

Context from HOA documents:
{context}

User question: {query}

Provide a clear, helpful answer based on the HOA documents. If you mention specific rules, tell them which document it's from."""

        try:
            print("🤖 Using OpenAI GPT-3.5-turbo for AI response...")
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful HOA assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            print("✅ OpenAI response received successfully")
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ OpenAI failed: {str(e)}")
            return f"Error calling OpenAI: {str(e)}"
    
    def generate_response_with_search(self, query: str, relevant_sections: List[Dict]) -> str:
        print("🔍 Using search-only mode (no AI)")
        if not relevant_sections:
            return "I don't have enough information to answer that question. Please try rephrasing or ask about a different topic."
        response_parts = []
        response_parts.append(f"Here's what I found about '{query}' in your HOA docs:")
        for i, section in enumerate(relevant_sections, 1):
            section_text = section['text']
            sentences = section_text.split('. ')
            summary = '. '.join(sentences[:3]) + '.' if len(sentences) > 3 else section_text
            response_parts.append(f"\n**From {section['document']}:**")
            response_parts.append(summary)
        response_parts.append(f"\n\n*This info comes from searching your HOA docs. Check the original documents for full details.*")
        return '\n'.join(response_parts)
    
    def answer_question(self, query: str, use_ai: str = "openai") -> Dict:
        relevant_sections = self.find_relevant_sections(query)
        
        # Use OpenAI by default, fall back to search if needed
        if use_ai == "openai":
            try:
                response = self.generate_response_with_openai(query, relevant_sections)
                method_used = "🤖 OpenAI GPT-3.5-turbo"
            except Exception as e:
                print(f"OpenAI failed, using search mode: {e}")
                response = self.generate_response_with_search(query, relevant_sections)
                method_used = "🔍 Search-only (OpenAI failed)"
        else:
            response = self.generate_response_with_search(query, relevant_sections)
            method_used = "🔍 Search-only"
        
        return {
            'question': query,
            'answer': response,
            'method_used': method_used,
            'sources': [
                {
                    'document': section['document'],
                    'section_index': section['section_index'],
                    'text_preview': section['text'][:800] + "..." if len(section['text']) > 800 else section['text']
                }
                for section in relevant_sections
            ]
        }
    
    def get_available_documents(self) -> List[str]:
        return list(self.documents.keys())
    
    def get_document_stats(self) -> Dict:
        stats = {}
        for doc_name, doc_data in self.documents.items():
            stats[doc_name] = {
                'sections': len(doc_data['sections']),
                'total_chars': len(doc_data['cleaned_text']),
                'file_size_mb': len(doc_data['raw_text']) / (1024 * 1024)
            }
        return stats
    
    def get_display_name(self, doc_name: str) -> str:
        """Get a clean display name for a document."""
        # Document name mapping for better display
        name_mapping = {
            'Amended and Restated Bylaws - Signed (1).pdf': 'HOA Bylaws',
            'Turtle Rock Crest Delinquency Policy Adopted 62023.pdf': 'Delinquency Policy',
            'Architectural Application Process.pdf': 'Architectural Process',
            'Amended and Restated CC&Rs - Recorded 5-1-2025 (1).pdf': 'CC&Rs (Covenants)',
            'TRCMA Revised Responsibility Matrix.pdf': 'Responsibility Matrix',
            '2020 Election Rules.pdf': 'Election Rules',
            'TRCMA Updated Parking Rules.pdf': 'Parking Rules',
            'TRCMA 2022 Paint Color.pdf': 'Paint Colors',
            'TRCMA 2022 Paint Color Matrix FINAL rev2 1 30 2023.pdf': 'Paint Color Matrix',
            'TRCMA 2022 Paint Color Matrix FINAL rev2 1 30 2023': 'Paint Color Matrix',
            'Architectural Application Turtle Rock Crest.pdf': 'Architectural Application',
            'Architectural Application Turtle Rock Crest': 'Architectural Application',
            'TRCMA Deck Maintenance Tips rev9 5 2022 FINAL.pdf': 'Deck Maintenance Tips',
            'TRCMA Deck Maintenance Tips rev9 5 2022 FINAL': 'Deck Maintenance Tips',
            'Turtle Rock Crest Enforcement and Fine Policy.pdf': 'Enforcement & Fine Policy',
            'Turtle Rock Crest Enforcement and Fine Policy': 'Enforcement & Fine Policy',
            'TimberTech Care Cleaning Guide.pdf': 'TimberTech Cleaning Guide',
            'TimberTech Care Cleaning Guide': 'TimberTech Cleaning Guide'
        }
        
        # Return mapped name or clean up the original
        if doc_name in name_mapping:
            return name_mapping[doc_name]
        else:
            # Clean up the filename for display
            display_name = doc_name.replace('.pdf', '')
            display_name = display_name.replace('_', ' ')
            display_name = display_name.replace('-', ' ')
            return display_name
    
    def generate_follow_up_questions(self, question: str, answer: str) -> List[str]:
        """Generate relevant follow-up questions based on the actual response content using AI."""
        try:
            prompt = f"""Based on this HOA Q&A conversation, generate 2-3 specific follow-up questions that a homeowner might naturally ask next. The questions should be based on the actual content of the answer, not just variations of the original question.

Original Question: {question}

Answer: {answer}

Generate follow-up questions that:
1. Ask for more details about specific points mentioned in the answer
2. Ask about related processes or next steps
3. Ask about exceptions, penalties, or alternatives
4. Are specific and actionable

Return only the questions, one per line, without numbering or bullet points."""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates relevant follow-up questions for HOA conversations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            # Parse the response into individual questions
            questions_text = response.choices[0].message.content.strip()
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            
            # Clean up any numbering or bullet points
            cleaned_questions = []
            for q in questions:
                # Remove common prefixes
                q = q.lstrip('•-123456789. ')
                if q and len(q) > 10:  # Only include substantial questions
                    cleaned_questions.append(q)
            
            return cleaned_questions[:2]  # Return max 2 questions
            
        except Exception as e:
            print(f"Error generating follow-up questions: {e}")
            # Fallback to simple questions if AI fails
            return [
                "Can you tell me more about this?",
                "What are the next steps?"
            ]

def create_streamlit_app():
    st.set_page_config(
        page_title="HOA Chatbot",
        page_icon="��",
        layout="wide"
    )
    st.title("🏠 Turtle Rock Crest HOA Assistant")
    st.markdown("Got questions about your HOA? Ask away!")

    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Loading your HOA documents..."):
            st.session_state.chatbot = HOAChatbot()

    # Check document status
    document_status = st.session_state.chatbot.get_document_status()
    processed_count = sum(1 for status in document_status.values() if status['status'] == 'processed')
    total_count = len(document_status)

    # Sidebar for document info
    with st.sidebar:
        st.header("Available Documents for Reference")
        
        # Custom CSS for left-aligned button text
        st.markdown("""
        <style>
        .stButton > button {
            text-align: left !important;
            justify-content: flex-start !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if processed_count > 0:
            for doc_name, status in document_status.items():
                if status['status'] == 'processed':
                    # Get clean display name
                    display_name = st.session_state.chatbot.get_display_name(doc_name)
                    
                    # Simple document button - just trigger a search
                    if st.button(f"{display_name}", key=f"doc_{doc_name}", use_container_width=True):
                        st.session_state.suggested_question = f"Tell me about {doc_name}"
                        st.rerun()
        else:
            st.warning("⚠️ No documents available")
            st.info("Documents need to be processed first")

    # Using OpenAI (AI-powered responses)
    mode = "openai (AI-powered)" # OpenAI provides intelligent responses

    # Suggested questions section
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Show suggested questions if no chat history
    if not st.session_state.chat_history:
        st.markdown("### Try asking about:")

        # Create columns for the suggested questions
        col1, col2 = st.columns(2)

        with col1:
            if st.button(" What are the parking rules?", use_container_width=True):
                st.session_state.suggested_question = "What are the parking rules?"
                st.rerun()
            if st.button(" How much are the monthly fees?", use_container_width=True):
                st.session_state.suggested_question = "How much are the monthly fees?"
                st.rerun()
            if st.button(" Can I have pets?", use_container_width=True):
                st.session_state.suggested_question = "Can I have pets?"
                st.rerun()
            if st.button(" What can I modify on my property?", use_container_width=True):
                st.session_state.suggested_question = "What can I modify on my property?"
                st.rerun()

        with col2:
            if st.button(" How do I submit a complaint?", use_container_width=True):
                st.session_state.suggested_question = "How do I submit a complaint?"
                st.rerun()
            if st.button(" Can I rent my place out?", use_container_width=True):
                st.session_state.suggested_question = "Can I rent my place out?"
                st.rerun()
            if st.button(" What are the quiet hours?", use_container_width=True):
                st.session_state.suggested_question = "What are the quiet hours?"
                st.rerun()
            if st.button(" How do I contact the board?", use_container_width=True):
                st.session_state.suggested_question = "How do I contact the board?"
                st.rerun()

        st.markdown("---")

    # Handle suggested question if one was clicked
    if hasattr(st.session_state, 'suggested_question') and st.session_state.suggested_question:
        suggested_q = st.session_state.suggested_question
        del st.session_state.suggested_question  # Clear it after using

        # Add the suggested question to chat history
        st.session_state.chat_history.append({'role': 'user', 'content': suggested_q})

        # Get the answer
        with st.spinner(""): # Removed loading text
            try:
                result = st.session_state.chatbot.answer_question(suggested_q, use_ai="openai")
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': result['answer'],
                    'sources': result['sources'],
                    'method_used': result.get('method_used', 'Unknown')
                })
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': error_msg
                })
        st.rerun() # Rerun to display the new messages

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            st.chat_message("user").write(message['content'])
        else:
            st.chat_message("assistant").write(message['content'])
            
            # Show which method was used
            if 'method_used' in message:
                st.caption(f"*{message['method_used']}*")
            
            # Show sources right after the answer
            if 'sources' in message:
                with st.expander("📄 View Sources", expanded=False):
                    st.markdown("""
                    <style>
                    .stExpander {
                        background-color: #2d3748;
                        border: 1px solid #4a5568;
                        border-radius: 8px;
                        padding: 10px;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    for j, source in enumerate(message['sources'], 1):
                        # Show the clean display name for the document
                        display_name = st.session_state.chatbot.get_display_name(source['document'])
                        
                        # Create a clean, simple source display
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.write(f"**Source {j}:**")
                        with col2:
                            # Check if PDF exists and create appropriate button
                            if os.path.exists(source['document']):
                                # Create a link to open PDF in new tab
                                pdf_url = f"file://{os.path.abspath(source['document'])}"
                                st.markdown(f"""
                                <a href="{pdf_url}" target="_blank" style="text-decoration: none;">
                                    <button style="
                                        background-color: #1f77b4;
                                        color: white;
                                        border: none;
                                        padding: 6px 12px;
                                        border-radius: 4px;
                                        cursor: pointer;
                                        width: 100%;
                                        font-size: 12px;
                                    ">📄 {display_name}</button>
                                </a>
                                """, unsafe_allow_html=True)
                            else:
                                # Fallback to search button if PDF not found
                                if st.button(
                                    label=display_name,
                                    key=f"source_pdf_{id(message)}_{j}_{source['document'].replace('.', '_').replace(' ', '_')}",
                                    help="Click to search this document",
                                    use_container_width=True
                                ):
                                    # Search for more information about this document
                                    st.session_state.suggested_question = f"Tell me more about {display_name}"
                                    st.rerun()
                        
                        # Show preview text in a subtle way
                        st.caption(f"*{source['text_preview'][:150]}...*")
                        st.divider()
            
            # Show follow-up questions after assistant responses (positioned after sources)
            if i > 0 and st.session_state.chat_history[i-1]['role'] == 'user':
                # Get the previous user question and current assistant answer
                user_question = st.session_state.chat_history[i-1]['content']
                assistant_answer = message['content']
                
                # Generate follow-up questions
                follow_ups = st.session_state.chatbot.generate_follow_up_questions(user_question, assistant_answer)
                
                if follow_ups:
                    st.markdown("**Suggested follow-up questions:**")
                    col1, col2 = st.columns(2)
                    
                    for j, follow_up in enumerate(follow_ups):
                        col = col1 if j % 2 == 0 else col2
                        with col:
                            if st.button(f"{follow_up}", key=f"followup_{i}_{j}", use_container_width=True):
                                st.session_state.suggested_question = follow_up
                                st.rerun()

    # Chat input
    if prompt := st.chat_input("What do you want to know about your HOA?"):
        st.chat_message("user").write(prompt)
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        
        with st.spinner(""): # Removed loading text
            try:
                result = st.session_state.chatbot.answer_question(prompt, use_ai="openai")
                st.chat_message("assistant").write(result['answer'])
                
                # Show which method was used
                st.caption(f"*{result.get('method_used', 'Unknown method')}*")
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': result['answer'],
                    'sources': result['sources'],
                    'method_used': result.get('method_used', 'Unknown')
                })
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.chat_message("assistant").write(error_msg)
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': error_msg
                })

    # Clear chat button
    if st.session_state.chat_history:
        if st.button(" Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    create_streamlit_app()