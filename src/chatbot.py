import os
import json
from typing import List, Dict, Optional
import numpy as np
from dotenv import load_dotenv
import streamlit as st

# Llama 3.1 for AI responses (local, free)
import requests

# Load config from .env file
load_dotenv()

class HOAChatbot:
    def __init__(self, data_file: str = "data/processed_documents.json"):
        self.data_file = data_file
        self.documents = self.load_documents()
        self.embedding_model = None
        self.document_embeddings = []
        
        # Set up Llama 3.1 - our local AI engine (free, no API keys needed)
        self.ollama_available = self._check_ollama()
        if self.ollama_available:
            print("✓ Llama 3.1 available via Ollama")
        else:
            print("⚠️ Ollama not available - using search mode")
            
        # Load the text embedding model for finding relevant docs
        self._load_embedding_model()
        
        # Build search index from all the document sections
        if self.embedding_model:
            self.document_embeddings = self.create_embeddings()
    
    def _check_ollama(self):
        """Check if Ollama is running and Llama 3.1 is available."""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any('llama3.1' in model.get('name', '') for model in models)
        except:
            pass
        return False
    
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
    
    def generate_response_with_llama(self, query: str, relevant_sections: List[Dict]) -> str:
        if not self.ollama_available:
            return "Llama 3.1 not available. Please make sure Ollama is running."
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
            response = requests.post('http://localhost:11434/api/generate', 
                                   json={
                                       'model': 'llama3.1:8b',
                                       'prompt': prompt,
                                       'stream': False
                                   },
                                   timeout=30)
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Ollama error: {response.status_code}"
        except Exception as e:
            return f"Error calling Llama: {str(e)}"
    
    def generate_response_with_search(self, query: str, relevant_sections: List[Dict]) -> str:
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
    
    def answer_question(self, query: str, use_ai: str = "llama") -> Dict:
        relevant_sections = self.find_relevant_sections(query)
        
        # Use Llama 3.1 if available, otherwise fall back to search
        if use_ai == "llama" and self.ollama_available:
            try:
                response = self.generate_response_with_llama(query, relevant_sections)
            except Exception as e:
                print(f"Llama failed, using search mode: {e}")
                response = self.generate_response_with_search(query, relevant_sections)
        else:
            response = self.generate_response_with_search(query, relevant_sections)
        
        return {
            'question': query,
            'answer': response,
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
        """Generate relevant follow-up questions based on the Q&A."""
        # Define follow-up question templates based on common HOA topics
        follow_up_templates = {
            'parking': [
                "What are the guest parking rules?",
                "Can I get a parking permit?",
                "What happens if I park in the wrong spot?"
            ],
            'modify': [
                "What's the approval process for modifications?",
                "How long does approval take?",
                "What modifications don't need approval?"
            ],
            'fees': [
                "When are fees due?",
                "What happens if I'm late on payments?",
                "Are there any fee discounts available?"
            ],
            'pets': [
                "How many pets can I have?",
                "Are there breed restrictions?",
                "What about pet deposits?"
            ],
            'rent': [
                "Can I rent out part of my property?",
                "What are the rental restrictions?",
                "Do I need to notify the HOA?"
            ],
            'complaint': [
                "How do I file a complaint?",
                "What's the complaint process?",
                "How long does it take to resolve?"
            ],
            'board': [
                "How do I contact the board?",
                "When are board meetings?",
                "How can I run for the board?"
            ],
            'quiet': [
                "What are the noise restrictions?",
                "When are quiet hours?",
                "What about construction noise?"
            ]
        }
        
        # Find relevant follow-ups based on question content
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        follow_ups = []
        
        # Check for keywords in the question
        for topic, questions in follow_up_templates.items():
            if topic in question_lower:
                follow_ups.extend(questions[:2])  # Take first 2 questions
                break
        
        # If no specific topic found, generate general follow-ups
        if not follow_ups:
            if 'rule' in question_lower or 'policy' in question_lower:
                follow_ups = [
                    "What are the penalties for violating this rule?",
                    "How is this rule enforced?"
                ]
            elif 'fee' in question_lower or 'cost' in question_lower:
                follow_ups = [
                    "When are these fees due?",
                    "What happens if I can't pay on time?"
                ]
            elif 'approval' in question_lower or 'permit' in question_lower:
                follow_ups = [
                    "How long does the approval process take?",
                    "What documents do I need to submit?"
                ]
            else:
                follow_ups = [
                    "What are the penalties for not following this?",
                    "How do I get more information about this?"
                ]
        
        return follow_ups[:3]  # Return max 3 follow-up questions

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
        st.header("📚 Available Documents for Reference")
        
        if processed_count > 0:
            for doc_name, status in document_status.items():
                if status['status'] == 'processed':
                    # Get clean display name
                    display_name = st.session_state.chatbot.get_display_name(doc_name)
                    
                    # Use expander for each document
                    with st.expander(f"{display_name}", expanded=False):
                        st.write(f"**File:** {doc_name}")
                        
                        # Show document stats
                        st.write(f"**Sections:** {status['sections']}")
                        st.write(f"**Characters:** {status['total_chars']:,}")
                        
                        # Show a sample of the document content
                        if doc_name in st.session_state.chatbot.documents:
                            doc_data = st.session_state.chatbot.documents[doc_name]
                            st.subheader("Document Preview")
                            preview_text = doc_data['cleaned_text'][:500] + "..." if len(doc_data['cleaned_text']) > 500 else doc_data['cleaned_text']
                            st.text_area("Content preview:", preview_text, height=150, disabled=True)
                        
                        # Check for PDF in multiple locations
                        pdf_found = False
                        pdf_path = None
                        
                        # Check common locations for PDFs
                        possible_paths = [
                            doc_name,  # Current directory
                            f"../{doc_name}",  # Parent directory
                            f"../pdfs/{doc_name}",  # PDFs folder
                            f"pdfs/{doc_name}",  # Local pdfs folder
                            f"documents/{doc_name}",  # Documents folder
                        ]
                        
                        for path in possible_paths:
                            if os.path.exists(path):
                                pdf_found = True
                                pdf_path = path
                                break
                        
                        if pdf_found and pdf_path:
                            # Create a link to open PDF in new tab
                            pdf_url = f"file://{os.path.abspath(pdf_path)}"
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"""
                                <a href="{pdf_url}" target="_blank" style="text-decoration: none;">
                                    <button style="
                                        background-color: #1f77b4;
                                        color: white;
                                        border: none;
                                        padding: 8px 16px;
                                        border-radius: 4px;
                                        cursor: pointer;
                                        width: 100%;
                                        font-size: 14px;
                                    ">📄 View PDF</button>
                                </a>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                with open(pdf_path, "rb") as pdf_file:
                                    pdf_bytes = pdf_file.read()
                                st.download_button(
                                    label="📥 Download",
                                    data=pdf_bytes,
                                    file_name=doc_name,
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                        else:
                            # Show document content in a more useful way
                            st.info("📄 Document content is available in the chat responses")
                            
                            # Add buttons for document interaction
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"🔍 Search in {display_name}", key=f"search_{doc_name}", use_container_width=True):
                                    st.session_state.suggested_question = f"Tell me about {display_name}"
                                    st.rerun()
                            
                            with col2:
                                if st.button(f"📖 View Content", key=f"view_{doc_name}", use_container_width=True):
                                    # Show document content in an expandable section
                                    st.session_state[f"show_content_{doc_name}"] = not st.session_state.get(f"show_content_{doc_name}", False)
                                    st.rerun()
                            
                            # Show document content if requested
                            if st.session_state.get(f"show_content_{doc_name}", False):
                                st.markdown("**📖 Document Content:**")
                                doc_data = st.session_state.chatbot.documents[doc_name]
                                st.text_area("", doc_data['cleaned_text'], height=300, disabled=True)
        else:
            st.warning("⚠️ No documents available")
            st.info("Documents need to be processed first")

    # Using Llama 3.1 (local AI, completely free)
    mode = "llama 3.1 (local, free)" # Llama 3.1 runs locally, no API keys needed

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
                result = st.session_state.chatbot.answer_question(suggested_q, use_ai="llama")
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': result['answer'],
                    'sources': result['sources']
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
                    st.markdown("**💡 Suggested follow-up questions:**")
                    col1, col2 = st.columns(2)
                    
                    for j, follow_up in enumerate(follow_ups):
                        col = col1 if j % 2 == 0 else col2
                        with col:
                            if st.button(f"❓ {follow_up}", key=f"followup_{i}_{j}", use_container_width=True):
                                st.session_state.suggested_question = follow_up
                                st.rerun()

    # Chat input
    if prompt := st.chat_input("What do you want to know about your HOA?"):
        st.chat_message("user").write(prompt)
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        
        with st.spinner(""): # Removed loading text
            try:
                result = st.session_state.chatbot.answer_question(prompt, use_ai="llama")
                st.chat_message("assistant").write(result['answer'])
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': result['answer'],
                    'sources': result['sources']
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