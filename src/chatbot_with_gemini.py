import os
import json
from typing import List, Dict, Optional
import numpy as np
from dotenv import load_dotenv
import streamlit as st

# Google's Gemini for AI responses
import google.generativeai as genai

# Load config from .env file
load_dotenv()

class HOAChatbot:
    def __init__(self, data_file: str = "data/processed_documents.json"):
        self.data_file = data_file
        self.documents = self.load_documents()
        self.embedding_model = None
        self.document_embeddings = []
        
        # Set up Gemini - our main AI engine
        self.gemini_api_key = os.getenv('GOOGLE_API_KEY')
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            try:
                # Try different Gemini models - some work better than others
                model_options = [
                    'models/gemini-1.5-pro',
                    'models/gemini-2.0-flash', 
                    'models/gemini-1.5-flash',
                    'models/gemini-2.0-flash-001'
                ]
                
                self.gemini_model = None
                for model_name in model_options:
                    try:
                        self.gemini_model = genai.GenerativeModel(model_name)
                        print(f"Got Gemini working with: {model_name}")
                        break
                    except Exception as e:
                        print(f"Model {model_name} didn't work: {e}")
                        continue
                
                if not self.gemini_model:
                    print("None of the Gemini models worked")
                    
            except Exception as e:
                print(f"Gemini model error: {e}")
                self.gemini_model = None
        else:
            self.gemini_model = None
            
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
        """Check which PDFs were processed successfully and which failed."""
        all_pdfs = self.get_all_pdf_files()
        status = {}
        
        for pdf_file in all_pdfs:
            if pdf_file in self.documents:
                status[pdf_file] = {
                    'status': 'processed',
                    'sections': len(self.documents[pdf_file]['sections']),
                    'total_chars': len(self.documents[pdf_file]['cleaned_text']),
                    'raw_length': len(self.documents[pdf_file]['raw_text'])
                }
            else:
                status[pdf_file] = {
                    'status': 'not_processed',
                    'sections': 0,
                    'total_chars': 0,
                    'raw_length': 0
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
    
    def generate_response_with_gemini(self, query: str, relevant_sections: List[Dict]) -> str:
        if not self.gemini_model:
            return "Gemini API key not available or model not found. Please check your GOOGLE_API_KEY configuration."
        if not relevant_sections:
            return "I don't have enough information to answer that question. Please try rephrasing or ask about a different topic."
        context = "\n\n".join([
            f"From {section['document']}:\n{section['text']}"
            for section in relevant_sections
        ])
        prompt = f"""You're helping someone with their HOA questions. Use the info below from their HOA docs to answer their question. If you can't find the answer in what's provided, just say so.\n\nHere's what I found in their HOA documents:\n{context}\n\nTheir question: {query}\n\nGive them a clear answer based on their HOA docs. If you mention specific rules, tell them which document it's from."""
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Try to list available models for debugging
            try:
                models = genai.list_models()
                model_list = '\n'.join([m.name for m in models])
                return f"Gemini error: {e}\nAvailable models:\n{model_list}"
            except Exception as e2:
                return f"Gemini had an issue: {str(e)}\nAlso couldn't list models: {str(e2)}"
    
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
    
    def answer_question(self, query: str, use_ai: str = "gemini") -> Dict:
        relevant_sections = self.find_relevant_sections(query)
        
        # Only use Gemini or search fallback
        if use_ai == "gemini" and self.gemini_model:
            try:
                response = self.generate_response_with_gemini(query, relevant_sections)
            except Exception as e:
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
        st.header("📄 Your Documents")
        
        if processed_count > 0:
            st.success(f"✅ {processed_count} document(s) processed")
            for doc_name, status in document_status.items():
                if status['status'] == 'processed':
                    st.write(f"• {doc_name}")
        else:
            st.warning("⚠️ No documents processed yet")
            st.info("Run `python src/pdf_processor.py` to process your PDFs")

    # Using Gemini for AI responses
    mode = "gemini (free)" # Gemini is the only option now

    # Suggested questions section
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Show suggested questions if no chat history
    if not st.session_state.chat_history:
        st.markdown("### �� Try asking about:")

        # Create columns for the suggested questions
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🏠 What are the parking rules?", use_container_width=True):
                st.session_state.suggested_question = "What are the parking rules?"
                st.rerun()
            if st.button("💰 How much are the monthly fees?", use_container_width=True):
                st.session_state.suggested_question = "How much are the monthly fees?"
                st.rerun()
            if st.button("�� Can I have pets?", use_container_width=True):
                st.session_state.suggested_question = "Can I have pets?"
                st.rerun()
            if st.button("��️ What can I modify on my property?", use_container_width=True):
                st.session_state.suggested_question = "What can I modify on my property?"
                st.rerun()

        with col2:
            if st.button("📝 How do I submit a complaint?", use_container_width=True):
                st.session_state.suggested_question = "How do I submit a complaint?"
                st.rerun()
            if st.button("🏠 Can I rent my place out?", use_container_width=True):
                st.session_state.suggested_question = "Can I rent my place out?"
                st.rerun()
            if st.button("�� What are the quiet hours?", use_container_width=True):
                st.session_state.suggested_question = "What are the quiet hours?"
                st.rerun()
            if st.button("📞 How do I contact the board?", use_container_width=True):
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
                result = st.session_state.chatbot.answer_question(suggested_q, use_ai="gemini")
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
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.chat_message("user").write(message['content'])
        else:
            st.chat_message("assistant").write(message['content'])
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
                    for i, source in enumerate(message['sources'], 1):
                        st.write(f"**Source {i}:**")
                        if os.path.exists(source['document']):
                            with open(source['document'], "rb") as pdf_file_obj:
                                pdf_bytes = pdf_file_obj.read()
                            st.download_button(
                                label=f"📄 {source['document']}",
                                data=pdf_bytes,
                                file_name=source['document'],
                                mime="application/pdf",
                                use_container_width=True,
                                key=f"source_pdf_{i}_{source['document'].replace('.', '_').replace(' ', '_')}"
                            )
                        st.write(f"**Preview:** {source['text_preview']}")
                        st.divider()

    # Chat input
    if prompt := st.chat_input("What do you want to know about your HOA?"):
        st.chat_message("user").write(prompt)
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        
        with st.spinner(""): # Removed loading text
            try:
                result = st.session_state.chatbot.answer_question(prompt, use_ai="gemini")
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
        if st.button("��️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    create_streamlit_app()