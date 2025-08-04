import os
import json
import openai
from typing import List, Dict, Optional
import numpy as np
from dotenv import load_dotenv
import streamlit as st

# Gemini import
import google.generativeai as genai

# Load environment variables
load_dotenv()

class HOAChatbot:
    def __init__(self, data_file: str = "data/processed_documents.json"):
        self.data_file = data_file
        self.documents = self.load_documents()
        self.embedding_model = None
        self.document_embeddings = []
        
        # Initialize OpenAI client
        try:
            self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        except:
            self.openai_client = None
        # Initialize Gemini client
        self.gemini_api_key = os.getenv('GOOGLE_API_KEY')
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            try:
                # Try available models in order of preference
                available_models = [
                    'models/gemini-1.5-pro',
                    'models/gemini-2.0-flash', 
                    'models/gemini-1.5-flash',
                    'models/gemini-2.0-flash-001'
                ]
                
                self.gemini_model = None
                for model_name in available_models:
                    try:
                        self.gemini_model = genai.GenerativeModel(model_name)
                        print(f"Successfully loaded Gemini model: {model_name}")
                        break
                    except Exception as e:
                        print(f"Failed to load {model_name}: {e}")
                        continue
                
                if not self.gemini_model:
                    print("Could not load any Gemini model")
                    
            except Exception as e:
                print(f"Gemini model error: {e}")
                self.gemini_model = None
        else:
            self.gemini_model = None
        # Load embedding model with error handling
        self._load_embedding_model()
        # Create embeddings for all document sections
        if self.embedding_model:
            self.document_embeddings = self.create_embeddings()
    def _load_embedding_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            print("✓ Embedding model loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load embedding model: {e}")
            self.embedding_model = None
    def load_documents(self) -> Dict:
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Data file {self.data_file} not found. Please run pdf_processor.py first.")
            return {}
    
    def get_all_pdf_files(self) -> List[str]:
        """Get all PDF files in the current directory."""
        pdf_files = []
        for file in os.listdir('.'):
            if file.endswith('.pdf'):
                pdf_files.append(file)
        return pdf_files
    
    def get_document_status(self) -> Dict:
        """Get status of all PDF files, including those that failed to process."""
        all_pdfs = self.get_all_pdf_files()
        status = {}
        
        for pdf_file in all_pdfs:
            if pdf_file in self.documents:
                status[pdf_file] = {
                    'status': 'processed',
                    'sections': len(self.documents[pdf_file]['sections']),
                    'total_chars': len(self.documents[pdf_file]['cleaned_text']),
                    'file_size_mb': len(self.documents[pdf_file]['raw_text']) / (1024 * 1024)
                }
            else:
                status[pdf_file] = {
                    'status': 'failed',
                    'reason': 'OCR required for scanned document',
                    'file_size_mb': os.path.getsize(pdf_file) / (1024 * 1024)
                }
        
        return status
    def create_embeddings(self) -> List[Dict]:
        if not self.embedding_model:
            print("❌ No embedding model available")
            return []
        embeddings = []
        try:
            for doc_name, doc_data in self.documents.items():
                for i, section in enumerate(doc_data['sections']):
                    embedding = self.embedding_model.encode(section)
                    embeddings.append({
                        'document': doc_name,
                        'section_index': i,
                        'text': section,
                        'embedding': embedding
                    })
            print(f"✓ Created embeddings for {len(embeddings)} document sections")
        except Exception as e:
            print(f"❌ Error creating embeddings: {e}")
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
            print(f"❌ Error in semantic search: {e}")
            return self.find_relevant_sections(query, top_k)
    def generate_response_with_openai(self, query: str, relevant_sections: List[Dict]) -> str:
        if not self.openai_client:
            return "OpenAI client not available. Please check your API key configuration."
        if not relevant_sections:
            return "I don't have enough information to answer that question. Please try rephrasing or ask about a different topic."
        context = "\n\n".join([
            f"From {section['document']}:\n{section['text']}"
            for section in relevant_sections
        ])
        prompt = f"""You are a helpful HOA assistant. Use the following information from HOA documents to answer the user's question. \nIf the information is not in the provided context, say so clearly.\n\nContext from HOA documents:\n{context}\n\nUser question: {query}\n\nPlease provide a clear, helpful answer based on the HOA documents. If you reference specific rules or sections, mention which document they come from."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful HOA assistant that provides accurate information based on HOA bylaws and CC&Rs."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Sorry, I encountered an error while generating a response: {str(e)}"
    def generate_response_with_gemini(self, query: str, relevant_sections: List[Dict]) -> str:
        if not self.gemini_model:
            return "Gemini API key not available or model not found. Please check your GOOGLE_API_KEY configuration."
        if not relevant_sections:
            return "I don't have enough information to answer that question. Please try rephrasing or ask about a different topic."
        context = "\n\n".join([
            f"From {section['document']}:\n{section['text']}"
            for section in relevant_sections
        ])
        prompt = f"""You are a helpful HOA assistant. Use the following information from HOA documents to answer the user's question. \nIf the information is not in the provided context, say so clearly.\n\nContext from HOA documents:\n{context}\n\nUser question: {query}\n\nPlease provide a clear, helpful answer based on the HOA documents. If you reference specific rules or sections, mention which document they come from."""
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
                return f"Sorry, I encountered an error while generating a response with Gemini: {str(e)}\nAlso failed to list models: {str(e2)}"
    def generate_response_with_search(self, query: str, relevant_sections: List[Dict]) -> str:
        if not relevant_sections:
            return "I don't have enough information to answer that question. Please try rephrasing or ask about a different topic."
        response_parts = []
        response_parts.append(f"Based on your question about '{query}', here's what I found in the HOA documents:")
        for i, section in enumerate(relevant_sections, 1):
            section_text = section['text']
            sentences = section_text.split('. ')
            summary = '. '.join(sentences[:3]) + '.' if len(sentences) > 3 else section_text
            response_parts.append(f"\n**From {section['document']}:**")
            response_parts.append(summary)
        response_parts.append(f"\n\n*This information was found by searching through the HOA documents. For the complete details, please refer to the original documents.*")
        return '\n'.join(response_parts)
    def answer_question(self, query: str, use_ai: str = "search") -> Dict:
        relevant_sections = self.find_relevant_sections(query)
        if use_ai == "openai" and self.openai_client:
            try:
                response = self.generate_response_with_openai(query, relevant_sections)
                if "Error code: 429" in response or "quota" in response.lower():
                    response = self.generate_response_with_search(query, relevant_sections)
            except Exception as e:
                response = self.generate_response_with_search(query, relevant_sections)
        elif use_ai == "gemini" and self.gemini_model:
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
                    'text_preview': section['text'][:200] + "..."
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
        page_icon="🏠",
        layout="wide"
    )
    st.title("🏠 HOA Chatbot")
    st.markdown("Ask questions about your HOA bylaws and CC&Rs!")
    
    # Check document status
    if 'chatbot' not in st.session_state:
        try:
            with st.spinner("Loading chatbot..."):
                st.session_state.chatbot = HOAChatbot()
        except Exception as e:
            st.error(f"❌ Error loading chatbot: {e}")
            st.info("Please make sure you've run the PDF processor first.")
            return
    
    # Show document status
    doc_status = st.session_state.chatbot.get_document_status()
    processed_count = sum(1 for status in doc_status.values() if status['status'] == 'processed')
    failed_count = sum(1 for status in doc_status.values() if status['status'] == 'failed')
    
    if processed_count > 0:
        st.success(f"✅ {processed_count} document(s) processed successfully")
    if failed_count > 0:
        st.warning(f"⚠️ {failed_count} document(s) failed to process (OCR required)")
        st.info("💡 **To process scanned PDFs (like CC&Rs), you need to install Tesseract OCR:**")
        st.code("brew install tesseract  # macOS\n# or download from https://github.com/tesseract-ocr/tesseract")
    
    mode = st.sidebar.selectbox("Choose answer mode", ["search (free)", "gemini (free)", "openai (paid)"])
    if mode == "search (free)":
        st.sidebar.info("🔍 Using document search mode (free, no API costs)")
    elif mode == "gemini (free)":
        st.sidebar.info("✨ Using Gemini AI mode (free, Google API key required)")
    else:
        st.sidebar.warning("🤖 Using OpenAI mode (requires OpenAI credits)")
    
    if not st.session_state.chatbot.documents:
        st.error("❌ No documents loaded. Please run pdf_processor.py first.")
        return
    with st.sidebar:
        st.header("📚 Available Documents")
        stats = st.session_state.chatbot.get_document_status()
        for pdf_file, doc_status in stats.items():
            with st.expander(pdf_file):
                st.write(f"**Status:** {doc_status['status']}")
                if doc_status['status'] == 'processed':
                    st.write(f"**Sections:** {doc_status['sections']}")
                    st.write(f"**Characters:** {doc_status['total_chars']:,}")
                    st.write(f"**Size:** {doc_status['file_size_mb']:.1f} MB")
                elif doc_status['status'] == 'failed':
                    st.write(f"**Reason:** {doc_status['reason']}")
                    st.write(f"**Size:** {doc_status['file_size_mb']:.1f} MB")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.chat_message("user").write(message['content'])
        else:
            st.chat_message("assistant").write(message['content'])
            if 'sources' in message:
                with st.expander("📄 View Sources"):
                    for i, source in enumerate(message['sources'], 1):
                        st.write(f"**Source {i}:** {source['document']}")
                        st.write(f"**Preview:** {source['text_preview']}")
                        st.divider()
    if prompt := st.chat_input("Ask a question about your HOA documents..."):
        st.chat_message("user").write(prompt)
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        with st.spinner("Searching documents and generating response..."):
            try:
                mode_map = {"search (free)": "search", "gemini (free)": "gemini", "openai (paid)": "openai"}
                result = st.session_state.chatbot.answer_question(prompt, use_ai=mode_map[mode])
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
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    create_streamlit_app() 