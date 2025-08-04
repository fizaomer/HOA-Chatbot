import os
import json
import openai
from typing import List, Dict, Optional
import numpy as np
from dotenv import load_dotenv
import streamlit as st

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
        
        # Load embedding model with error handling
        self._load_embedding_model()
        
        # Create embeddings for all document sections
        if self.embedding_model:
            self.document_embeddings = self.create_embeddings()
        
    def _load_embedding_model(self):
        """Load the embedding model with error handling."""
        try:
            from sentence_transformers import SentenceTransformer
            # Set environment variable to avoid tokenizer warnings
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            
            # Load model with specific device configuration
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            print("✓ Embedding model loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load embedding model: {e}")
            self.embedding_model = None
        
    def load_documents(self) -> Dict:
        """Load processed documents from JSON file."""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Data file {self.data_file} not found. Please run pdf_processor.py first.")
            return {}
    
    def create_embeddings(self) -> List[Dict]:
        """Create embeddings for all document sections."""
        if not self.embedding_model:
            print("❌ No embedding model available")
            return []
            
        embeddings = []
        
        try:
            for doc_name, doc_data in self.documents.items():
                for i, section in enumerate(doc_data['sections']):
                    # Create embedding for this section
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
        """Find the most relevant document sections for a given query."""
        if not self.document_embeddings or not self.embedding_model:
            # Fallback: return first few sections
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
            # Create embedding for the query
            query_embedding = self.embedding_model.encode(query)
            
            # Calculate similarities
            similarities = []
            for doc_embedding in self.document_embeddings:
                similarity = np.dot(query_embedding, doc_embedding['embedding']) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding['embedding'])
                )
                similarities.append((similarity, doc_embedding))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [item[1] for item in similarities[:top_k]]
        except Exception as e:
            print(f"❌ Error in semantic search: {e}")
            # Fallback: return first few sections
            return self.find_relevant_sections(query, top_k)
    
    def generate_response_with_openai(self, query: str, relevant_sections: List[Dict]) -> str:
        """Generate a response using OpenAI API."""
        if not self.openai_client:
            return "OpenAI client not available. Please check your API key configuration."
            
        if not relevant_sections:
            return "I don't have enough information to answer that question. Please try rephrasing or ask about a different topic."
        
        # Prepare context from relevant sections
        context = "\n\n".join([
            f"From {section['document']}:\n{section['text']}"
            for section in relevant_sections
        ])
        
        # Create the prompt
        prompt = f"""You are a helpful HOA assistant. Use the following information from HOA documents to answer the user's question. 
        If the information is not in the provided context, say so clearly.

        Context from HOA documents:
        {context}

        User question: {query}

        Please provide a clear, helpful answer based on the HOA documents. If you reference specific rules or sections, mention which document they come from."""

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
    
    def generate_response_with_search(self, query: str, relevant_sections: List[Dict]) -> str:
        """Generate a response using only document search (no AI generation)."""
        if not relevant_sections:
            return "I don't have enough information to answer that question. Please try rephrasing or ask about a different topic."
        
        # Create a response based on the most relevant sections
        response_parts = []
        response_parts.append(f"Based on your question about '{query}', here's what I found in the HOA documents:")
        
        for i, section in enumerate(relevant_sections, 1):
            # Extract a summary from the section
            section_text = section['text']
            # Take the first few sentences as a summary
            sentences = section_text.split('. ')
            summary = '. '.join(sentences[:3]) + '.' if len(sentences) > 3 else section_text
            
            response_parts.append(f"\n**From {section['document']}:**")
            response_parts.append(summary)
        
        response_parts.append(f"\n\n*This information was found by searching through the HOA documents. For the complete details, please refer to the original documents.*")
        
        return '\n'.join(response_parts)
    
    def answer_question(self, query: str, use_ai: bool = True) -> Dict:
        """Answer a question about HOA documents."""
        # Find relevant sections
        relevant_sections = self.find_relevant_sections(query)
        
        # Generate response
        if use_ai and self.openai_client:
            try:
                response = self.generate_response_with_openai(query, relevant_sections)
                # Check if response contains an error
                if "Error code: 429" in response or "quota" in response.lower():
                    # Fall back to search-based response
                    response = self.generate_response_with_search(query, relevant_sections)
            except Exception as e:
                # Fall back to search-based response
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
        """Get list of available documents."""
        return list(self.documents.keys())
    
    def get_document_stats(self) -> Dict:
        """Get statistics about the loaded documents."""
        stats = {}
        for doc_name, doc_data in self.documents.items():
            stats[doc_name] = {
                'sections': len(doc_data['sections']),
                'total_chars': len(doc_data['cleaned_text']),
                'file_size_mb': len(doc_data['raw_text']) / (1024 * 1024)
            }
        return stats

# Streamlit interface
def create_streamlit_app():
    st.set_page_config(
        page_title="HOA Chatbot",
        page_icon="🏠",
        layout="wide"
    )
    
    st.title("🏠 HOA Chatbot")
    st.markdown("Ask questions about your HOA bylaws and CC&Rs!")
    
    # Add mode selection
    use_ai = st.sidebar.checkbox("Use AI Generation (requires OpenAI credits)", value=False)
    if not use_ai:
        st.sidebar.info("🔍 Using document search mode (free, no API costs)")
    else:
        st.sidebar.warning("🤖 Using AI mode (requires OpenAI credits)")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        try:
            with st.spinner("Loading chatbot..."):
                st.session_state.chatbot = HOAChatbot()
            
            if st.session_state.chatbot.documents:
                st.success("✅ Chatbot loaded successfully!")
            else:
                st.error("❌ No documents loaded. Please run pdf_processor.py first.")
                return
                
        except Exception as e:
            st.error(f"❌ Error loading chatbot: {e}")
            st.info("Please make sure you've run the PDF processor first.")
            return
    
    # Sidebar with document info
    with st.sidebar:
        st.header("📚 Available Documents")
        stats = st.session_state.chatbot.get_document_stats()
        
        for doc_name, doc_stats in stats.items():
            with st.expander(doc_name):
                st.write(f"**Sections:** {doc_stats['sections']}")
                st.write(f"**Characters:** {doc_stats['total_chars']:,}")
                st.write(f"**Size:** {doc_stats['file_size_mb']:.1f} MB")
    
    # Main chat interface
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.chat_message("user").write(message['content'])
        else:
            st.chat_message("assistant").write(message['content'])
            
            # Show sources if available
            if 'sources' in message:
                with st.expander("📄 View Sources"):
                    for i, source in enumerate(message['sources'], 1):
                        st.write(f"**Source {i}:** {source['document']}")
                        st.write(f"**Preview:** {source['text_preview']}")
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your HOA documents..."):
        # Add user message to chat
        st.chat_message("user").write(prompt)
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        
        # Get response
        with st.spinner("Searching documents and generating response..."):
            try:
                result = st.session_state.chatbot.answer_question(prompt, use_ai=use_ai)
                
                # Add assistant response to chat
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
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    create_streamlit_app() 