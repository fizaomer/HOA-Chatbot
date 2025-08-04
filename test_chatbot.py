#!/usr/bin/env python3
"""
Simple command-line test for the HOA Chatbot
"""

import os
import sys
from src.chatbot import HOAChatbot

def main():
    print("🏠 HOA Chatbot - Command Line Test")
    print("=" * 50)
    
    try:
        # Initialize chatbot
        print("Loading chatbot...")
        chatbot = HOAChatbot()
        
        if not chatbot.documents:
            print("❌ No documents loaded. Please run pdf_processor.py first.")
            return
        
        print(f"✅ Chatbot loaded with {len(chatbot.documents)} document(s)")
        
        # Show available documents
        print("\n📚 Available Documents:")
        for doc_name in chatbot.get_available_documents():
            stats = chatbot.get_document_stats()[doc_name]
            print(f"  - {doc_name}")
            print(f"    Sections: {stats['sections']}")
            print(f"    Characters: {stats['total_chars']:,}")
        
        # Test questions
        test_questions = [
            "What are the voting rights for members?",
            "How many directors are on the board?",
            "What are the qualifications for being a director?",
            "How often are board meetings held?",
            "What are the duties of the president?"
        ]
        
        print("\n🤖 Testing Chatbot with Sample Questions:")
        print("=" * 50)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Question: {question}")
            print("-" * 40)
            
            try:
                result = chatbot.answer_question(question)
                print(f"Answer: {result['answer']}")
                
                if result.get('sources'):
                    print("\nSources:")
                    for j, source in enumerate(result['sources'], 1):
                        print(f"  {j}. {source['document']}")
                        
            except Exception as e:
                print(f"Error: {e}")
            
            print()
        
        print("✅ Chatbot test completed successfully!")
        print("\n💡 The web interface should be available at: http://localhost:8501")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have:")
        print("1. Run 'python src/pdf_processor.py' to process documents")
        print("2. Set up your OpenAI API key in the .env file")

if __name__ == "__main__":
    main() 