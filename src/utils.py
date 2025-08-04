import os
import json
from typing import Dict, List, Optional
import re

def validate_pdf_files(pdf_dir: str = ".") -> List[str]:
    """Validate that PDF files exist and are readable."""
    pdf_files = []
    for file in os.listdir(pdf_dir):
        if file.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, file)
            if os.path.isfile(pdf_path) and os.access(pdf_path, os.R_OK):
                pdf_files.append(file)
            else:
                print(f"⚠️  Warning: Cannot read {file}")
    
    return pdf_files

def create_sample_questions() -> List[str]:
    """Create sample questions for testing the chatbot."""
    return [
        "What are the rules about parking in the community?",
        "How much are the monthly HOA fees?",
        "What are the pet restrictions?",
        "Can I make modifications to my property?",
        "What is the process for submitting a complaint?",
        "Are there any restrictions on renting my property?",
        "What are the quiet hours in the community?",
        "How do I contact the HOA board?",
        "What are the landscaping requirements?",
        "Are there any restrictions on holiday decorations?"
    ]

def extract_key_sections(text: str) -> Dict[str, str]:
    """Extract key sections from HOA documents."""
    sections = {}
    
    # Common HOA section patterns
    patterns = {
        'parking': r'(?i)(parking|vehicle|car|garage).*?(?=\n[A-Z]|$)',
        'fees': r'(?i)(fee|assessment|dues|payment).*?(?=\n[A-Z]|$)',
        'pets': r'(?i)(pet|animal|dog|cat).*?(?=\n[A-Z]|$)',
        'modifications': r'(?i)(modification|alteration|improvement|change).*?(?=\n[A-Z]|$)',
        'complaints': r'(?i)(complaint|grievance|dispute|violation).*?(?=\n[A-Z]|$)',
        'renting': r'(?i)(rent|lease|tenant|rental).*?(?=\n[A-Z]|$)',
        'quiet_hours': r'(?i)(quiet|noise|hours|time).*?(?=\n[A-Z]|$)',
        'contact': r'(?i)(contact|board|officer|management).*?(?=\n[A-Z]|$)',
        'landscaping': r'(?i)(landscape|yard|garden|maintenance).*?(?=\n[A-Z]|$)',
        'decorations': r'(?i)(decoration|holiday|seasonal|display).*?(?=\n[A-Z]|$)'
    }
    
    for section_name, pattern in patterns.items():
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            sections[section_name] = ' '.join(matches)
    
    return sections

def format_response_for_display(response: Dict) -> str:
    """Format chatbot response for better display."""
    formatted = f"**Question:** {response['question']}\n\n"
    formatted += f"**Answer:** {response['answer']}\n\n"
    
    if response.get('sources'):
        formatted += "**Sources:**\n"
        for i, source in enumerate(response['sources'], 1):
            formatted += f"{i}. {source['document']}\n"
    
    return formatted

def save_chat_history(history: List[Dict], filename: str = "data/chat_history.json"):
    """Save chat history to file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Chat history saved to {filename}")

def load_chat_history(filename: str = "data/chat_history.json") -> List[Dict]:
    """Load chat history from file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def get_document_preview(doc_data: Dict, max_length: int = 500) -> str:
    """Get a preview of document content."""
    text = doc_data.get('cleaned_text', '')
    if len(text) <= max_length:
        return text
    
    # Try to find a good break point
    break_point = text.rfind('.', 0, max_length)
    if break_point == -1:
        break_point = text.rfind(' ', 0, max_length)
    
    if break_point > 0:
        return text[:break_point] + "..."
    else:
        return text[:max_length] + "..."

def analyze_document_complexity(text: str) -> Dict:
    """Analyze the complexity of a document."""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    paragraphs = text.split('\n\n')
    
    # Remove empty elements
    sentences = [s.strip() for s in sentences if s.strip()]
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0
    
    return {
        'total_words': len(words),
        'total_sentences': len(sentences),
        'total_paragraphs': len(paragraphs),
        'avg_sentence_length': round(avg_sentence_length, 1),
        'avg_paragraph_length': round(avg_paragraph_length, 1),
        'reading_level': 'Complex' if avg_sentence_length > 20 else 'Moderate' if avg_sentence_length > 15 else 'Simple'
    } 