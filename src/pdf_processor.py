import pdfplumber
import os
import json
from typing import Dict, List, Tuple
import re

# OCR imports
try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: OCR dependencies not installed. Scanned PDFs cannot be processed.")
    print("Install with: pip install pytesseract Pillow pdf2image")

class HOAPDFProcessor:
    def __init__(self, pdf_dir: str = "."):
        self.pdf_dir = pdf_dir
        self.processed_data = {}
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file using pdfplumber, with OCR fallback for scanned PDFs."""
        # First try normal text extraction
        text = self._extract_text_normal(pdf_path)
        
        # If no text extracted and OCR is available, try OCR
        if not text and OCR_AVAILABLE:
            print(f"  No text found, attempting OCR for {os.path.basename(pdf_path)}...")
            text = self._extract_text_with_ocr(pdf_path)
        
        return text
    
    def _extract_text_normal(self, pdf_path: str) -> str:
        """Extract text using pdfplumber (for text-based PDFs)."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                total_pages = len(pdf.pages)
                print(f"  Processing {total_pages} pages...")
                
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                        
                        # Progress indicator for large documents
                        if total_pages > 10 and (i + 1) % 10 == 0:
                            print(f"    Processed {i + 1}/{total_pages} pages...")
                            
                    except Exception as e:
                        print(f"    Warning: Error processing page {i + 1}: {e}")
                        continue
                        
                return text
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return ""
    
    def _extract_text_with_ocr(self, pdf_path: str) -> str:
        """Extract text using OCR (for scanned PDFs)."""
        try:
            # Set Tesseract path for macOS Homebrew installation
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
            
            # Convert PDF to images with poppler path
            print("    Converting PDF to images...")
            images = convert_from_path(
                pdf_path, 
                dpi=300,  # Higher DPI for better OCR accuracy
                poppler_path='/opt/homebrew/bin'  # Specify poppler path
            )
            
            text = ""
            total_pages = len(images)
            print(f"    Running OCR on {total_pages} pages...")
            
            for i, image in enumerate(images):
                try:
                    # Extract text using OCR
                    page_text = pytesseract.image_to_string(image)
                    if page_text.strip():
                        text += page_text + "\n"
                    
                    # Progress indicator
                    if total_pages > 10 and (i + 1) % 10 == 0:
                        print(f"      OCR processed {i + 1}/{total_pages} pages...")
                        
                except Exception as e:
                    print(f"      Warning: Error processing page {i + 1} with OCR: {e}")
                    continue
            
            return text
            
        except Exception as e:
            print(f"Error during OCR processing: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and headers
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)  # Remove standalone page numbers
        # Clean up line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()
    
    def split_into_sections(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Split text into manageable chunks for processing."""
        # First, try to split by article headers
        article_splits = re.split(r'\n(ARTICLE [IVX]+)', text)
        
        sections = []
        current_section = ""
        
        for i, part in enumerate(article_splits):
            if i == 0:  # First part (before any article)
                current_section = part
            else:
                # This is an article header or content
                if part.strip().startswith('ARTICLE'):
                    # Save previous section if it has content
                    if current_section.strip():
                        sections.extend(self._split_large_section(current_section, max_chunk_size))
                    current_section = part
                else:
                    current_section += part
        
        # Add the last section
        if current_section.strip():
            sections.extend(self._split_large_section(current_section, max_chunk_size))
        
        # If no articles found, split by other headers
        if len(sections) <= 1:
            sections = self._split_by_headers(text, max_chunk_size)
        
        return sections
    
    def _split_large_section(self, section: str, max_chunk_size: int) -> List[str]:
        """Split a large section into smaller chunks."""
        if len(section) <= max_chunk_size:
            return [section]
        
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', section)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_by_headers(self, text: str, max_chunk_size: int) -> List[str]:
        """Split text by various header patterns."""
        # Split by numbered sections (e.g., "1.1", "2.3", etc.)
        section_splits = re.split(r'\n(\d+\.\d+\.?\s+)', text)
        
        sections = []
        current_section = ""
        
        for i, part in enumerate(section_splits):
            if i == 0:
                current_section = part
            else:
                if re.match(r'\d+\.\d+\.?\s+', part):
                    # Save previous section
                    if current_section.strip():
                        sections.extend(self._split_large_section(current_section, max_chunk_size))
                    current_section = part
                else:
                    current_section += part
        
        if current_section.strip():
            sections.extend(self._split_large_section(current_section, max_chunk_size))
        
        return sections
    
    def process_hoa_documents(self) -> Dict:
        """Process all HOA PDF documents in the directory."""
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_dir, pdf_file)
            print(f"Processing: {pdf_file}")
            
            # Extract text
            raw_text = self.extract_text_from_pdf(pdf_path)
            if not raw_text:
                print(f"  ❌ Failed to extract text from {pdf_file}")
                continue
                
            # Clean text
            cleaned_text = self.clean_text(raw_text)
            
            # Split into sections
            sections = self.split_into_sections(cleaned_text)
            
            # Store processed data
            self.processed_data[pdf_file] = {
                'raw_text': raw_text,
                'cleaned_text': cleaned_text,
                'sections': sections,
                'metadata': {
                    'file_name': pdf_file,
                    'total_sections': len(sections),
                    'total_length': len(cleaned_text),
                    'raw_length': len(raw_text)
                }
            }
            
            print(f"  ✓ Processed {pdf_file}: {len(sections)} sections, {len(cleaned_text):,} characters")
        
        return self.processed_data
    
    def save_processed_data(self, output_file: str = "data/processed_documents.json"):
        """Save processed data to JSON file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved processed data to {output_file}")
    
    def get_document_summary(self) -> Dict:
        """Get a summary of processed documents."""
        summary = {}
        for doc_name, doc_data in self.processed_data.items():
            summary[doc_name] = {
                'sections': doc_data['metadata']['total_sections'],
                'characters': doc_data['metadata']['total_length'],
                'raw_characters': doc_data['metadata']['raw_length'],
                'first_100_chars': doc_data['cleaned_text'][:100] + "..."
            }
        return summary

if __name__ == "__main__":
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Process PDFs
    processor = HOAPDFProcessor()
    processed_data = processor.process_hoa_documents()
    
    # Save results
    processor.save_processed_data()
    
    # Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    summary = processor.get_document_summary()
    for doc_name, info in summary.items():
        print(f"\n📄 {doc_name}")
        print(f"   Sections: {info['sections']}")
        print(f"   Characters: {info['characters']:,}")
        print(f"   Raw Characters: {info['raw_characters']:,}")
        print(f"   Preview: {info['first_100_chars']}") 