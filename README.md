# 🏠 HOA Chatbot

An intelligent chatbot that can answer questions about your HOA bylaws and CC&Rs using AI-powered document analysis.

## Features

- **PDF Processing**: Automatically extracts and processes text from HOA documents
- **Semantic Search**: Finds relevant sections using advanced text embeddings
- **AI-Powered Responses**: Generates accurate answers based on your specific HOA documents
- **Web Interface**: Beautiful Streamlit interface for easy interaction
- **Source Tracking**: Shows which documents were referenced for each answer

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Process Your PDF Documents

```bash
python src/pdf_processor.py
```

This will:
- Extract text from your PDF files
- Clean and structure the content
- Save processed data to `data/processed_documents.json`

### 4. Launch the Chatbot

```bash
streamlit run src/chatbot.py
```

The chatbot will open in your browser at `http://localhost:8501`

## Project Structure

```
HOA Chatbot/
├── 📄 Amended and Restated Bylaws - Signed (1).pdf
├── 📄 Amended and Restated CC&Rs - Recorded 5-1-2025 (1).pdf
├── src/
│   ├── pdf_processor.py     # PDF text extraction and processing
│   ├── chatbot.py          # Main chatbot with AI integration
│   └── utils.py            # Helper functions
├── data/                   # Processed document storage
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## How It Works

### 1. Document Processing
- **PDF Extraction**: Uses `pdfplumber` to extract text from your HOA documents
- **Text Cleaning**: Removes formatting artifacts and normalizes text
- **Section Splitting**: Breaks documents into manageable chunks for analysis

### 2. Semantic Search
- **Embeddings**: Creates vector representations of document sections using `sentence-transformers`
- **Similarity Matching**: Finds the most relevant sections for any question
- **Context Retrieval**: Provides relevant document sections to the AI

### 3. AI Response Generation
- **OpenAI Integration**: Uses GPT-3.5-turbo for natural language responses
- **Context-Aware**: Answers are based on your specific HOA documents
- **Source Attribution**: Shows which documents were referenced

## Example Questions

The chatbot can answer questions like:

- "What are the parking rules in our community?"
- "How much are the monthly HOA fees?"
- "Can I have pets in my unit?"
- "What modifications can I make to my property?"
- "How do I submit a complaint?"
- "Are there restrictions on renting my property?"
- "What are the quiet hours?"
- "How do I contact the HOA board?"

## Configuration

### Customizing the Chatbot

You can modify the chatbot behavior by editing `src/chatbot.py`:

- **Model Selection**: Change the OpenAI model (default: `gpt-3.5-turbo`)
- **Response Length**: Adjust `max_tokens` parameter
- **Temperature**: Control response creativity (default: `0.3`)
- **Search Results**: Change `top_k` for number of relevant sections

### Adding More Documents

Simply add new PDF files to the project directory and re-run:

```bash
python src/pdf_processor.py
```

The chatbot will automatically include the new documents in its knowledge base.

## Troubleshooting

### Common Issues

1. **"Data file not found"**
   - Run `python src/pdf_processor.py` first to process your PDFs

2. **"OpenAI API key not found"**
   - Make sure you have a `.env` file with your OpenAI API key

3. **PDF processing errors**
   - Ensure PDFs are not password-protected
   - Check that PDFs contain extractable text (not just images)

4. **Memory issues with large documents**
   - The system automatically chunks large documents
   - If issues persist, reduce `max_chunk_size` in `pdf_processor.py`

### Performance Tips

- **Faster Processing**: Use smaller chunk sizes for quicker embedding generation
- **Better Accuracy**: Increase `top_k` for more comprehensive search results
- **Cost Optimization**: Use `gpt-3.5-turbo` instead of `gpt-4` for lower costs

## API Usage

You can also use the chatbot programmatically:

```python
from src.chatbot import HOAChatbot

# Initialize chatbot
chatbot = HOAChatbot()

# Ask a question
result = chatbot.answer_question("What are the parking rules?")
print(result['answer'])
```

## Contributing

Feel free to enhance the chatbot by:

- Adding new document processing capabilities
- Improving the semantic search algorithm
- Enhancing the web interface
- Adding support for more file formats

## License

This project is open source and available under the MIT License.

## Support

If you encounter any issues or have questions, please:

1. Check the troubleshooting section above
2. Review the error messages in the console
3. Ensure all dependencies are properly installed
4. Verify your OpenAI API key is valid and has sufficient credits

---

**Happy HOA Chatting! 🏠✨** 