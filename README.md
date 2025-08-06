# 🏠 Turtle Rock Crest HOA Assistant

A chatbot that helps residents find answers in their HOA documents quickly and easily.

## What it does

- **Processes PDFs**: Extracts text from HOA bylaws, CC&Rs, and other documents
- **Smart search**: Finds relevant sections when you ask questions
- **AI-powered answers**: Uses Google's Gemini to give natural responses
- **Web interface**: Clean, easy-to-use chat interface
- **Source tracking**: Shows you exactly which documents were used

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Process Your Documents

```bash
python src/pdf_processor.py
```

This processes your PDF files and saves the extracted data to `data/processed_documents.json`.

### 4. Run the Assistant

```bash
streamlit run src/chatbot_with_gemini.py
```

Open your browser to `http://localhost:8501` to start chatting.

## Deploying to GitHub

### Initial Setup

1. **Initialize Git repository** (if not already done):
```bash
git init
```

2. **Add your files**:
```bash
git add .
```

3. **Make your first commit**:
```bash
git commit -m "Initial commit: HOA Assistant"
```

4. **Create a new repository on GitHub**:
   - Go to github.com and click "New repository"
   - Name it something like "hoa-assistant" or "turtle-rock-chatbot"
   - Don't initialize with README (we already have one)

5. **Connect and push to GitHub**:
```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### Important Notes

- The `.env` file is already in `.gitignore` so your API keys won't be uploaded
- PDF files are also ignored to keep the repository size manageable
- Only the code and processed data will be pushed to GitHub

### Updating Your Repository

When you make changes:

```bash
git add .
git commit -m "Description of your changes"
git push
```

## Project Structure

```
HOA Chatbot/
├── src/
│   ├── pdf_processor.py     # PDF text extraction
│   ├── chatbot_with_gemini.py  # Main chatbot app
│   └── utils.py            # Helper functions
├── data/                   # Processed documents
├── requirements.txt        # Dependencies
├── .env                   # API keys (not in git)
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## How it works

### Document Processing
- **Text extraction**: Uses pdfplumber to pull text from PDFs
- **OCR support**: Can handle scanned documents using Tesseract
- **Text cleaning**: Removes formatting issues and normalizes text
- **Chunking**: Breaks documents into searchable sections

### Search & AI
- **Embeddings**: Creates searchable versions of document sections
- **Semantic search**: Finds relevant content based on meaning, not just keywords
- **Gemini AI**: Uses Google's model to generate natural responses
- **Source tracking**: Shows which documents were used for each answer

## Example Questions

Try asking about:

- "What are the parking rules?"
- "How much are the monthly fees?"
- "Can I have pets?"
- "What can I modify on my property?"
- "How do I submit a complaint?"
- "Can I rent my place out?"
- "What are the quiet hours?"
- "How do I contact the board?"

## Configuration

### Customizing the Assistant

Edit `src/chatbot_with_gemini.py` to change:

- **AI model**: Switch between different Gemini models
- **Response length**: Adjust how detailed answers are
- **Search results**: Change how many document sections to use
- **UI elements**: Modify the web interface

### Adding New Documents

Just drop new PDF files in the project folder and run:

```bash
python src/pdf_processor.py
```

The assistant will automatically include them.

## Troubleshooting

### Common Problems

1. **"Data file not found"**
   - Run the PDF processor first: `python src/pdf_processor.py`

2. **"API key not found"**
   - Check that your `.env` file has the correct GOOGLE_API_KEY

3. **PDF processing fails**
   - Make sure PDFs aren't password-protected
   - For scanned PDFs, install Tesseract: `brew install tesseract`

4. **Memory issues**
   - Large documents are automatically chunked
   - Reduce chunk size in `pdf_processor.py` if needed

### Performance Tips

- **Faster processing**: Use smaller chunk sizes
- **Better accuracy**: Increase the number of search results
- **Cost savings**: Gemini is free for reasonable usage

## Using the Code

You can also use the chatbot in your own scripts:

```python
from src.chatbot_with_gemini import HOAChatbot

chatbot = HOAChatbot()
result = chatbot.answer_question("What are the parking rules?")
print(result['answer'])
```

## Contributing

Feel free to improve the assistant by:

- Adding support for more document types
- Improving the search algorithm
- Enhancing the web interface
- Adding new features

## License

MIT License - feel free to use and modify.

## Support

If you run into issues:

1. Check the troubleshooting section above
2. Look at the console output for error messages
3. Make sure all dependencies are installed
4. Verify your API key is working

---

**Built for Turtle Rock Crest residents** 🏠 