# Document Q&A RAG Chatbot

This application uses Retrieval-Augmented Generation (RAG) to answer questions based on your documents. It will only respond to questions that are related to the content in your documents.

## Project Structure

```
project/
├── app.py                  # Main application file
├── requirements.txt        # Dependencies
├── documents/              # Directory for your documents
└── README.md               # Project documentation
```

## Setup Instructions

### 1. Prerequisites

- Python 3.9+ installed
- Visual Studio Code
- OpenAI API key

### 2. Local Development

1. Create a new directory for your project:
   ```bash
   mkdir rag-chatbot
   cd rag-chatbot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `documents` directory:
   ```bash
   mkdir documents
   ```

5. Add your document files (PDFs, TXTs, etc.) to the `documents` directory.

6. Run the application:
   ```bash
   streamlit run app.py
   ```

### 3. Using the Application

1. Enter your OpenAI API key in the sidebar.
2. Confirm the documents directory path (default is `./documents`).
3. Click "Process Documents" to index your documents.
4. Ask questions related to your documents in the chat interface.

### 4. Deployment to Streamlit Community Cloud

1. Create a GitHub repository and push your code:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and sign in.

3. Click "New app" and select your GitHub repository.

4. Set the main file path to `app.py`.

5. Add your OpenAI API key as a secret:
   - Name: `OPENAI_API_KEY`
   - Value: Your OpenAI API key

6. Click "Deploy" to deploy your application.

## Notes

- The application only answers questions related to the provided documents.
- Questions outside the scope of your documents will be rejected.
- Make sure your documents are in a format that can be processed by LangChain (PDF, TXT, DOCX, etc.).
- For large document collections, processing may take some time.