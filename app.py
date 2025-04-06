import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import glob
import re

# Set the page title and header
st.set_page_config(page_title="Document Q&A Bot", layout="wide")
st.title("Document Q&A Bot")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation" not in st.session_state:
    st.session_state.conversation = None

if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False

# API key handling
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Sidebar for documents
st.sidebar.header("Document Settings")
documents_path = st.sidebar.text_input("Documents Directory Path", value="./documents")
process_docs_button = st.sidebar.button("Process Documents")

# Simple functions to read different file types
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        return file.read()

def read_pdf_file(file_path):
    try:
        import pypdf
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
            return text
    except Exception as e:
        st.sidebar.error(f"Error reading PDF: {str(e)}")
        return ""

# Function to load and process documents
def process_documents(directory_path):
    try:
        documents = []
        
        # Get all files in the directory
        all_files = []
        for ext in ["*.txt", "*.pdf"]:
            all_files.extend(glob.glob(os.path.join(directory_path, ext)))
        
        if not all_files:
            st.sidebar.error(f"No supported documents found in {directory_path}")
            return None
        
        st.sidebar.info(f"Found {len(all_files)} document(s) to process")
        
        # Process each file
        for file_path in all_files:
            try:
                file_extension = os.path.splitext(file_path)[1].lower()
                
                if file_extension == '.txt':
                    content = read_text_file(file_path)
                elif file_extension == '.pdf':
                    content = read_pdf_file(file_path)
                else:
                    st.sidebar.warning(f"Skipping unsupported file: {file_path}")
                    continue
                
                # Only add documents with content
                if content and content.strip():
                    file_name = os.path.basename(file_path)
                    documents.append({
                        "page_content": content,
                        "metadata": {"source": file_name}
                    })
            except Exception as e:
                st.sidebar.error(f"Error processing file {file_path}: {str(e)}")
        
        if not documents:
            st.sidebar.error("No document content could be extracted")
            return None
        
        st.sidebar.success(f"Successfully processed {len(documents)} document(s)")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Convert our documents to the format expected by text_splitter
        langchain_documents = []
        for doc in documents:
            langchain_documents.append({
                "page_content": doc["page_content"],
                "metadata": doc["metadata"]
            })
            
        chunks = text_splitter.create_documents(
            [doc["page_content"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents]
        )
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        st.sidebar.success("Documents processed and indexed successfully!")
        return vectorstore
    
    except Exception as e:
        st.sidebar.error(f"Error processing documents: {str(e)}")
        return None

# Process the documents when the button is clicked
if process_docs_button:
    if not openai_api_key:
        st.sidebar.error("Please enter your OpenAI API key")
    else:
        with st.sidebar.status("Processing documents..."):
            vectorstore = process_documents(documents_path)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.docs_processed = True

# Create the ConversationalRetrievalChain when documents are processed
if st.session_state.docs_processed and "vectorstore" in st.session_state:
    # Define custom prompts
    qa_template = """
    You are a helpful assistant that can only answer questions based on the provided context.
    If the question cannot be answered using the information provided, respond with:
    "I cannot answer this question as it's not related to the documents I have access to."
    Do not make up or infer information that is not explicitly stated in the context.

    Context: {context}

    Question: {question}

    Answer:
    """
    QA_PROMPT = PromptTemplate(
        template=qa_template, 
        input_variables=["context", "question"]
    )

    # Create the chain
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    st.session_state.conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )

# Display the chat interface
st.subheader("Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
if user_input := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Check if documents have been processed
    if not st.session_state.docs_processed:
        response = "Please process your documents first using the sidebar."
    elif not openai_api_key:
        response = "Please enter your OpenAI API key in the sidebar."
    else:
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.conversation(
                    {"question": user_input, "chat_history": [(m["content"], m["content"]) for m in st.session_state.messages if m["role"] == "assistant"]}
                )
                response = result["answer"]
                st.write(response)
                
                # Display sources (optional)
                if "source_documents" in result and result["source_documents"]:
                    with st.expander("Sources"):
                        for i, doc in enumerate(result["source_documents"]):
                            st.write(f"Source {i+1}:")
                            st.write(doc.page_content)
                            st.write("---")
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Show instructions and info
if not st.session_state.messages:
    st.info("Enter the path to your documents directory, click 'Process Documents', and then ask questions about your documents.")