#!/usr/bin/env python3

import os
import re
import io
import tiktoken
from email import policy
from email.parser import BytesParser
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import faiss

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool


def count_tokens(text, model="cl100k_base"):
    """
    Count the number of tokens in the text using tiktoken.
    
    Args:
        text (str): The text to count tokens for
        model (str): The tokenizer model to use (default: cl100k_base for GPT-4)
        
    Returns:
        int: Number of tokens in the text
    """
    encoder = tiktoken.get_encoding(model)
    return len(encoder.encode(text))

def parse_email_file(file_path):
    """
    Parse a raw email file into a structured Document object.
    
    Args:
        file_path (str): Path to the email file
        
    Returns:
        Document: A Document object with the email content and metadata
    """
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
            data = io.BytesIO(data)
            
        # Parse the email data
        msg = BytesParser(policy=policy.default).parse(data)
        
        # Extract the relevant parts
        subject = msg.get('subject', '')
        sender = msg.get('from', '')
        date = msg.get('date', '')
        
        # Handle multipart messages
        if msg.is_multipart():
            content = ""
            for part in msg.iter_parts():
                if part.get_content_type() == 'text/plain':
                    content += part.get_content()
                elif part.get_content_type() == 'text/html':
                    soup = BeautifulSoup(part.get_content(), 'html.parser')
                    content += soup.get_text()
        else:
            # Handle text/html content
            if msg.get_content_type() == 'text/html':
                soup = BeautifulSoup(msg.get_content(), 'html.parser')
                content = soup.get_text()
            else:
                content = msg.get_content()
        
        # Clean up the content
        content = re.sub(r'\n\n+', '\n\n', content).strip()
        
        # Create metadata
        metadata = {
            'source': file_path,
            'subject': subject,
            'sender': sender,
            'date': date
        }
        
        return Document(page_content=content, metadata=metadata)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return Document(
            page_content=f"Error processing file: {e}",
            metadata={'source': file_path, 'error': str(e)}
        )

def load_bitcoin_emails(repository_path):
    """
    Load Bitcoin emails from the repository structure.
    
    Args:
        repository_path (str): Path to the Bitcoin mail repository
        
    Returns:
        list: A list of Document objects containing the email content
    """
    print("Loading Bitcoin mail archives...")
    
    # Get all subdirectories (hex folders)
    hex_folders = [f for f in os.listdir(repository_path) if os.path.isdir(os.path.join(repository_path, f))]
    
    documents = []
    total_files = 0
    processed_files = 0
    
    # Process each hex folder
    for folder in hex_folders:
        folder_path = os.path.join(repository_path, folder)
        email_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        total_files += len(email_files)
        
        for file_name in email_files:
            file_path = os.path.join(folder_path, file_name)
            doc = parse_email_file(file_path)
            if doc:
                documents.append(doc)
                processed_files += 1
    
    print(f"Processed {processed_files} out of {total_files} email files from Bitcoin mail archives.")
    
    # Count total tokens in documents
    total_tokens = 0
    for doc in tqdm(documents):
        tokens = count_tokens(doc.page_content)
        total_tokens += tokens
    
    print(f"Total tokens in loaded documents: {total_tokens}")
    
    return documents

def save_emails_full(documents, output_filename="emails_full.txt"):
    """
    Save the email documents to a file
    
    Args:
        documents (list): List of Document objects to save
        output_filename (str): Name of the output file
    """
    with open(output_filename, "w", encoding="utf-8") as f:
        # Write each document
        for i, doc in enumerate(documents):
            # Get metadata
            source = doc.metadata.get('source', 'Unknown source')
            subject = doc.metadata.get('subject', 'No subject')
            sender = doc.metadata.get('sender', 'Unknown sender')
            date = doc.metadata.get('date', 'Unknown date')
            
            # Write the document with proper formatting
            f.write(f"EMAIL {i+1}\n")
            f.write(f"SOURCE: {source}\n")
            f.write(f"SUBJECT: {subject}\n")
            f.write(f"FROM: {sender}\n")
            f.write(f"DATE: {date}\n")
            f.write("CONTENT:\n")
            f.write(doc.page_content)
            f.write("\n\n" + "="*80 + "\n\n")

    print(f"Documents concatenated into {output_filename}")

def split_documents(documents):
    """
    Split documents into smaller chunks for improved retrieval.
    
    Args:
        documents (list): List of Document objects to split
        
    Returns:
        list: A list of split Document objects
    """
    print("Splitting documents...")
    
    # Initialize text splitter using tiktoken for accurate token counting
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=4000,  # Smaller chunks for email content
        chunk_overlap=200  # Less overlap for emails
    )
    
    # Split documents into chunks
    split_docs = text_splitter.split_documents(documents)
    
    print(f"Created {len(split_docs)} chunks from documents.")
    
    # Count total tokens in split documents
    total_tokens = 0
    for doc in split_docs:
        total_tokens += count_tokens(doc.page_content)
    
    print(f"Total tokens in split documents: {total_tokens}")
    
    return split_docs

def create_vectorstore(splits, persist_path="bitcoin_emails_faiss"):
    """
    Create a FAISS vector store from document chunks using SentenceTransformer.
    
    Args:
        splits (list): List of split Document objects to embed
        persist_path (str): Directory path to save the FAISS index
        
    Returns:
        FAISS: A FAISS vector store containing the embedded documents
    """
    print("Creating FAISS vector store...")

    # Load the embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Prepare texts and metadata
    texts = [doc.page_content for doc in splits]
    metadatas = [doc.metadata for doc in splits]

    # Generate embeddings
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, f"{persist_path}.index")
    print(f"FAISS index saved to {persist_path}.index")

    # Save metadata separately (you can serialize it with pickle or json)
    import pickle
    with open(f"{persist_path}_meta.pkl", "wb") as f:
        pickle.dump(metadatas, f)
    print(f"Metadata saved to {persist_path}_meta.pkl")

    return index

def process_bitcoin_emails(repo_path, output_file="emails_full.txt", vectorstore_path="bitcoin_emails_vectorstore.parquet"):
    """
    Process Bitcoin emails from start to finish.
    
    Args:
        repo_path (str): Path to the Bitcoin mail repository
        output_file (str): Path to save the full email content
        vectorstore_path (str): Path to save the vector store
        
    Returns:
        SKLearnVectorStore: The created vector store
    """
    # Load the documents
    documents = load_bitcoin_emails(repo_path)
    
    # Save the documents to a file
    save_emails_full(documents, output_file)
    
    # Split the documents
    split_docs = split_documents(documents)
    
    # Create the vector store
    vectorstore = create_vectorstore(split_docs, vectorstore_path)
    return vectorstore

@tool
def bitcoin_email_query_tool(query: str):
    """
    Query the Bitcoin email archives using a FAISS retriever.
    
    Args:
        query (str): The query to search the email archives with

    Returns:
        str: A formatted string of the retrieved email documents
    """
    import faiss
    import pickle
    import numpy as np
    from sentence_transformers import SentenceTransformer
    
    # Paths may need to be adjusted based on where the script is run
    current_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(current_dir, "bitcoin_emails_vectorstore.parquet.index")
    meta_path = os.path.join(current_dir, "bitcoin_emails_vectorstore.parquet_meta.pkl")
    
    # Load the embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load the FAISS index
    index = faiss.read_index(index_path)
    
    # Load the metadata
    with open(meta_path, "rb") as f:
        metadatas = pickle.load(f)
    
    # Encode the query
    query_vector = embedding_model.encode([query], convert_to_numpy=True)
    
    # Perform the search (k=3 means return top 3 matches)
    k = 3
    distances, indices = index.search(query_vector, k)
    
    print(f"Retrieved {len(indices[0])} relevant email documents")
    
    # Format the results
    formatted_context = ""
    for i, idx in enumerate(indices[0]):
        # Get the metadata for this document
        metadata = metadatas[idx]
        
        subject = metadata.get('subject', 'No subject')
        sender = metadata.get('sender', 'Unknown sender')
        date = metadata.get('date', 'Unknown date')
        source = metadata.get('source', 'Unknown source')
        
        try:
            with open(source, 'rb') as f:
                data = f.read()
                data = io.BytesIO(data)
                
            # Parse the email data
            msg = BytesParser(policy=policy.default).parse(data)
            
            # Extract content
            if msg.is_multipart():
                content = ""
                for part in msg.iter_parts():
                    if part.get_content_type() == 'text/plain':
                        content += part.get_content()
                    elif part.get_content_type() == 'text/html':
                        soup = BeautifulSoup(part.get_content(), 'html.parser')
                        content += soup.get_text()
            else:
                if msg.get_content_type() == 'text/html':
                    soup = BeautifulSoup(msg.get_content(), 'html.parser')
                    content = soup.get_text()
                else:
                    content = msg.get_content()
            
            # Clean up the content
            content = re.sub(r'\n\n+', '\n\n', content).strip()
        except Exception as e:
            content = f"Error retrieving content: {e}"
        
        formatted_context += f"==EMAIL {i+1}==\n"
        formatted_context += f"SUBJECT: {subject}\n"
        formatted_context += f"FROM: {sender}\n"
        formatted_context += f"DATE: {date}\n"
        formatted_context += f"DISTANCE: {distances[0][i]:.4f}\n"  # Add relevance score
        formatted_context += f"CONTENT:\n{content}\n\n"
    
    return formatted_context

def main(repo_path, output_file, vectorstore_path):
    """
    Main function to process Bitcoin emails and set up the MCP server.
    """
    # Process Bitcoin emails if needed
    if not os.path.exists(vectorstore_path):
        process_bitcoin_emails(repo_path, output_file, vectorstore_path)
    print("generated vectorstore")
    

if __name__ == "__main__":
    # Set paths
    repo_path = "/home/shashank/Desktop/SOB/MailRAG/data/bitcoin_resources"
    output_file = "emails_full.txt"
    vectorstore_path = "bitcoin_emails_vectorstore.parquet"
    
    main(repo_path, output_file, vectorstore_path)