#!/usr/bin/env python3

import os
import warnings

# Suppress huggingface_hub warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import the processing functions from our main module
from main import (
    load_bitcoin_emails,
    split_documents,
    create_vectorstore,
    bitcoin_email_query_tool
)

@tool
def bitcoin_email_query(query: str):
    """
    Query the Bitcoin email archives using a retriever.
    """
    print('Tool called with query:', query)
    return bitcoin_email_query_tool.invoke(query)

def test_retrieval():
    """
    Test the retrieval system with a sample query using Ollama
    """
    # Path to the vector store
    vectorstore_path = "bitcoin_emails_vectorstore.parquet.index"
    
    # Check if vector store exists, if not, create it
    if not os.path.exists(vectorstore_path):
        # Path to the Bitcoin repository
        repo_path = input("Enter path to bitcoin_repo directory: ")
        
        # Load and process documents
        documents = load_bitcoin_emails(repo_path)
        split_docs = split_documents(documents)
        create_vectorstore(split_docs, vectorstore_path)
    
    # Test query
    test_query = "p2p authentication and encryption"
    print(f"\nTesting query: '{test_query}'")
    results = bitcoin_email_query(test_query)
    print("\nRESULTS:")
    print(results)
    
    # Test with Ollama
    print("\nTesting with Ollama LLM...")
    
    # Replace with your preferred Ollama model
    # Common options: llama3, mistral, codellama, etc.
    ollama_model = "llama3"
    
    # Create the Ollama LLM instance
    llm = ChatOllama(model=ollama_model)
    
    # Create the system prompt that includes instructions and tool usage
    system_template = """You are a helpful assistant that can answer questions about Bitcoin development discussions 
    from the mailing list archives. Use the retrieved information to answer the user's questions.
    If you don't know the answer, say "I don't know."

    Here is relevant information from the Bitcoin email archives:
    {context}
    """
    
    user_template = "{question}"
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", user_template)
    ])
    
    # Create the RAG chain
    chain = (
        {"context": bitcoin_email_query, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Test the chain with a sample query
    query = "What did Jonas Schnelli propose about p2p authentication and encryption?"
    print("\nUser Query:", query)
    
    response = chain.invoke(query)
    print("\nOLLAMA RESPONSE:")
    print(response)

def interactive_query():
    """
    Allow the user to interactively query the Bitcoin email archives using Ollama
    """
    # Path to the vector store
    vectorstore_path = "bitcoin_emails_vectorstore.parquet.index"
    
    # Check if vector store exists, if not, create it
    if not os.path.exists(vectorstore_path):
        # Path to the Bitcoin repository
        repo_path = input("Enter path to bitcoin_repo directory: ")
        
        # Load and process documents
        documents = load_bitcoin_emails(repo_path)
        split_docs = split_documents(documents)
        create_vectorstore(split_docs, vectorstore_path)
    
    # Create the Ollama LLM instance
    # Replace with your preferred Ollama model
    ollama_model = input("Enter Ollama model to use (default: mistral-nemo:latest): ") or "mistral-nemo:latest"
    llm = ChatOllama(model=ollama_model)
    
    # Create the system prompt with instructions
    system_template = """You are a helpful assistant that can answer questions about Bitcoin development discussions 
    from the mailing list archives. Use the retrieved information to answer the user's questions.
    If you don't know the answer, say "I don't know."

    Here is relevant information from the Bitcoin email archives:
    {context}
    """
    
    user_template = "{question}"
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", user_template)
    ])
    
    # Create the RAG chain
    chain = (
        {"context": bitcoin_email_query, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print(f"\nBitcoin Email Query System with Ollama ({ollama_model})")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            break
            
        print("\nRetrieving information...")
        response = chain.invoke(query)
        print("\nResponse:")
        print(response)

if __name__ == "__main__":
    # Uncomment the function you want to run
    # test_retrieval()
    interactive_query()




