#!/usr/bin/env python3

import os
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

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
    return bitcoin_email_query_tool(query)

def test_retrieval():
    """
    Test the retrieval system with a sample query
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
    
    # Test with LLM
    # print("\nTesting with LLM...")
    
    # llm = ChatAnthropic(model="claude-3-7-sonnet-latest", temperature=0,anthropic_api_key="sk-ant-api03-apikey")
    # augmented_llm = llm.bind_tools([bitcoin_email_query])
    
    # instructions = """You are a helpful assistant that can answer questions about Bitcoin development discussions 
    # from the mailing list archives. Use the bitcoin_email_query tool for any questions about Bitcoin development.
    # If you don't know the answer, say "I don't know."""
    
    # messages = [
    #     {"role": "system", "content": instructions},
    #     {"role": "user", "content": "What did Jonas Schnelli propose about p2p authentication and encryption?"}
    # ]
    
    # message = augmented_llm.invoke(messages)
    # print("\nLLM RESPONSE:")
    # print(message.content)

if __name__ == "__main__":
    test_retrieval()