from mcp.server.fastmcp import FastMCP
from main import bitcoin_email_query_tool
import os
import io

# Create the MCP server
mcp = FastMCP("Bitcoin-Mail-Archives-MCP-Server")

# Add the Bitcoin email query tool
@mcp.tool()
def bitcoin_email_search(query: str):
    """
    Search the Bitcoin email archives for relevant information.
    
    Args:
        query (str): The query to search the email archives with

    Returns:
        str: A formatted string of the retrieved email documents
    """
    return bitcoin_email_query_tool(query)

# Add a resource for the full email archive
@mcp.resource("docs://bitcoin/mail-archives")
def get_all_bitcoin_emails(output_file) -> str:
    """
    Get all the Bitcoin email archives. Returns the contents of the emails_full.txt file,
    which contains a curated set of Bitcoin mailing list archives. This is useful
    for a comprehensive response to questions about Bitcoin development.

    Args: None

    Returns:
        str: The contents of the Bitcoin email archives
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        doc_path = os.path.join(current_dir, output_file)
        with open(doc_path, 'r', encoding='utf-8') as file:
            return io.BytesIO(file.read())
    except Exception as e:
        return f"Error reading email archive file: {str(e)}"

# Run the server
mcp.run(transport='stdio')