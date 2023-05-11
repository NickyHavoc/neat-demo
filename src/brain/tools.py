from typing import Optional
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun
from pydantic import BaseModel, Field

from documents import DocumentHandler


class DocumentSearchTool(BaseTool):
    document_handler: DocumentHandler
    name = "Document Search"
    description = "useful for retrieving documents from the database"

    def _run(self, query: str) -> str:
        results =self. document_handler.search(query)
        if bool(results):
            return results[0].best_chunks[0][0].content
        return "No answer could be found."
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
