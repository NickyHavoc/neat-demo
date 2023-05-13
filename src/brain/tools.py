from typing import Optional
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun
from pydantic import BaseModel, Field

from documents import DocumentMinion


class DocumentSearchTool(BaseTool):
    document_minion: DocumentMinion
    name = "Document Search"
    description = "useful for retrieving documents from the database"

    @classmethod
    def from_document_minion(cls, document_minion: DocumentMinion):
        return cls(
            document_minion=document_minion,
            name=document_minion.name,
            description=document_minion.description
        )

    def _run(self, query: str) -> str:
        results =self. document_minion.search(query)
        if bool(results):
            return results[0].best_chunks[0][0].content
        return "No answer could be found."
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
