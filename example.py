from pathlib import Path
from configurable_chat_agent import DocumentBase, LLMWrapper

wrapper = LLMWrapper()
doc_base = DocumentBase(
    wrapper=wrapper,
    path=Path(__file__).parent / "example_agent_config"
)
search_res = doc_base.score_query(query="I have a question concerning our income statement.")

print("")
