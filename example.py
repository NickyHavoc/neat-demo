from pathlib import Path
from configurable_chat_agent import DocumentMinion, LLMWrapper

wrapper = LLMWrapper()
doc_base = DocumentMinion(
    wrapper=wrapper,
    path=Path(__file__).parent / "example_agent_config"
)
search_res = doc_base.score_query(query="I have a question concerning our income statement.")

print("")
