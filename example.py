from pathlib import Path

from neat_ai_assistant.documents.documents import DocumentMinion
from neat_ai_assistant.agent.agent import NeatAgent
from neat_ai_assistant.agent.conversation_history import ConversationHistory
from neat_ai_assistant.agent.tools import DocumentSearchTool, DuckDuckGoSearchTool, RetrieveConversationHistoryTool, WeatherRetrievalTool, StockTradingTool


documents_path = Path(__file__).parent / "example_docs"
doc_minion = DocumentMinion()
doc_minion.instantiate_database(
    documents_path=documents_path,
    update=False
)
history = ConversationHistory()
tool = WeatherRetrievalTool()
tool.run(json_query={"location": "Heidelberg"})
tools = [
    DuckDuckGoSearchTool(),
    DocumentSearchTool.from_document_minion(
        document_minion=doc_minion
    ),
    RetrieveConversationHistoryTool(
        history=history
    ),
    # WeatherRetrievalTool(),
    StockTradingTool()
]
agent = NeatAgent(tools=tools, history=history)

while True:
    user_message = input("TYPE YOUR INPUT: ")
    for r in agent.reply_to(message_string=user_message):
        print(r)

# TODO:
# OpenAI may return shit like this:

# {
# "assistant to=functions.retrieve_conversation_history)": {
# "n": 3,
# "reasoning": "The user's question is lacking in complete information. There's no mention of what stocks to compare. I need previous context to understand it better."
# }}
# As an AI, I apologize for the ambiguity I faced with your previous message. May I kindly ask you to specify the two companies' stock prices you wish to compare.
# As soon as I get the necessary information from you, I'll happily
# provide you with a comparison between the two stock prices.

# Where the json is, for some reason, inside of "content" rather than
# "function_call"
