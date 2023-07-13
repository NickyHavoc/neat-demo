from typing import List
import gradio as gr
from pathlib import Path

from neat_ai_assistant.documents.documents import DocumentMinion
from neat_ai_assistant.agent.agent import NeatAgent
from neat_ai_assistant.agent.tools import DocumentSearchTool, DuckDuckGoSearchTool


documents_path = Path(__file__).parent / "example_docs"
doc_minion = DocumentMinion()
doc_minion.instantiate_database(
    documents_path=documents_path,
    update=True
)
tools = [
    DuckDuckGoSearchTool(),
    DocumentSearchTool.from_document_minion(
        document_minion=doc_minion
    )
]
brain = NeatAgent(tools=tools)


def color_text(result):
    color_dict = {
        "thought": "blue",
        "function_call": "green",
        "answer": "red",
        "input": "purple"
    }

    colored_text = f"<p style='color: {color_dict[result.type]}'>{result.text}</p>"
    return colored_text


def chat_interface(inputs):
    chat_counter = 0
    history = []
    token_counter = 0

    for result in brain.reply_to(message_string=inputs):
        colored_result = color_text(result)
        history.append(colored_result)
        chat = [(history[i], history[i + 1]) for i in range(0, len(history) - 1, 2)]
        token_counter += 1
        yield chat, history, chat_counter


iface = gr.Interface(fn=chat_interface, inputs="text", outputs="html")
iface.launch()

