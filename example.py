from pathlib import Path
from neat_demo import AgentHangout


documents_dir = Path(__file__).parent / "example_documents"
agent_hangout = AgentHangout(documents_dir)
print(agent_hangout.get_entry_message())

print("")
