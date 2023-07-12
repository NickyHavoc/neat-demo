"""
Documentation for the neat_demo module.
"""

# Load .env values into environment variables
# Environment variables that are already set will not be overwritten
from .llm_wrapper import LLMWrapper
from dotenv import load_dotenv

load_dotenv()

# Expose the module's objects
