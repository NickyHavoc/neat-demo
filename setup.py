from setuptools import setup, find_packages
from pathlib import Path


def readme():
    with open("README.md") as f:
        return f.read()


def version():
    exec(open("src/version.py").read())
    return locals()["__version__"]


def requirements():
    requirements = Path("requirements.txt").read_text().splitlines()
    return requirements


setup(
    name="configurable_chat_agent",
    url="https://github.com/NiklasFinken/configurable-chat-agent.git",
    author="Niklas Finken",
    author_email="niklasfinken@me.com",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=requirements(),
    description="Repository to configure a GPT 3.5-based chat agent in minutes.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    version=version(),
    # package_data={
    #     # If any package contains *.json files, include them:
    #     "": ["**/*.json"],
    # },
    # include_package_data=True,
)
