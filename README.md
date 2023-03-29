# Configurable Chat Agent

Repository with quickly configurable chat agent on your own data.


# Installation

This codebase can be installed as git clone and editable intall or as pip package.
**Please git clone and install editable for training or running tests.**

## Use a conda environment

An own environment is advisable so that dependencies are not mixed with other projects.

When conda is installed and loaded you can use the following commands to create a new conda environment.

```console
conda create -n configurable_chat_agent python=3.9
y
conda activate configurable_chat_agent
```

## Install editable with ssh

```console
git clone git@github.com:NiklasFinken/configurable-chat-agent.git
cd configurable_chat_agent
pip install -e .
```

## Install editable with https

```console
git clone https://github.com/NiklasFinken/configurable-chat-agent.git
cd configurable_chat_agent
pip install -e .
```

## Install as package with ssh

```console
pip install --upgrade git@github.com:NiklasFinken/configurable-chat-agent.git
```

## Install as package with https

```console
pip install --upgrade git+https://github.com/NiklasFinken/configurable-chat-agent.git
```

## Create your .env file

In order to set credentials to the api that should never be pushed to a repository, copy the file ".env_example", rename the copy to ".env" and paste your personal API token.


# Run the server & frontend

Once you installed the necessary python packages, including this module, you can start the frontend and server.

```console
cd svelte-chat-app
npm install
npm install axios
```

Start the frontend by running:

```console
npm run dev
```

We also need to start the backend...

```console
uvicorn main:app --reload
```


# Add new requirements

Requirements are compile to fix all versions, including those of dependencies of dependencies. To add new requirements add the package to `requirements.in`. Then run:

```
pip-compile requirements.in
```

# Run tests locally

To run tests use either of the following commands

```console
pytest # this by default runs all tests in tests/
pytest -v # run all tests verbose (see which test is currently running)
pytest -s # run all tests and print whatever the code would print
pytest -x # run all tests and stop after first failure
pytest -vsx # all of the above
```
