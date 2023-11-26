# neat

neat is a ReAct-like agent accessing all sorts of data & APIs

![In case you were wondering how the agent does this.](https://github.com/NickyHavoc/neat-demo/blob/main/neat_ai_assistant.png)

## Getting started

This project uses [Poetry](https://python-poetry.org/) for dependency management. If you haven't installed Poetry yet, you can do so by following the instructions on the official website.

Sidenote: poetry >>> conda & pip.

Once you have Poetry installed, you have 2 options.

### Include it in your dependencies

``` toml
[tool.poetry.dependencies]
neat-ai-assistant = {git = "https://github.com/NickyHavoc/neat-demo.git", branch="main"}
```

Needless to say, you need poetry set up in your environment to do this.

Optionally, you can include some extra dependencies:

``` toml
neat-ai-assistant = {git = "https://github.com/NickyHavoc/neat-demo.git", branch="main", extras = ["tool-extension"]}
```

### Clone the repo

```console
git clone https://github.com/NickyHavoc/neat-demo.git
cd neat-demo
poetry install
```

If you want to use some extra tools that require special dependencies you can install them like so:

``` console
poetry install -E tool-extension
```

## Create your `.env` file

In order to set credentials to the api that should never be pushed to a repository, copy the file ".env_example", rename the copy to ".env" and paste your personal API tokens.
At least, you must include your "OPENAI_API_KEY" here.

## Launching the server

### Backend

This guide assumes that you have already installed FastAPI and an ASGI server like Uvicorn, which is required to run a FastAPI application.
If you haven't installed these yet, you can do so using poetry:

``` console
poetry add fastapi uvicorn
```

Navigate to Your Project Directory:
Open a terminal or command prompt and navigate to the directory where your main.py file is located.

``` console
uvicorn main:app --reload
```

Once the server is running, you should see output in the terminal indicating that it's live.
By default, the server will be available at `http://127.0.0.1:8000`.
You can use a tool like curl to interact with the API.

API Documentation:
FastAPI automatically generates interactive API documentation for your server.
You can visit it at `http://127.0.0.1:8000/docs`.
This page will allow you to see all your endpoints and test them directly from the browser.

### Frontend

Simply navigate to the `react-ui` directory and start npm, like so:

``` console
cd react-ui
npm start
```

To learn more, check out the [frontend-readme](https://github.com/NickyHavoc/neat-demo/blob/main/react-ui/README.md).

## Run tests locally

To run tests use either of the following commands

```console
pytest # this by default runs all tests in tests/
pytest -v # run all tests verbose (see which test is currently running)
pytest -s # run all tests and print whatever the code would print
pytest -x # run all tests and stop after first failure
pytest -vsx # all of the above
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
