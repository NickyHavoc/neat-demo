# neat

neat is a ReAct-like agent accessing all sorts of data & APIs

## Install editable with https

This project uses [Poetry](https://python-poetry.org/) for dependency management. If you haven't installed Poetry yet, you can do so by following the instructions on the official website.

Once you have Poetry installed, clone the repository and install the dependencies as follows:

```console
git clone https://github.com/NickyHavoc/neat-demo.git
cd neat-demo
poetry install
```

If you want to create a new virtual environment for the project using Poetry, you can do so:

``` console
poetry shell
poetry install
```

## Create your .env file

In order to set credentials to the api that should never be pushed to a repository, copy the file ".env_example", rename the copy to ".env" and paste your personal API tokens.

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
