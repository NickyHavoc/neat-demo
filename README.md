# Configurable Chat Agent

Repository with quickly configurable chat agent on your own data.


## Install editable with https

```console
git clone https://github.com/NickyHavoc/neat-demo.git
cd neat_demo
pip install -e .
```

## Create your .env file

In order to set credentials to the api that should never be pushed to a repository, copy the file ".env_example", rename the copy to ".env" and paste your personal API token.

# Run tests locally

To run tests use either of the following commands

```console
pytest # this by default runs all tests in tests/
pytest -v # run all tests verbose (see which test is currently running)
pytest -s # run all tests and print whatever the code would print
pytest -x # run all tests and stop after first failure
pytest -vsx # all of the above
```
