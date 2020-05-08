SOURCE_DIRECTORY := src/
TEST_DIRECTORY := test/

all: lint test

install_dev_requirements:
	pip install --requirement config/requirements-dev.txt

lint:
	flake8 $(SOURCE_DIRECTORY) $(TEST_DIRECTORY)
	PYTHONPATH=$(SOURCE_DIRECTORY) pylint $(SOURCE_DIRECTORY) $(TEST_DIRECTORY)
	MYPYPATH=$(SOURCE_DIRECTORY) mypy $(SOURCE_DIRECTORY) $(TEST_DIRECTORY)

test:
	PYTHONPATH=$(SOURCE_DIRECTORY) python -m unittest

.PHONY: all install_dev_requirements lint test
