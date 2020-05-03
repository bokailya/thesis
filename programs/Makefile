SOURCE_DIRECTORY := src/

SOURCES := $(shell find $(SOURCE_DIRECTORY) -name '*.py')

all: lint

install_dev_requirements:
	pip install --requirement config/requirements-dev.txt

lint: $(SOURCES)
	flake8 $(SOURCE_DIRECTORY)
	pylint $(SOURCE_DIRECTORY)
	MYPYPATH=stubs/ mypy $(SOURCE_DIRECTORY)

