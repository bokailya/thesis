SOURCE_DIRECTORY := src/
STUB_DIRECTORY := stubs/

SOURCES := $(shell find $(SOURCE_DIRECTORY) -name '*.py')
STUBS := $(shell find $(STUB_DIRECTORY) -name '*.pyi')

all: lint

install_dev_requirements:
	pip install --requirement config/requirements-dev.txt

lint: $(SOURCES) $(STUBS)
	flake8 $(SOURCE_DIRECTORY) $(STUB_DIRECTORY)
	PYTHONPATH=src/ pylint $(SOURCE_DIRECTORY) $(STUB_DIRECTORY)
	MYPYPATH=$(STUB_DIRECTORY) mypy $(SOURCE_DIRECTORY) $(STUB_DIRECTORY)

