SOURCE_DIRECTORY := src/
SOURCES := $(shell find $(SOURCE_DIRECTORY) -name "*.m")
TEST_FILENAME := tests/neumann_control_test.m


check: lint

lint:
	./lint $(SOURCES) $(TEST_FILENAME)

.PHONY: check lint
