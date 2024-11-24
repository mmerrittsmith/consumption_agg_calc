# Makefile for running construct_housing_sub_agg.py

PYTHON = python

SCRIPT = src/construct_housing_sub_agg.py
TEST_SCRIPT = tests/test_housing_sub_agg.py


all: run

run:
	$(PYTHON) $(SCRIPT)

clean:
	@echo "Cleaning up..."
	# Add commands to remove any generated files, if applicable
	# For example: rm -f output.csv
	rm -f outputs/housing_level_df.csv
	rm -f outputs/housing_hh_level_df.csv
	rm -f plots/housing_consumption_vs_rexp_cat01.png

tests:
	PYTHONPATH=$(shell pwd) $(PYTHON) -m pytest $(TEST_SCRIPT)

.PHONY: all run clean tests