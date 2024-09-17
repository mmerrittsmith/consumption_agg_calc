# Makefile for running construct_consumption_agg.py

# Python interpreter
PYTHON = python

# The main script
SCRIPT = src/construct_consumption_agg.py

# Default target
all: run

# Run the script
run:
	$(PYTHON) $(SCRIPT)

# Clean target (optional, add any temporary files or outputs here)
clean:
	@echo "Cleaning up..."
	# Add commands to remove any generated files, if applicable
	# For example: rm -f output.csv
	rm -f outputs/food_level_df.csv
	rm -f outputs/hh_level_df.csv
	rm -f plots/food_consumption_vs_rexp_cat01.png

# Phony targets
.PHONY: all run clean