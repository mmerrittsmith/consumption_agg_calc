# Makefile for running all consumption aggregate calculations

PYTHON = python

X01_DIR = x01_food_sub_agg
X02_DIR = x02_non_food_sub_agg  
X03_DIR = x03_durable_goods_sub_agg
X04_DIR = x04_housing_sub_agg
X05_DIR = x05_agg_sub_aggs

all: x01 x02 x03 x04 x05

x01:
	@echo "Running food sub-aggregate calculations..."
	$(MAKE) -C $(X01_DIR) run

x02:
	@echo "Running non-food sub-aggregate calculations..."
	$(MAKE) -C $(X02_DIR) run

x03:
	@echo "Running durable goods sub-aggregate calculations..."
	$(MAKE) -C $(X03_DIR) run

x04:
	@echo "Running housing sub-aggregate calculations..."
	$(MAKE) -C $(X04_DIR) run

x05: x01 x02 x03 x04
	@echo "Aggregating all components..."
	$(MAKE) -C $(X05_DIR) run

clean:
	@echo "Cleaning all directories..."
	$(MAKE) -C $(X01_DIR) clean
	$(MAKE) -C $(X02_DIR) clean
	$(MAKE) -C $(X03_DIR) clean
	$(MAKE) -C $(X04_DIR) clean
	$(MAKE) -C $(X05_DIR) clean

tests:
	@echo "Running all tests..."
	$(MAKE) -C $(X01_DIR) tests
	$(MAKE) -C $(X02_DIR) tests
	$(MAKE) -C $(X03_DIR) tests
	$(MAKE) -C $(X04_DIR) tests
	$(MAKE) -C $(X05_DIR) tests

.PHONY: all x01 x02 x03 x04 x05 clean tests
