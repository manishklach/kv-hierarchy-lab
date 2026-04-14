PYTHON ?= python
ARTIFACT_DIR ?= artifacts/default_sweep

.PHONY: test bench plots smoke

test:
	$(PYTHON) -m pytest

bench:
	$(PYTHON) scripts/run_benchmarks.py --output-dir $(ARTIFACT_DIR)

plots:
	$(PYTHON) scripts/plot_results.py --results $(ARTIFACT_DIR)/results.json --out-dir $(ARTIFACT_DIR)/plots

smoke:
	$(PYTHON) examples/basic_simulation.py
