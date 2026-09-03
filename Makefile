PYTHON ?= .venv/bin/python
ARGS ?=

.DEFAULT_GOAL := all
.PHONY: all check report

all:
	bash scripts/run_all.sh $(ARGS)

check:
	bash scripts/run_all.sh --dry-run $(ARGS)

report:
	$(PYTHON) -m tfgnn.report
