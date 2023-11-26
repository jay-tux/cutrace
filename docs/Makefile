SHELL := /bin/bash

docs:
	if [ ! -f bin/activate ]; then python3 -m venv . && source bin/activate && pip install -r requirements.txt; fi
	source bin/activate && sphinx-build . html

.PHONY: docs