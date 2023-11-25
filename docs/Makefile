SHELL := /bin/bash

bin/activate: requirements.txt
	python3 -m venv . && source bin/activate && pip install -r requirements.txt

docs: bin/activate
	source bin/activate && sphinx-build . html