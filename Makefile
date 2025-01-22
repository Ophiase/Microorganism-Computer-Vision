.PHONY: pip extract transform detection coherence analysis report

#########################################

pip:
	pip install -r requirements.txt

extract:
	python3 extract.py

transform:
	python3 transform.py

detection:
	python3 object_detection.py

render:
	python3 render.py

coherence:
	echo "TODO"

analysis:
	echo "TODO"

report:
	echo "TODO"

#########################################

all: 
	@echo "Specify a target. Default behavior is disabled."
