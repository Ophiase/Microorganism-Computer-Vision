.PHONY: pip extract transform detection render analysis report

#########################################

pip:
	pip install -r requirements.txt

extract:
	python3 -m script.extract

transform:
	python3 -m script.transform

detection:
	python3 -m script.object_detection

render:
	python3 -m script.render

analysis:
	python3 -m script.statistical_tests

report:
	echo "TODO"

#########################################

detection_and_analysis: detection analysis

all: 
	@echo "Specify a target. Default behavior is disabled."
