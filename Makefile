.PHONY: pip extract transform detection render analysis report

#########################################

pip:
	pip install -r requirements.txt

extract:
	python3 -m script.main --task extract

transform:
	python3 -m script.main --task transform 
	
detection:
	python3 -m script.main --task detection

synthetic:
	python3 -m script.main --task synthetic

render:
	python3 -m script.main --task render

analysis:
	python3 -m script.main --task analysis

report:
	echo "TODO..."

test:
	echo "TODO..."

#########################################

detection_and_analysis: detection analysis

all: 
	@echo "Specify a target. Default behavior is disabled."
