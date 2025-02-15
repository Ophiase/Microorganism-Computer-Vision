.PHONY: pip extract transform detection render analysis report test

VERBOSE=$(if $(V),--verbose,)

#########################################

pip:
	pip install -r requirements.txt

extract:
	python3 -m script.main --task extract $(VERBOSE)

transform:
	python3 -m script.main --task transform $(VERBOSE)
	
detection:
	python3 -m script.main --task detection $(VERBOSE)

synthetic:
	python3 -m script.main --task synthetic $(VERBOSE)

render:
	python3 -m script.main --task render $(VERBOSE)

analysis:
	python3 -m script.main --task analysis $(VERBOSE)

report:
	echo "TODO..."

test:
	echo "TODO...."

#########################################

detection_and_analysis: detection analysis

all: 
	@echo "Specify a target. Default behavior is disabled."
