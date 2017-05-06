DATA_PATH=./driving_data/exp-vs-inexp-extended
SAMPLE=../data/sample
LAST=../data/last
MEAN=../data/mean
GAUSSIAN=../data/gaussian
PERIOD=10
MASK_SIZE=10


mask: clean_all
	mkdir -p ${LAST}/expert ${LAST}/inexpert
	mkdir -p ${MEAN}/expert ${MEAN}/inexpert
	mkdir -p ${GAUSSIAN}/expert ${GAUSSIAN}/inexpert
	python mask_datasampling.py $(DATA_PATH)/expert ${LAST}/expert -p ${PERIOD} -m 1 -s ${MASK_SIZE}
	python mask_datasampling.py $(DATA_PATH)/inexpert ${LAST}/inexpert -p ${PERIOD} -m 1 -s ${MASK_SIZE}
	python mask_datasampling.py $(DATA_PATH)/expert ${MEAN}/expert -p ${PERIOD} -m 2 -s ${MASK_SIZE}
	python mask_datasampling.py $(DATA_PATH)/inexpert ${MEAN}/inexpert -p ${PERIOD} -m 2 -s ${MASK_SIZE}
	python mask_datasampling.py $(DATA_PATH)/expert ${GAUSSIAN}/expert -p ${PERIOD} -m 3 -s $(shell echo $(MASK_SIZE)\+1 | bc)
	python mask_datasampling.py $(DATA_PATH)/inexpert ${GAUSSIAN}/inexpert -p ${PERIOD} -m 3 -s $(shell echo $(MASK_SIZE)\+1 | bc)

sample: clean_all
	mkdir -p ${SAMPLE}/expert ${SAMPLE}/inexpert
	python datasampling.py ${PERIOD} $(DATA_PATH)/expert ${SAMPLE}/expert
	python datasampling.py ${PERIOD} $(DATA_PATH)/inexpert ${SAMPLE}/inexpert

clean:
	rm -rf __pycache__ *.pyc

clean_target:
	rm -rf ${SAMPLE}/expert ${SAMPLE}/inexpert
	rm -rf ${LAST}/expert ${LAST}/inexpert
	rm -rf ${MEAN}/expert ${MEAN}/inexpert
	rm -rf ${GAUSSIAN}/expert ${GAUSSIAN}/inexpert

clean_all: clean clean_target

