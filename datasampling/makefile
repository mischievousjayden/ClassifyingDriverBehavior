DATA_PATH=./raw_data/exp-vs-inexp-filtered
SAMPLE=../data/sample
LAST=../data/last
MEAN=../data/mean
GAUSSIAN=../data/gaussian
PERIOD=10

mask: clean
	mkdir -p ${LAST}/expert ${LAST}/inexpert
	mkdir -p ${MEAN}/expert ${MEAN}/inexpert
	mkdir -p ${GAUSSIAN}/expert ${GAUSSIAN}/inexpert
	python mask_datasampling.py $(DATA_PATH)/expert ${LAST}/expert -p ${PERIOD} -m 1 -s 10
	python mask_datasampling.py $(DATA_PATH)/inexpert ${LAST}/inexpert -p ${PERIOD} -m 1 -s 10
	python mask_datasampling.py $(DATA_PATH)/expert ${MEAN}/expert -p ${PERIOD} -m 2 -s 10
	python mask_datasampling.py $(DATA_PATH)/inexpert ${MEAN}/inexpert -p ${PERIOD} -m 2 -s 10
	python mask_datasampling.py $(DATA_PATH)/expert ${GAUSSIAN}/expert -p ${PERIOD} -m 3 -s 11
	python mask_datasampling.py $(DATA_PATH)/inexpert ${GAUSSIAN}/inexpert -p ${PERIOD} -m 3 -s 11

sample: clean
	mkdir -p ${SAMPLE}/expert ${SAMPLE}/inexpert
	python datasampling.py ${PERIOD} $(DATA_PATH)/expert ${SAMPLE}/expert
	python datasampling.py ${PERIOD} $(DATA_PATH)/inexpert ${SAMPLE}/inexpert

clean:
	rm -rf __pycache__
	rm -rf ${SAMPLE}/expert ${SAMPLE}/inexpert
	rm -rf ${LAST}/expert ${LAST}/inexpert
	rm -rf ${MEAN}/expert ${MEAN}/inexpert
	rm -rf ${GAUSSIAN}/expert ${GAUSSIAN}/inexpert

