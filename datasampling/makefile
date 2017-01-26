DATA_PATH="./raw_data/exp-vs-inexp-filtered"

sample: clean
	mkdir ../data/expert
	mkdir ../data/inexpert
	python datasampling.py 10 $(DATA_PATH)/expert ../data/expert
	python datasampling.py 10 $(DATA_PATH)/inexpert ../data/inexpert

clean:
	rm -rf ../data/expert
	rm -rf ../data/inexpert

