PROG_FILTERED=classify_filtered_driving_data.py
PROG_RAW=classify_raw_driving_data.py
FILTER=last
DATADIR=../data/${FILTER}
LOGDIR=./logs/test/${FILTER}


run_backup:
	python3 ${PROG_RAW} ${DATADIR} -n 16 -l ${LOGDIR}/16_neurons
	cd ${LOGDIR} && zip -r 16_neurons.zip 16_neurons && rm -rf 16_neurons

run_filtered:
	python3 ${PROG_FILTERED} ${DATADIR} -n 16 -l ${LOGDIR}/16_neurons

run_raw:
	python3 ${PROG_RAW} ${DATADIR} -n 16 -l ${LOGDIR}/16_neurons

run_tensorboard:
	tensorboard --logdir=${LOGDIR}/16_neurons/groupid0

backup:
	cd ${LOGDIR} && zip -r 16_neurons.zip 16_neurons

clean:
	rm -rf logs __pycache__ util/__pycache__

