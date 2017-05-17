PROG_FILTERED=classify_filtered_driving_data.py
PROG_RAW=classify_raw_driving_data.py
PROG_AE=multi_autoencoder.py
FILTER=last
DATADIR=../data/${FILTER}
LOGROOT=./logs
LOGDIR=${LOGROOT}/test/${FILTER}


exp:
	python3 ${PROG_RAW} ${DATADIR} -n 16 -l ${LOGDIR}/16_neurons
	cd ${LOGDIR} && zip -r 16_neurons.zip 16_neurons && rm -rf 16_neurons

run_filtered:
	python3 ${PROG_FILTERED} ${DATADIR} -n 16 -l ${LOGROOT}/filtered/${FILTER}/16_neurons

run_raw:
	python3 ${PROG_RAW} ${DATADIR} -n 16 -l ${LOGROOT}/raw/${FILTER}/16_neurons

run_ae:
	python3 ${PROG_AE} ${DATADIR} -l ${LOGROOT}/autoencoder/${FILTER}

run_tensorboard:
	tensorboard --logdir=${LOGDIR}/16_neurons/groupid0

backup:
	cd ${LOGDIR} && zip -r 16_neurons.zip 16_neurons

clean:
	rm -rf logs __pycache__ util/__pycache__

