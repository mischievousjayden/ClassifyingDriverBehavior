CLASS=inexpert
FILENAME=sample_filtered-SCH_Sub_Inexp_Drive_1.dat

RESULT1=filtered_1_10_result
RESULT2=filtered_1_20_result
RESULT3=filtered_1_50_result
RESULT4=compare_result_last_1_10

result:
	python3 result_graph.py ../result/$(RESULT1).csv ../pictures/result_pictures/$(RESULT1).png "1/10 Average Accuracy vs. The Number of Neurons"
	python3 result_graph.py ../result/$(RESULT2).csv ../pictures/result_pictures/$(RESULT2).png "1/20 Average Accuracy vs. The Number of Neurons"
	python3 result_graph.py ../result/$(RESULT3).csv ../pictures/result_pictures/$(RESULT3).png "1/50 Average Accuracy vs. The Number of Neurons"
	python3 result_graph.py ../result/$(RESULT4).csv ../pictures/result_pictures/$(RESULT4).png "Compare Human vs. MAE vs. SAE"

data:
	python3 data_graph.py ${CLASS}/${FILENAME} 2

clean:
	rm -rf __pycache__

