import numpy as np
import matplotlib.pyplot as plt

num_neurons = ["16", "32", "64", "128", "256"]
raw_accuracy = [0.5, 0.5, 0.6875, 0.59375, 0.5]
avg_accuracy = [0.75, 0.5, 0.25, 1.0, 0.5]

width = 0.6
color = "blue"
plt.bar(range(len(raw_accuracy)), raw_accuracy, width=width, color=color)
plt.xticks(range(len(raw_accuracy)), num_neurons)
plt.title("Accuracy vs. The number of Neurons")
plt.xlabel("Accuracy")
plt.ylabel("The Number of Neurons")
plt.show()

