import numpy as np
import matplotlib.pyplot as plt

num_neurons = ["16", "32", "64", "128", "256"]
raw_accuracy = [0.5, 0.5, 0.6875, 0.59375, 0.5]
avg_accuracy = [0.75, 0.5, 0.25, 1.0, 0.5]

ind = np.arange(len(raw_accuracy));
width = 0.3
color1 = "blue"
plt.bar(ind-width/2, raw_accuracy, width=width, color=color1)
color2 = "red"
plt.bar(ind+width/2, avg_accuracy, width=width, color=color2)
plt.xticks(range(len(raw_accuracy)), num_neurons)
plt.title("Accuracy vs. The number of Neurons")
plt.xlabel("Accuracy")
plt.ylabel("The Number of Neurons")
plt.show()

