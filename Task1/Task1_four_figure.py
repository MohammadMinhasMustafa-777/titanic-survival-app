import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv(r'C:\Python 2k25\Task1\iris.csv')  # Assumes iris.csv in week1/
print("Stats:\n", df.describe())

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(df['sepal_length'], df['sepal_width'])
plt.title('Sepal Length vs Width')

plt.subplot(2, 2, 2)
df['petal_length'].hist(bins=20)
plt.title('Petal Length Distribution')

plt.subplot(2, 2, 3)
df.boxplot(column='sepal_length', by='species', ax=plt.gca())
plt.title('')

plt.tight_layout()
plt.savefig(r'C:\Python 2k25\Task1\iris_analysis.png')
plt.show()  # Optional: View in popup
print("Plots saved to plots/!")