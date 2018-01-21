
import matplotlib.pyplot as plt
from numpy import random as rdm
from pandas import DataFrame
from pandas.plotting import scatter_matrix
df = DataFrame(rdm.randn(1000, 4), columns=['a', 'b', 'c', 'd'])
scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')

plt.show()
