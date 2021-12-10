import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def save_map(list_2d):
  plt.figure()
  sns.heatmap(list_2d,xticklabels = range(2,8,1),yticklabels=range(7, 1, -1),vmin = 0)
  plt.savefig('seaborn_heatmap_individual.png')
  plt.close('all')
