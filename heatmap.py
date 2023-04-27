import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def save_map(list_2d,BN,depth):
  plt.figure()
  d = "all" if depth == "all" else "individual"+depth
  sns.heatmap(list_2d,xticklabels = range(2,8,1),yticklabels=range(7, 1, -1),vmin = 0)
  plt.savefig('{}-2-heatmap-dropout-{}.png'.format(BN,d))
  plt.close('all')

def save_map_synapse(list_2d,BN,depth):
  plt.figure()
  d = "all" if depth == "all" else "individual"+depth
  sns.heatmap(list_2d,xticklabels = range(2,8,1),yticklabels=range(7, 1, -1),vmin = 0)
  plt.savefig('Synapse_{}-heatmap-{}.png'.format(BN,d))
  plt.close('all')

