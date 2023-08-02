import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def save_map(list_2d,BN,depth):
  plt.figure()
  d = "all" if depth == "all" else "individual"+depth
  sns.heatmap(list_2d,xticklabels = range(2,8,1),yticklabels=range(7, 1, -1),vmin = 0)
  plt.savefig('cvc_{}-heatmap-{}.png'.format(BN,d))
  plt.close('all')


def save_map_dropout(list_2d,BN,depth):
  plt.figure()
  d = "all" if depth == "all" else "individual"+depth
  sns.heatmap(list_2d,xticklabels = range(2,8,1),yticklabels=range(7, 1, -1),vmin = 0)
  plt.savefig('cvc_{}-heatmap-{}_dropout.png'.format(BN,d))
  plt.close('all')

def save_map_synapse(list_2d,BN,depth):
  plt.figure()
  d = "all" if depth == "all" else "individual"+depth
  sns.heatmap(list_2d,xticklabels = range(2,8,1),yticklabels=range(7, 1, -1),vmin = 0)
  plt.savefig('Synapse_{}-heatmap-{}.png'.format(BN,d))
  plt.close('all')

def save_map_synapse_dropout(list_2d,BN,depth):
  plt.figure()
  d = "all" if depth == "all" else "individual"+depth
  sns.heatmap(list_2d,xticklabels = range(2,8,1),yticklabels=range(7, 1, -1),vmin = 0)
  plt.savefig('Synapse_{}-heatmap-{}_dropout.png'.format(BN,d))
  plt.close('all')

def save_map_dsb(list_2d,BN,depth):
  plt.figure()
  d = "all" if depth == "all" else "individual"+depth
  sns.heatmap(list_2d,xticklabels = range(2,8,1),yticklabels=range(7, 1, -1),vmin = 0)
  plt.savefig('dsb_{}-heatmap-{}.png'.format(BN,d))
  plt.close('all')

def save_map_dsb_dropout(list_2d,BN,depth):
  plt.figure()
  d = "all" if depth == "all" else "individual"+depth
  sns.heatmap(list_2d,xticklabels = range(2,8,1),yticklabels=range(7, 1, -1),vmin = 0)
  plt.savefig('dsb_{}-heatmap-{}_dropout.png'.format(BN,d))
  plt.close('all')

def save_map_cbis(list_2d,BN,depth):
  plt.figure()
  d = "all" if depth == "all" else "individual"+depth
  sns.heatmap(list_2d,xticklabels = range(2,8,1),yticklabels=range(7, 1, -1),vmin = 0)
  plt.savefig('cbis_{}-heatmap-{}.png'.format(BN,d))
  plt.close('all')

def save_map_cbis_dropout(list_2d,BN,depth):
  plt.figure()
  d = "all" if depth == "all" else "individual"+depth
  sns.heatmap(list_2d,xticklabels = range(2,8,1),yticklabels=range(7, 1, -1),vmin = 0)
  plt.savefig('cbis_{}-heatmap-{}_dropout.png'.format(BN,d))
  plt.close('all')


def save_map_busi(list_2d,BN,depth):
  plt.figure()
  d = "all" if depth == "all" else "individual"+depth
  sns.heatmap(list_2d,xticklabels = range(2,8,1),yticklabels=range(7, 1, -1),vmin = 0)
  plt.savefig('busi_{}-heatmap-{}.png'.format(BN,d))
  plt.close('all')

def save_map_busi_dropout(list_2d,BN,depth):
  plt.figure()
  d = "all" if depth == "all" else "individual"+depth
  sns.heatmap(list_2d,xticklabels = range(2,8,1),yticklabels=range(7, 1, -1),vmin = 0)
  plt.savefig('busi_{}-heatmap-{}_dropout.png'.format(BN,d))
  plt.close('all')




