import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# plot confusion matrix

# labels = ['contradiction', 'entailment', 'neutral']
def plot_confusion_matrix():
  labels = ['cont.', 'entail.', 'neut.']
  cm = np.array([[8, 1, 1], [3, 5, 2 ], [2, 2, 6]])
  df_cm = pd.DataFrame(cm, labels, labels)

  fig = plt.figure(figsize=(20,6))
  ax1 = fig.add_subplot(131)
  ax2 = fig.add_subplot(132)
  ax3 = fig.add_subplot(133)

  ax1.set(title="Fine-tune XLM-R")
  ax2.set(title="Meta-learning En")
  ax3.set(title="Meta-learning En,Fr")



  sn.set(font_scale=0.8) # for label size
  sn.heatmap(df_cm, ax=ax1, cmap="Blues", vmin=0, vmax=10, annot=True, annot_kws={"size": 10}) # font size
  sn.heatmap(df_cm, ax=ax2, cmap="Blues", vmin=0, vmax=10, annot=True, annot_kws={"size": 10}) # font size
  sn.heatmap(df_cm, ax=ax3, cmap="Blues", vmin=0, vmax=10, annot=True, annot_kws={"size": 10}) # font size

  ax1.set_xlabel('Predicted label')
  ax1.set_ylabel('True label')
  ax2.set_xlabel('Predicted label')
  ax2.set_ylabel('True label')
  ax3.set_xlabel('Predicted label')
  ax3.set_ylabel('True label')

  fig.savefig('confusion_matrix.png', format='png', dpi=800)
  plt.show()


def plot_cca():
  x = range(12)
  y_1 = [
    0.9535018842326805,
    0.9336592956526056,
    0.8974597892120914,
    0.8772908926673537,
    0.8245836852939218,
    0.7766242644800921,
    0.7644845555573542,
    0.746254565470776,
    0.7259084485389197,
    0.7078838277221133,
    0.6950564479433542,
    0.6737965381909125
  ]
  y_2 = [
    0.9968445518521888,
    0.9827014147553522,
    0.9614060765438898,
    0.9387345081820319,
    0.9205085691941529,
    0.9006617367880692,
    0.8818581914572393,
    0.8637575997704574,
    0.7962959336439903,
    0.7494117948594855,
    0.7058843318498922,
    0.675215592789189
  ]
  y_3 = [
    0.9960603187335559,
    0.9801339315207048,
    0.9569743527638649,
    0.9297731281640361,
    0.9058912969910408,
    0.8924045386082659,
    0.8708411667158665,
    0.8580053862413699,
    0.7955811240022209,
    0.740074158712195,
    0.6969297142284184,
    0.6810987694452476
  ]
  y_4 = [

  ]

  fig = plt.figure(figsize=(8,6))
  plt.plot(x, y_1, '-o', label='mBERT nli-(en,fr,es,de)')
  plt.plot(x, y_2, '-s', label='XLM-R nli-(en)')
  plt.plot(x, y_3, '-*', label='XLM-R nli-(en,fr,es,de)')
  # plt.plot(x, y_4, '-^', label='MetaLearn nli-(en)')
  plt.legend(loc='best')
  plt.xlabel('layer')
  plt.ylabel('CCA similarity')
  plt.xticks(np.arange(0, 12, step=1.))
  plt.yticks(np.arange(0.5, 1.01, step=0.1))
  # plt.grid()
  fig.savefig('cca.png', format='png', dpi=800)
  plt.show()


if __name__ == '__main__':
  plot_confusion_matrix()
  # plot_cca()
