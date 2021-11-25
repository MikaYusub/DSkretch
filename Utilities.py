import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import glob
from typing import List


class DrawPlots:
    def __init__(self, json):

        self.json = json

    def draw_plots(self) -> List[str] :
        if not os.path.isdir('plots'):
            os.mkdir('plots')

        df = pd.read_json(self.json)

        plt.scatter(df['ceiling_max'], df['floor_max'])
        plt.title('Comparing ceiling_max and floor_max')
        plt.xlabel('Ceiling_max')
        plt.ylabel('Floor_max')
        plt.savefig(f'plots\\{plt.gca().get_title().replace(" ","_")}.jpg')
        plt.show()

        # Plotting confusion matrix for results of model prediction
        df_cm = pd.DataFrame(
            confusion_matrix(df['gt_corners'], df['rb_corners']),
            index=["4 corners", "6 corners", "8 corners", "10 corners"],
            columns=["4 corners", "6 corners", "8 corners", "10 corners"])

        ax = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g')

        plt.title('Confusion matrix', fontsize=20)
        plt.xlabel('Predicted by model', fontsize=15)
        plt.ylabel('Ground truth', fontsize=15)
        plt.savefig(f'plots\\{ax.get_title().replace(" ","_")}.jpg')
        plt.show()

        f, ax = plt.subplots(figsize=(13, 13))
        corr = df.corr()
        corr_fig = sns.heatmap(
            corr,
            mask=np.zeros_like(corr, dtype=np.bool),
            cmap='Blues',
            square=True,
            ax=ax,
            annot=True,
            fmt='.3f').set_title('Correlation between columns')
        fig = corr_fig.get_figure()
        fig.savefig(f'plots\\{ax.get_title().replace(" ","_")}.jpg')
        plt.show()
        
        f1_score(df['gt_corners'], df['rb_corners'], average='weighted')
        # F1 score is one of the best metrics to measure model performance in multiclass classification problem.
        # Score 1.0 show that our model predicted all values correct.

        return [path for path in glob.glob("plots\*")]