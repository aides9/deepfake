"""
Created on Thu Dec 31 04:25:24 2020

@author: Jen
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, auc, roc_curve, log_loss, precision_recall_curve, average_precision_score

class Eval():
    def __init__(self, model, dataset, device):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.testlabels = []
        self.predictions = []

    def run(self, batch_size=16):
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,shuffle=True)
        # Test the model with unseen data

        for _, (inputs, labels) in tqdm(enumerate(loader),total=len(loader)):
          inputs = inputs.to(self.device)
          if hasattr(self.model, 'CapsuleNet'):
            _, preds = self.model(inputs)
          else:
            preds = self.model(inputs)
            preds = torch.sigmoid(preds)
          self.predictions.extend(preds.cpu().detach().numpy())
          self.testlabels.extend(labels.cpu().detach().numpy())
          
        print('F1 Score:', f1_score(self.testlabels, np.round(self.predictions), average="macro"))
        print('Log Loss:', log_loss(self.testlabels, self.predictions))
        print("Confusion Matrix:")
        print(confusion_matrix(self.testlabels, np.round(self.predictions)))
        pass
    
    def plot_precisionrecall(self):
        precision, recall, _ = precision_recall_curve([1- x for x in self.testlabels],[1-y for y in self.predictions])
        plt.plot(recall, precision)
            
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.title("precision vs. recall curve")
        plt.show()
        pass
        
    def plot_roc(self):
        fpr, tpr, _ = roc_curve(self.testlabels,self.predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        pass
        
        


