import pandas as pd
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 7]

from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score, classification_report, precision_recall_fscore_support


def get_scores(model, iterator, batch_size, device, n_classes=6):
    n_samples = len(iterator.dataset)

    true_list = []
    prob_list = []
    pred_list = []

    y_true = torch.Tensor(n_samples).to(device)
    y_prob = torch.Tensor(n_samples, n_classes).to(device)
    y_pred = torch.Tensor(n_samples).to(device)

    #deactivating dropout layers
    model.eval()

    #deactivates autograd
    with torch.no_grad():
        for batch in iterator:

            img, label, metadata = batch
            img      = img.to(device)
            label    = label.to(device)
            metadata = metadata.to(device)

            #convert to 1d tensor
            scores = model(img, metadata)
            probs  = torch.softmax(scores, dim=1)
            preds  = probs.argmax(dim=1)

            true_list.append(label)
            prob_list.append(probs)
            pred_list.append(preds)

    torch.cat(true_list, out=y_true)
    torch.cat(prob_list, out=y_prob)
    torch.cat(pred_list, out=y_pred)

    return y_true.cpu().numpy(), y_prob.cpu().numpy(), y_pred.cpu().numpy()


def get_metrics(y_true, y_prob, y_pred):
    report_dict  = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    metrics_dict                      = report_dict['weighted avg']
    metrics_dict['accuracy']          = metrics_dict['recall']             # acc = microprec= microrecall = microf1 = weightrecall
    metrics_dict['balanced_accuracy'] = report_dict['macro avg']['recall'] # bacc = macrorecall
    metrics_dict['auc']               = roc_auc_score(y_true, y_prob, average='weighted', multi_class='ovr')

    return metrics_dict


def display_metrics(metrics_dict, model_name=0):
    display(pd.DataFrame(metrics_dict, index=[model_name]))
    

def df_to_latex(df):
    header       = f'\\begin{{tabular}}{{{"c"*len(df.columns)}}}\n\\toprule\n'
    column_names = f'{" & ".join(df.columns)}\\\\\n\\midrule\n'
    data         = '\\\\\n'.join([f'{" & ".join([str(x) for x in df.iloc[i]])}' for i in range(len(df))])
    table        = f'{header}{column_names}{data}\\\\\n\\bottomrule\n\\end{{tabular}}'
    return table


def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
