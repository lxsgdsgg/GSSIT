import pandas as pd
import numpy as np
from sklearn import metrics

def load_data(path):
    with open(path, 'r') as f:
        lines = f.read().split('\n')[0:-1]
    lines = [l.strip() for l in lines]
    return lines

def cal_accuracy(rst_file_dir, y_test_dir):
    rst_contents = pd.read_csv(rst_file_dir, sep='\t', header=None)
    value_list = rst_contents.values
    pred = value_list.argmax(axis=1)
    labels = []

    y2id, id2y = get_y_to_id()
    with open(y_test_dir, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            label = line.strip()
            labels.append(y2id[label])
            line = f.readline()
    labels = np.asarray(labels)
    error_file=open('wrong_pred.txt', 'w')
    test_x = load_data('../data/test_x.txt')
    # for i in range(len(labels)):
    #     if labels[i] != pred[i]:
    #         error_file.write(str(id2y[labels[i]])+' '+str(id2y[pred[i]])+' '+test_x[i]+'\n')


    accuracy = metrics.accuracy_score(y_true=labels, y_pred=pred)
    print(accuracy)
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(labels, pred, digits=4))


def get_y_to_id():
    # y_vocab = ['description', 'expected', 'others', 'reproduce']
    y_vocab = ['aspect evaluation', 'feature request', 'information giving', 'information seeking', 'problem discovery', 'solution proposal', 'others']
    y2idx = {token: idx for idx, token in enumerate(y_vocab)}
    idx2y = {idx: token for idx, token in enumerate(y_vocab)}
    return y2idx, idx2y


if __name__ == '__main__':
    # count = len(open(r"../data/statutes_small/test_x_no_seg.txt", 'r', encoding='utf-8').readlines())
    # print(count)
    cal_accuracy(rst_file_dir='../model/tensorflow/test_results.tsv', y_test_dir='../data/tensorflow/test_y.txt')
