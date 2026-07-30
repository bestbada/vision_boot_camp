import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# utils/plot_metrics.py

def plot_training_curve(results_csv_path, save_path=None):
    df = pd.read_csv(results_csv_path)
    df.columns = df.columns.str.strip()

    # 의미: pandas Series를 numpy 배열로 명시적으로 변환
    # 사용 이유: 오래된 matplotlib이 Series를 numpy처럼 다루려다 실패하는
    #           버전 궁합 문제를 우회하기 위함
    epochs = df['epoch'].to_numpy()
    precision = df['metrics/precision(B)'].to_numpy()
    recall = df['metrics/recall(B)'].to_numpy()

    plt.figure()
    plt.plot(epochs, precision, label="Precision")
    plt.plot(epochs, recall, label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Training Precision/Recall over Epochs")

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_class_performance(metrics, class_names, save_path=None):
    class_list = list(class_names.values())
    precision_per_class = np.asarray(metrics.box.p)   # numpy 배열로 명시 변환
    recall_per_class = np.asarray(metrics.box.r)

    x = range(len(class_list))
    plt.figure()
    plt.bar([i - 0.2 for i in x], precision_per_class, width=0.4, label="Precision")
    plt.bar([i + 0.2 for i in x], recall_per_class, width=0.4, label="Recall")
    plt.xticks(x, class_list)
    plt.ylabel("Score")
    plt.legend()
    plt.title("Class-wise Precision/Recall")

    if save_path:
        plt.savefig(save_path)
    plt.show()
