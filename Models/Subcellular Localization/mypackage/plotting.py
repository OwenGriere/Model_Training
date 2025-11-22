import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
##################################################### PLotting #####################################################

def plot_training_curves(train_losses, val_losses, train_accs=None, val_accs=None, model_name="Model", verbose=False, test=False):
    plt.ioff()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train loss')
    if val_losses:
        plt.plot(val_losses, label='Test loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Courbes de pertes - {model_name}")
    plt.grid(True)
    plt.legend()

    if train_accs is not None and val_accs is not None:
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train acc')
        plt.plot(val_accs, label='Test acc')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy - {model_name}")
        plt.grid(True)
        plt.legend()
    
    txt=''
    if test:
        txt='test_'
    plt.tight_layout()
    os.makedirs(f"Figures/{model_name}", exist_ok=True)
    loss_path = f"Figures/{model_name}/{txt}loss_and_accuracy.png"
    plt.savefig(loss_path)
    plt.close()

def plot_confusion_matrix(cf_matrix, classes=None, title="Matrice de confusion", model_name="Model", Norm=True, verbose=False, test=False):
    plt.ioff()
    n_class = cf_matrix.shape[0]
    counts_per_class = cf_matrix.sum(axis=1)

    if classes is None:
        classes = [str(i) for i in range(n_class)]
    if Norm:
        cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(cf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(n_class)
    ytick_labels = [f"({int(counts_per_class[i])} elements) {classes[i]}" for i in range(n_class)]
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, ytick_labels)

    thresh = cf_matrix.max() / 2

    for i in range(n_class):
        for j in range(n_class):
            value = f"{cf_matrix[i, j]:.2f}"
            plt.text(j, i, value,
                     horizontalalignment="center",
                     color="white" if cf_matrix[i, j] > thresh else "black")
            
    txt=""
    if test:
        txt='test_'
    
    os.makedirs(f"Figures/{model_name}", exist_ok=True)
    plt.ylabel("Label réel")
    plt.xlabel("Label prédit")
    plt.tight_layout()
    cf_path = f"Figures/{model_name}/{txt}confusion.png"
    plt.savefig(cf_path)
    plt.close()
