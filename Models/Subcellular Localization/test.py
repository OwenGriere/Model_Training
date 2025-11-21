import os
import sys
import yaml
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,optimizer=None,device=cpu,floatX=float32"
sys.path = [p for p in sys.path if ".local/lib/python3.8/site-packages" not in p]
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import itertools
import theano
import theano.tensor as T
import lasagne

from mypackage.building_models import *
from mypackage.confusionmatrix import ConfusionMatrix
from mypackage.utils import iterate_minibatches, LSTMAttentionDecodeFeedbackLayer, import_config
from mypackage.metrics_mc import gorodkin, IC
from mypackage.plotting import *

def load_npz_weights(npz_path):
    data = np.load(npz_path)
    weights = [data[f"arr_{i}"] for i in range(len(data.files))]
    return weights

def evaluate_model(val_fn, l_out, test_data, batch_size, uses_mask, verbose=False):
    X_test, y_test, mask_test = test_data
    n_class = int(np.max(y_test) + 1)

    test_err = 0.0
    test_batches = 0
    confusion_test = ConfusionMatrix(n_class)

    for inputs, targets, in_masks in iterate_minibatches(
        X_test, y_test, mask_test,
        batchsize=batch_size,
        shuffle=False,
        sort_len=False):

        inputs = inputs.astype("float32")
        targets = targets.astype("int32")

        if uses_mask and in_masks is not None:
            in_masks = in_masks.astype("float32")
            loss_batch, preds = val_fn(inputs, targets, in_masks)
        else:
            loss_batch, preds = val_fn(inputs, targets)

        test_err += float(loss_batch)
        test_batches += 1
        pred_labels = np.argmax(preds, axis=-1)
        confusion_test.batch_add(targets, pred_labels)

    test_loss = test_err / max(1, test_batches)
    test_acc = confusion_test.accuracy()
    cf_test = confusion_test.ret_mat()

    g_test = gorodkin(cf_test)
    ic_test = IC(cf_test)
    history = {
        'loss': test_loss,
        'cf': cf_test,
        'accuracy': test_acc,
        'gorodkin':g_test,
        'IC': ic_test
    }
    if verbose:
        tqdm.write(f"[RESULTS] test_loss = {test_loss:.6f}")
        tqdm.write(f"[RESULTS] test_acc  = {test_acc*100:.2f}%")
        tqdm.write(f"[RESULTS] Gorodkin  = {g_test:.4f}")
        tqdm.write(f"[RESULTS] IC        = {ic_test:.4f}")

    return history

def main():
    parser = argparse.ArgumentParser(description="Tester un modèle Lasagne sauvegardé en .npz")
    parser.add_argument("--config_path", type=str, default='./config/simple_model.yaml', help="Chemin vers la config YAML utilisée à l'entraînement.")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin vers le fichier .npz du modèle sauvegardé.")
    args = parser.parse_args()

    CONFIG = import_config(args.config_path)
    if CONFIG["dataset"]["test_path"] is None:
        raise ValueError("Aucun test_path dans la config. Impossible d'évaluer.")

    train_npz = np.load(CONFIG["dataset"]["train_path"])
    X_train = train_npz["X_train"]
    y_train = train_npz["y_train"]

    test_npz = np.load(CONFIG["dataset"]["test_path"])
    X_test = test_npz["X_val"]
    y_test = test_npz["y_val"]
    mask_test = test_npz["mask_val"]
    test_data = (X_test, y_test, mask_test)

    uses_mask = CONFIG["model"]["name"] in ["CNN-LSTM", "CNN-LSTM-Attention"]

    # Reconstruction du réseau
    _, val_fn, l_out = build_model_from_name(
        model_name=CONFIG["model"]["name"],
        X_train=X_train,
        y_train=y_train,
        batch_size=CONFIG["training"]["batch_size"],
        n_hid=CONFIG["model"]["n_hid"],
        n_filt=CONFIG["model"]["n_filt"],
        lr=CONFIG["training"]["learning_rate"],
        drop_prob=CONFIG["model"]["drop_prob"]
    )

    tqdm.write(f"[INFO] Modèle {CONFIG["model"]["name"]} reconstruit.")

    # Chargement des poids .npz et injection dans le réseau
    weights = load_npz_weights(args.model_path)
    lasagne.layers.set_all_param_values(l_out, weights)
    tqdm.write(f"[INFO] Poids chargés depuis {args.model_path}.")

    # Évaluation
    history = evaluate_model(val_fn, l_out, test_data, CONFIG["training"]["batch_size"], uses_mask, verbose=args.verbose)

    os.makedirs("Figures", exist_ok=True)
    plt.ioff()

    # === Training curve === #
    plot_training_curves(
            train_losses=history['train_losses'],
            val_losses=history['val_losses'],
            train_accs=history.get('train_accs', None),
            val_accs=history.get('val_accs', None),
            model_name=f'{CONFIG["model"]["name"]}_test_{CONFIG["ID"]}',
            verbose=args.verbose
        )
        
    # === Confusion matrix === #
    cf_val = history['cf_val']
    classes = ['Nucleus','Cytoplasm','Extracellular','Mitochondrion','Cell membrane','ER',
           'Chloroplast','Golgi apparatus','Lysosome','Vacuole']
    plot_confusion_matrix(
                cf_matrix=cf_val,
                classes=classes,
                title=f"Matrice de confusion - {CONFIG["model"]["name"]}",
                model_name=f'{CONFIG["model"]["name"]}_test_{CONFIG["ID"]}',
                verbose=args.verbose
        )

if __name__ == '__main__':
    main()