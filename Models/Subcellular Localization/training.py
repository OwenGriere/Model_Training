import os
import sys
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,optimizer=None,device=cpu,floatX=float32"
sys.path = [p for p in sys.path if ".local/lib/python3.8/site-packages" not in p]
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import theano
import theano.tensor as T
import lasagne
import pandas as pd
from models import FFN, CNN, CNN_LSTM, CNN_LSTM_Att
from confusionmatrix import ConfusionMatrix
from utils import iterate_minibatches, LSTMAttentionDecodeFeedbackLayer
from metrics_mc import gorodkin, IC

##################################################### Helper Function #####################################################
def import_config():
    return 0

def save_params():
    pd.read_parquet("./params/parameter_file")
    return 0

def build_model_from_name(model_name,
                          X_train, y_train,
                          batch_size,
                          n_hid,
                          n_filt,
                          lr,
                          drop_prob):
    """
    Wrapper pour choisir la bonne fonction de création de réseau
    en fonction du nom du modèle.
    Retourne (train_fn, val_fn, l_out).
    """
    if model_name == 'FFN':
        return build_FFN_network(
            X_train, y_train,
            batch_size=batch_size,
            n_hid=n_hid,
            lr=lr,
            drop_prob=drop_prob
        )

    elif model_name == 'CNN':
        return build_CNN_network(
            X_train, y_train,
            batch_size=batch_size,
            n_hid=n_hid,
            n_filt=n_filt,
            lr=lr,
            drop_prob=drop_prob
        )

    elif model_name == 'CNN_LSTM':
        return build_CNN_LSTM_network(
            X_train, y_train,
            batch_size=batch_size,
            n_hid=n_hid,
            n_filt=n_filt,
            lr=lr,
            drop_prob=drop_prob
        )

    elif model_name == 'CNN-LSTM-Attention':
        return build_CNN_LSTM_Att_network(
            X_train, y_train,
            batch_size=batch_size,
            n_hid=n_hid,
            n_filt=n_filt,
            lr=lr,
            drop_prob=drop_prob
        )

    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

##################################################### Build different Network #####################################################

def build_FFN_network(X_train, y_train,
                      batch_size=128,
                      n_hid=30,
                      lr=0.0025,
                      drop_prob=0.5):
    """
    Construit le réseau FFN comme dans le notebook FFN.ipynb.
    Retourne (train_fn, val_fn, l_out).
    """
    # On récupère les dimensions directement depuis les données
    seq_len = X_train.shape[1]
    n_feat = X_train.shape[2]
    n_class = int(np.max(y_train) + 1)

    # Appel à la fonction FFN définie dans models.py
    train_fn, val_fn, l_out = FFN(
        batch_size=batch_size,
        seq_len=seq_len,
        n_hid=n_hid,
        n_feat=n_feat,
        n_class=n_class,
        lr=lr,
        drop_prob=drop_prob
    )

    return train_fn, val_fn, l_out

def build_CNN_network(X_train, y_train,
                      batch_size=128,
                      n_hid=30,
                      n_filt=10,
                      lr=0.005,
                      drop_prob=0.5):
    """
    Construit le réseau CNN comme dans le notebook CNN.ipynb.
    Retourne (train_fn, val_fn, l_out).
    """
    seq_len = X_train.shape[1]
    n_feat = X_train.shape[2]
    n_class = int(np.max(y_train) + 1)

    train_fn, val_fn, l_out = CNN(
        batch_size=batch_size,
        seq_len=seq_len,
        n_hid=n_hid,
        n_feat=n_feat,
        n_class=n_class,
        n_filt=n_filt,
        lr=lr,
        drop_prob=drop_prob
    )

    return train_fn, val_fn, l_out

def build_CNN_LSTM_network(X_train, y_train,
                           batch_size=128,
                           n_hid=15,
                           n_filt=10,
                           lr=0.0025,
                           drop_prob=0.5):
    """
    Construit le réseau CNN-LSTM comme dans le notebook CNN-LSTM.ipynb.
    Retourne (train_fn, val_fn, l_out).
    Ce modèle utilise un masque à l'entraînement et à la validation.
    """
    seq_len = X_train.shape[1]
    n_feat = X_train.shape[2]
    n_class = int(np.max(y_train) + 1)

    train_fn, val_fn, l_out = CNN_LSTM(
        batch_size=batch_size,
        seq_len=seq_len,
        n_hid=n_hid,
        n_feat=n_feat,
        n_class=n_class,
        n_filt=n_filt,
        lr=lr,
        drop_prob=drop_prob
    )

    return train_fn, val_fn, l_out

def build_CNN_LSTM_Att_network(X_train, y_train,
                               batch_size=128,
                               n_hid=15,
                               n_filt=10,
                               lr=0.0025,
                               drop_prob=0.5):
    """
    Construit le réseau CNN-LSTM-Attention comme dans le notebook
    CNN-LSTM-Attention.ipynb.
    Retourne (train_fn, val_fn, l_out).
    Ce modèle utilise un masque et la fonction de validation renvoie
    également les poids d'attention.
    """
    seq_len = X_train.shape[1]
    n_feat = X_train.shape[2]
    n_class = int(np.max(y_train) + 1)

    train_fn, val_fn, l_out = CNN_LSTM_Att(
        batch_size=batch_size,
        seq_len=seq_len,
        n_hid=n_hid,
        n_feat=n_feat,
        n_class=n_class,
        n_filt=n_filt,
        lr=lr,
        drop_prob=drop_prob
    )

    return train_fn, val_fn, l_out

##################################################### PLotting #####################################################

def plot_training_curves(train_losses, val_losses, train_accs=None, val_accs=None, model_name="Model"):
    """
    Trace les courbes de perte et éventuellement d'accuracy,
    dans le style utilisé dans les notebooks.
    """

    plt.figure(figsize=(10, 5))

    # === Courbe de perte === #
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train loss')
    if val_losses:
        plt.plot(val_losses, label='Test loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Courbes de pertes - {model_name}")
    plt.grid(True)
    plt.legend()

    # === Courbe d'accuracy === #
    if train_accs is not None and val_accs is not None:
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train acc')
        plt.plot(val_accs, label='Test acc')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy - {model_name}")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    loss_path = f"Figures/{model_name}_loss_and_accuracy.png"
    plt.savefig(loss_path)
    tqdm.write(f"[INFO] Taining curves saved: {loss_path}")
    plt.close()

def plot_confusion_matrix_(cf_matrix, classes=None, title="Matrice de confusion", model_name="Model", Norm=True):
    """
    Reproduit le style d'affichage des notebooks pour la matrice de confusion.
    cf_matrix : matrice numpy NxN retournée par ConfusionMatrix.ret_mat()
    """

    n_class = cf_matrix.shape[0]

    if classes is None:
        classes = [str(i) for i in range(n_class)]
    if Norm:
        cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(cf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(n_class)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cf_matrix.max() / 2

    for i in range(n_class):
        for j in range(n_class):
            plt.text(j, i, cf_matrix[i, j],
                     horizontalalignment="center",
                     color="white" if cf_matrix[i, j] > thresh else "black")

    plt.ylabel("Label réel")
    plt.xlabel("Label prédit")
    plt.tight_layout()
    cf_path = f"Figures/{model_name}_confusion.png"
    plt.savefig(cf_path)
    plt.close()
    tqdm.write(f"[INFO] Confusion matrix saved: {cf_path}")

##################################################### Model Training #####################################################

def train_model(model_name,
                train_data,
                test_data=None,
                batch_size=32,
                num_epochs=20,
                lr=0.001,
                n_hid=128,
                n_filt=64,
                drop_prob=0.5,
                save_params_name=None,
                verbose=False):

    # Dépaquetage des données
    X_train, y_train, mask_train = train_data
    if test_data is not None:
        X_test, y_test, mask_test = test_data
    else:
        X_test = y_test = mask_test = None

    # Infos sur les shapes et le nombre de classes
    seq_len = X_train.shape[1]
    n_feat = X_train.shape[2]
    n_class = int(np.max(y_train) + 1)

    tqdm.write(f"[INFO] Train shape: {X_train.shape}, n_class: {n_class}")
    if X_test is not None:
        tqdm.write(f"[INFO] Test shape: {X_test.shape}, n_class: {n_class}")

    # Les modèles LSTM consomment un masque
    uses_mask = model_name in ['CNN_LSTM', 'CNN-LSTM-Attention']

    
    train_fn, val_fn, l_out = build_model_from_name(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        batch_size=batch_size,
        n_hid=n_hid,
        n_filt=n_filt,
        lr=lr,
        drop_prob=drop_prob
    )
    tqdm.write(f'[INFO] {model_name} model build with success\n')

    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    best_params = None

    with tqdm(total=num_epochs, desc=f"[TRAINING] {model_name} training phase", ncols=90) as pbar:
        for epoch in range(1, num_epochs + 1):
            # ============================
            #      PHASE TRAINING
            # ============================
            train_err = 0.0
            train_batches = 0
            confusion_train = ConfusionMatrix(n_class)

            for inputs, targets, in_masks in iterate_minibatches(
                X_train, y_train, mask_train,
                batchsize=batch_size,
                shuffle=True,
                sort_len=uses_mask):

                inputs = inputs.astype('float32')
                targets = targets.astype('int32')

                if uses_mask and in_masks is not None:
                    in_masks = in_masks.astype('float32')
                    loss_batch, preds = train_fn(inputs, targets, in_masks)
                else:
                    loss_batch, preds = train_fn(inputs, targets)

                train_err += float(loss_batch)
                train_batches += 1
                pred_labels = np.argmax(preds, axis=-1)
                confusion_train.batch_add(targets, pred_labels)

            train_loss = train_err / max(1, train_batches)
            train_losses.append(train_loss)
            train_acc = confusion_train.accuracy()
            cf_train = confusion_train.ret_mat()

            # ============================
            #   PHASE TEST / VALIDATION
            # ============================
            if X_test is not None:
                val_err = 0.0
                val_batches = 0
                confusion_valid = ConfusionMatrix(n_class)

                for inputs, targets, in_masks in iterate_minibatches(
                    X_test, y_test, mask_test,
                    batchsize=batch_size,
                    shuffle=False,
                    sort_len=False):
                    inputs = inputs.astype('float32')
                    targets = targets.astype('int32')

                    if uses_mask and in_masks is not None:
                        in_masks = in_masks.astype('float32')
                        out = val_fn(inputs, targets, in_masks)
                    else:
                        out = val_fn(inputs, targets)

                    # CNN_LSTM_Att peut renvoyer (loss, preds, alphas)
                    val_loss_batch = float(out[0])
                    preds = out[1]
                    val_err += val_loss_batch
                    val_batches += 1
                    pred_labels = np.argmax(preds, axis=-1)
                    confusion_valid.batch_add(targets, pred_labels)

                val_loss = val_err / max(1, val_batches)
                val_losses.append(val_loss)
                val_acc = confusion_valid.accuracy()
                cf_val = confusion_valid.ret_mat()
                if verbose:
                    tqdm.write(
                        f"Epoch {epoch:02d}/{num_epochs} | "
                        f"train_loss {train_loss:.6f} | test_loss {val_loss:.6f} | "
                        f"train_acc {train_acc*100:.2f}% | test_acc {val_acc*100:.2f}%"
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = lasagne.layers.get_all_param_values(l_out)
                    if save_params_name is not None:
                        os.makedirs('params', exist_ok=True)
                        np.savez(os.path.join('params', save_params_name), *best_params)

                if verbose:
                    tqdm.write("  training Gorodkin:\t{:.2f}".format(gorodkin(cf_train)))
                    tqdm.write("  validation Gorodkin:\t{:.2f}".format(gorodkin(cf_val)))
                    tqdm.write("  training IC:\t\t{:.2f}".format(IC(cf_train)))
                    tqdm.write("  validation IC:\t{:.2f}".format(IC(cf_val)))

            else:
                if verbose:
                    tqdm.write(
                        f"Epoch {epoch:02d}/{num_epochs} | "
                        f"train_loss {train_loss:.6f} | train_acc {train_acc*100:.2f}%"
                    )

            pbar.update(1)

    # ============================
    #   SAUVEGARDE FINALE
    # ============================
    if save_params_name is None:
        save_params_name = f"{model_name}_params.npz"

    if best_params is not None:
        np.savez(os.path.join('params', save_params_name), *best_params)
    else:
        # Si pas de test set ou pas d'amélioration, on sauve l'état final
        np.savez(
            os.path.join('params', save_params_name),
            *lasagne.layers.get_all_param_values(l_out)
        )

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'cf_val': cf_val,
        'cf_train': cf_train,
        'train_acc':train_acc,
        'val_acc': val_acc
    }
    return l_out, history

##################################################### Main #####################################################

def main():
    parser = argparse.ArgumentParser(description="Lancer l'entraînement d'un modèle du projet Subcellular Localization")
    
    parser.add_argument('--model', type=str,choices=['FFN', 'CNN', 'CNN_LSTM', 'CNN-LSTM-Attention'],default='CNN',help='Nom du modèle à entraîner')
    parser.add_argument('-i', '--trainset',help="Fichier .npz contenant les données d'entraînement (X_train, y_train, mask_train)")
    parser.add_argument('-t', '--testset', help="Fichier .npz contenant les données de test (X_test, y_test, mask_test) pour l'évaluation")
    parser.add_argument('--epochs', type=int, default=20, help='Nombre d’epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Taille de mini-batch')
    parser.add_argument('--lr', type=float, default=0.001, help='Taux d’apprentissage')
    parser.add_argument('--n_hid', type=int, default=128, help='Nombre de neurones cachés')
    parser.add_argument('--n_filt', type=int, default=64, help='Nombre de filtres convolutifs')
    parser.add_argument('--drop', type=float, default=0.5, help='Taux de dropout')
    parser.add_argument('--no_plot', action='store_true', help='Ne pas afficher les courbes de perte')
    parser.add_argument('--verbose', action='store_true', help='Ne pas afficher les infos de training')
    args = parser.parse_args()

    # === Chargement des données d'entraînement === #
    train_npz = np.load(args.trainset)
    X_train = train_npz['X_train']
    y_train = train_npz['y_train']
    mask_train = train_npz['mask_train']
    train_data = (X_train, y_train, mask_train)

    test_data = None
    if args.testset is not None:
        test_npz = np.load(args.testset)
        X_test = test_npz['X_val']
        y_test = test_npz['y_val']
        mask_test = test_npz['mask_val']
        test_data = (X_test, y_test, mask_test)

    model_label = args.model.replace('-', '_')
    params_fname = f"{model_label}_params.npz"

    # === Entraînement du modèle choisi === #
    l_out, history = train_model(
        model_name=args.model,
        train_data=train_data,
        test_data=test_data,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        n_hid=args.n_hid,
        n_filt=args.n_filt,
        drop_prob=args.drop,
        save_params_name=params_fname,
        verbose=args.verbose)

    tqdm.write(f"[INFO] Training finished\n")

    if not args.no_plot:
        os.makedirs("Figures", exist_ok=True)
        plt.ioff()

        ### Training curve ###
        plot_training_curves(
            train_losses=history['train_losses'],
            val_losses=history['val_losses'],
            train_accs=history.get('train_accs', None),
            val_accs=history.get('val_accs', None),
            model_name=args.model
        )
        
        ### Confusion matrix ###
        cf_val = history['cf_val']
        classes = ['Nucleus','Cytoplasm','Extracellular','Mitochondrion','Cell membrane','ER',
           'Chloroplast','Golgi apparatus','Lysosome','Vacuole']
        plot_confusion_matrix_(
                cf_matrix=cf_val,
                classes=classes,
                title=f"Matrice de confusion - {args.model}",
                model_name=args.model
        )

if __name__ == '__main__':
    main()