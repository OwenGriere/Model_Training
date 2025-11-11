#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_models.py
-------------------------------------------------------
Script unifié pour entraîner et évaluer les modèles
du projet "Subcellular Localization".

Modèles pris en charge :
 - FFN
 - CNN
 - CNN_LSTM
 - CNN_LSTM_Att (avec module d’attention)

Ce script réutilise les fonctions de construction
de réseaux définies dans models.py (Theano + Lasagne).
-------------------------------------------------------
Usage :
    python run_models.py --model CNN_LSTM_Att --epochs 30 --batch_size 32
-------------------------------------------------------
"""

import os
import sys
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,optimizer=None,device=cpu,floatX=float32"
sys.path = [p for p in sys.path if ".local/lib/python3.8/site-packages" not in p]
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools

# Imports du dépôt (Theano / Lasagne)
import theano
import lasagne
from models import FFN, CNN, CNN_LSTM, CNN_LSTM_Att


# ---------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------
def save_params(l_out, fname):
    """
    Sauvegarde les paramètres du réseau dans le dossier 'params/'.
    Les paramètres sont récupérés depuis la couche de sortie l_out.
    """
    os.makedirs('params', exist_ok=True)
    params = lasagne.layers.get_all_param_values(l_out)
    np.savez(os.path.join('params', fname), *params)

def plot_losses(train_losses, val_losses, model_name):
    """
    Trace les courbes de perte d'entraînement et de test/validation en fonction des epochs.
    Le vecteur val_losses peut éventuellement être vide si aucun ensemble de test
    n'est utilisé pendant l'entraînement.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train')

    if len(val_losses) > 0:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Test')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Courbes de perte - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_confusion_and_plot(y_true, y_pred, classes=None, title='Matrice de confusion'):
    """
    Calcule l’accuracy globale et affiche une matrice de confusion simple
    entre les labels réels y_true et les labels prédits y_pred.
    """
    from sklearn.metrics import confusion_matrix, accuracy_score

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    if classes is None:
        classes = [str(i) for i in range(cm.shape[0])]

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.ylabel('Label réel')
    plt.xlabel('Label prédit')
    plt.tight_layout()
    plt.show()

def train_model(model_name,
                train_data,
                test_data=None,
                batch_size=32,
                num_epochs=20,
                lr=0.001,
                n_hid=128,
                n_filt=64,
                drop_prob=0.5,
                save_params_name=None):
    """
    Entraîne et évalue un modèle choisi parmi FFN, CNN, CNN_LSTM ou CNN_LSTM_Att.

    Paramètres
    ----------
    model_name : str
        Nom du modèle ('FFN', 'CNN', 'CNN_LSTM', 'CNN-LSTM-Attention').
    train_data : tuple
        Tuple (X_train, y_train, mask_train).
    test_data : tuple ou None
        Tuple (X_test, y_test, mask_test). Sert ici de jeu de validation/test
        évalué à chaque epoch.
    batch_size : int
        Taille des mini-batchs.
    num_epochs : int
        Nombre d'epochs.
    lr : float
        Taux d'apprentissage.
    n_hid : int
        Nombre de neurones cachés.
    n_filt : int
        Nombre de filtres convolutifs (pour les modèles CNN / CNN_LSTM).
    drop_prob : float
        Taux de dropout.
    save_params_name : str ou None
        Nom du fichier .npz où sauvegarder les meilleurs paramètres.

    Retour
    ------
    l_out : lasagne.layers.Layer
        Couche de sortie du réseau.
    history : dict
        Dictionnaire contenant 'train_losses' et 'val_losses'.
    """

    # Dépaquetage des données d'entraînement
    X_train, y_train, mask_train = train_data

    # Dépaquetage des données de test (si fournies)
    if test_data is not None:
        X_test, y_test, mask_test = test_data
    else:
        X_test, y_test, mask_test = None, None, None

    seq_len = X_train.shape[1]
    n_feat = X_train.shape[2]
    n_class = int(np.max(y_train) + 1)

    tqdm.write(f"Train shape: {X_train.shape}, n_class: {n_class}")

    # Sélection du modèle à partir du nom choisi
    if model_name == 'FFN':
        model_fn = FFN
        model_args = (batch_size, seq_len, n_hid, n_feat, n_class, lr, drop_prob)
    elif model_name == 'CNN':
        model_fn = CNN
        model_args = (batch_size, seq_len, n_hid, n_feat, n_class, n_filt, lr, drop_prob)
    elif model_name == 'CNN_LSTM':
        model_fn = CNN_LSTM
        model_args = (batch_size, seq_len, n_hid, n_feat, n_class, n_filt, lr, drop_prob)
    elif model_name == 'CNN-LSTM-Attention':
        model_fn = CNN_LSTM_Att
        model_args = (batch_size, seq_len, n_hid, n_feat, n_class, n_filt, lr, drop_prob)
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    # Construction du modèle choisi
    train_fn, val_fn, l_out = model_fn(*model_args)

    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    best_params = None

    n_samples = X_train.shape[0]
    nbatches = max(1, n_samples // batch_size)

    # Boucle d'entraînement principale
    with tqdm(total=num_epochs, desc=f"Training {model_name}", ncols=90) as pbar:
        for epoch in range(1, num_epochs + 1):
            # Mélange des données d'entraînement
            perm = np.random.permutation(n_samples)
            X_train_sh = X_train[perm]
            y_train_sh = y_train[perm]
            mask_train_sh = mask_train[perm] if mask_train is not None else None

            epoch_train_losses = []

            # Parcours des mini-batchs
            for b in range(nbatches):
                start = b * batch_size
                end = (b + 1) * batch_size
                xb = X_train_sh[start:end]
                yb = y_train_sh[start:end]

                if mask_train_sh is not None:
                    mb = mask_train_sh[start:end]
                    res = train_fn(
                        xb.astype('float32'),
                        yb.astype('int32'),
                        mb.astype('float32')
                    )
                else:
                    res = train_fn(
                        xb.astype('float32'),
                        yb.astype('int32')
                    )

                loss = res[0]
                epoch_train_losses.append(float(loss))

            train_loss = float(np.mean(epoch_train_losses))
            train_losses.append(train_loss)

            # Évaluation sur le jeu de test si disponible
            if X_test is not None:
                if mask_test is not None:
                    out = val_fn(
                        X_test.astype('float32'),
                        y_test.astype('int32'),
                        mask_test.astype('float32')
                    )
                else:
                    out = val_fn(
                        X_test.astype('float32'),
                        y_test.astype('int32')
                    )

                # out peut être (loss, preds) ou (loss, preds, alphas)
                val_loss = float(out[0])
                val_losses.append(val_loss)

                tqdm.write(
                    f"Epoch {epoch:02d}/{num_epochs} | "
                    f"train_loss {train_loss:.6f} | "
                    f"test_loss {val_loss:.6f}"
                )

                # Mise à jour du meilleur modèle
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = lasagne.layers.get_all_param_values(l_out)
                    if save_params_name is not None:
                        os.makedirs('params', exist_ok=True)
                        np.savez(os.path.join('params', save_params_name), *best_params)
            else:
                tqdm.write(
                    f"Epoch {epoch:02d}/{num_epochs} | "
                    f"train_loss {train_loss:.6f}"
                )

            pbar.update(1)

    # Sauvegarde finale des paramètres
    if save_params_name is None:
        save_params_name = f"{model_name}_params.npz"

    if best_params is not None:
        np.savez(os.path.join('params', save_params_name), *best_params)
    else:
        np.savez(
            os.path.join('params', save_params_name),
            *lasagne.layers.get_all_param_values(l_out)
        )

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    return l_out, history

def main():
    parser = argparse.ArgumentParser(
        description="Lancer l'entraînement d'un modèle du projet Subcellular Localization."
    )
    parser.add_argument(
        '--model', type=str,
        choices=['FFN', 'CNN', 'CNN_LSTM', 'CNN-LSTM-Attention'],
        default='CNN',
        help='Nom du modèle à entraîner.'
    )
    parser.add_argument(
        '-i', '--trainset',
        help="Fichier .npz contenant les données d'entraînement (X_train, y_train, mask_train)."
    )
    parser.add_argument(
        '-t', '--testset',
        help="Fichier .npz contenant les données de test (X_test, y_test, mask_test) pour l'évaluation."
    )
    parser.add_argument('--epochs', type=int, default=20, help='Nombre d’epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Taille de mini-batch.')
    parser.add_argument('--lr', type=float, default=0.001, help='Taux d’apprentissage.')
    parser.add_argument('--n_hid', type=int, default=128, help='Nombre de neurones cachés.')
    parser.add_argument('--n_filt', type=int, default=64, help='Nombre de filtres convolutifs.')
    parser.add_argument('--drop', type=float, default=0.5, help='Taux de dropout.')
    parser.add_argument('--no_plot', action='store_true', help='Ne pas afficher les courbes de perte.')
    args = parser.parse_args()

    # Chargement des données d'entraînement
    train_npz = np.load(args.trainset)
    X_train = train_npz['X_train']
    y_train = train_npz['y_train']
    mask_train = train_npz['mask_train']

    train_data = (X_train, y_train, mask_train)

    # Chargement des données de test
    test_data = None
    if args.testset is not None:
        test_npz = np.load(args.testset)
        X_test = test_npz['X_test']
        y_test = test_npz['y_test']
        mask_test = test_npz['mask_test']
        test_data = (X_test, y_test, mask_test)

    model_label = args.model.replace('-', '_')
    params_fname = f"{model_label}_params.npz"

    # Entraînement du modèle choisi
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
        save_params_name=params_fname
    )

    # Tracé des pertes
    if not args.no_plot:
        plot_losses(history['train_losses'], history['val_losses'], args.model)

    print(f"Entraînement terminé. Paramètres sauvegardés dans 'params/{params_fname}'.")

if __name__ == '__main__':
    main()
