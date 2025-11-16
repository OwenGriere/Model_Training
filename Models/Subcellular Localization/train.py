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

from mypackage.models import FFN, CNN, CNN_LSTM, CNN_LSTM_Att
from mypackage.confusionmatrix import ConfusionMatrix
from mypackage.utils import iterate_minibatches, LSTMAttentionDecodeFeedbackLayer
from mypackage.metrics_mc import gorodkin, IC

##################################################### Helper Function #####################################################
def ensure_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    else:
        return [x]

def import_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def import_params(path, model):
    if os.path.exists(path):
        df = pd.read_parquet(path)
        return df
    else:
        df = pd.DataFrame(columns=model)
        return df

def save_params(df, **params):
    df.loc[len(df)] = params
    return None

def save_model(params, path):
    values = [p.get_value() for p in params]
    np.savez(path, *values)

##################################################### Build different Network #####################################################

def build_model_from_name(model_name,
                          X_train, y_train,
                          batch_size,
                          n_hid,
                          n_filt,
                          lr,
                          drop_prob):

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

    elif model_name == 'CNN-LSTM':
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

def build_FFN_network(X_train, y_train,
                      batch_size=128,
                      n_hid=30,
                      lr=0.0025,
                      drop_prob=0.5):
    
    seq_len = X_train.shape[1]
    n_feat = X_train.shape[2]
    n_class = int(np.max(y_train) + 1)


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

def plot_training_curves(train_losses, val_losses, train_accs=None, val_accs=None, model_name="Model", verbose=False):
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

    plt.tight_layout()
    loss_path = f"Figures/{model_name}_loss_and_accuracy.png"
    plt.savefig(loss_path)
    plt.close()
    if verbose:
        tqdm.write(f"[INFO] Taining curves saved: {loss_path}")

def plot_confusion_matrix_(cf_matrix, classes=None, title="Matrice de confusion", model_name="Model", Norm=True, verbose=False):

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

    plt.ylabel("Label réel")
    plt.xlabel("Label prédit")
    plt.tight_layout()
    cf_path = f"Figures/{model_name}_confusion.png"
    plt.savefig(cf_path)
    plt.close()
    if verbose:
        tqdm.write(f"[INFO] Confusion matrix saved: {cf_path}")

##################################################### Model Training #####################################################

def train_model(ID, model_name, 
                train_data,
                test_data=None,
                batch_size=32,
                num_epochs=20,
                lr=0.001,
                n_hid=128,
                n_filt=64,
                drop_prob=0.5,
                save_params_frame=None,
                verbose=False,
                all_verbose=False,
                early_stopping=True,
                patience=10,
                min_delta=0.0):

    X_train, y_train, mask_train = train_data
    if test_data is not None:
        X_test, y_test, mask_test = test_data
    else:
        X_test = y_test = mask_test = None

    seq_len = X_train.shape[1]
    n_feat = X_train.shape[2]
    n_class = int(np.max(y_train) + 1)
    if verbose:
        tqdm.write(f"[INFO] Train shape: {X_train.shape}, n_class: {n_class}")
        if X_test is not None:
            tqdm.write(f"[INFO] Test shape: {X_test.shape}, n_class: {n_class}")

    # Pour les models utilisant un mask
    uses_mask = model_name in ['CNN-LSTM', 'CNN-LSTM-Attention']

    
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
    if verbose or all_verbose:
        tqdm.write(f'[INFO] {model_name} model build with success\n')

    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    best_epoch = 0
    epochs_no_improve = 0
    cf_val = None
    best_params_values = None 

    with tqdm(total=num_epochs, desc=f"[TRAINING] ID={ID} | {model_name}", ncols=150) as pbar:
        
        pbar.set_postfix({
                "bs": batch_size,
                "lr": lr,
                "hid": n_hid,
                "filt": n_filt,
                "drop": drop_prob
            })
        
        for epoch in range(1, num_epochs + 1):
            # === TRAINING du model === #

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

            # ===  PHASE TEST / VALIDATION ===
            
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
                if all_verbose:
                    tqdm.write(
                        f"Epoch {epoch:02d}/{num_epochs} | "
                        f"train_loss {train_loss:.6f} | test_loss {val_loss:.6f} | "
                        f"train_acc {train_acc*100:.2f}% | test_acc {val_acc*100:.2f}%"
                    )
                # === EARLY-STOPPING === #
                if early_stopping:
                    if val_loss < (best_val_loss - min_delta):
                        best_val_loss = val_loss
                        best_params_values = lasagne.layers.get_all_param_values(l_out)
                        best_epoch = epoch
                        epochs_no_improve = 0

                    else:
                        epochs_no_improve += 1

                        if epochs_no_improve >= patience:
                            if verbose or all_verbose:
                                tqdm.write(
                                    f"[EARLY STOPPING] Stopping at epoch {epoch} "
                                    f"(best epoch = {best_epoch}, best val_loss = {best_val_loss:.6f})\n"
                                )
                            pbar.update(1)
                            break

                if all_verbose:
                    tqdm.write("  training Gorodkin:\t{:.2f}".format(gorodkin(cf_train)))
                    tqdm.write("  validation Gorodkin:\t{:.2f}".format(gorodkin(cf_val)))
                    tqdm.write("  training IC:\t\t{:.2f}".format(IC(cf_train)))
                    tqdm.write("  validation IC:\t{:.2f}".format(IC(cf_val)))

                if save_params_frame is not None:
                    g_train = gorodkin(cf_train)
                    g_val   = gorodkin(cf_val)
                    ic_train = IC(cf_train)
                    ic_val   = IC(cf_val)

                    # === confusion_score = g_val

                    save_params(
                        save_params_frame,
                        ID=f"{ID}_{model_name}_{epoch}",
                        model_name=model_name,
                        epoch=epoch,
                        batch_size=batch_size,
                        lr=lr,
                        n_hid=n_hid,
                        n_filt=n_filt if model_name != 'FFN' else np.nan,
                        drop_prob=drop_prob,
                        seq_len=seq_len,
                        n_feat=n_feat,
                        n_class=n_class,
                        uses_mask=uses_mask,
                        attention=(model_name == 'CNN-LSTM-Attention'),
                        train_loss=train_loss,
                        val_loss=val_loss,
                        train_acc=train_acc,
                        val_acc=val_acc,
                        gorodkin_train=g_train,
                        gorodkin_val=g_val,
                        IC_train=ic_train,
                        IC_val=ic_val
                    )
            else:
                if all_verbose:
                    tqdm.write(
                        f"Epoch {epoch:02d}/{num_epochs} | "
                        f"train_loss {train_loss:.6f} | train_acc {train_acc*100:.2f}%"
                    )

            pbar.update(1)

    if early_stopping and best_params_values is not None:
        lasagne.layers.set_all_param_values(l_out, best_params_values)
        
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

def main(ID, model_name, batch_size, num_epochs, lr, n_hid, n_filt, drop_prob, train_data, test_data):

    # === Entraînement du modèle choisi === #
    l_out, history = train_model(ID,
        model_name=model_name,
        train_data=train_data,
        test_data=test_data,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        n_hid=n_hid,
        n_filt=n_filt,
        drop_prob=drop_prob,
        save_params_frame=params_frame,
        verbose=args.verbose,
        all_verbose=args.all_verbose)
    
    if args.verbose or args.all_verbose:
        tqdm.write(f"[INFO] Training finished\n")

    params_frame.to_parquet(args.params_path)

    if args.verbose or args.all_verbose:
        tqdm.write(f"[INFO] Parameters saved\n")

    if not args.no_plot:
        os.makedirs("Figures", exist_ok=True)
        plt.ioff()

        # === Training curve === #
        plot_training_curves(
            train_losses=history['train_losses'],
            val_losses=history['val_losses'],
            train_accs=history.get('train_accs', None),
            val_accs=history.get('val_accs', None),
            model_name=f'{model_name}_{ID}',
            verbose=args.all_verbose
        )
        
        # === Confusion matrix === #
        cf_val = history['cf_val']
        classes = ['Nucleus','Cytoplasm','Extracellular','Mitochondrion','Cell membrane','ER',
           'Chloroplast','Golgi apparatus','Lysosome','Vacuole']
        plot_confusion_matrix_(
                cf_matrix=cf_val,
                classes=classes,
                title=f"Matrice de confusion - {model_name}",
                model_name=f'{model_name}_{ID}',
                verbose=args.all_verbose
        )
    return lasagne.layers.get_all_params(l_out) 

if __name__ == '__main__':

    cols = ["ID",
        "model_name", "epoch",
        "batch_size", "lr", "n_hid", "n_filt", "drop_prob",
        "seq_len", "n_feat", "n_class",
        "uses_mask", "attention",
        "train_loss", "val_loss",
        "train_acc", "val_acc",
        "gorodkin_train", "gorodkin_val",
        "IC_train", "IC_val"
    ]

    # === Chargement des parametres === #
    parser = argparse.ArgumentParser(description="Lancer l'entraînement d'un modèle du projet Subcellular Localization")

    parser.add_argument('-c', '--config_path', type=str, default='./config/simple_model.yaml', help='Add config path')
    parser.add_argument('-p', '--params_path', type=str, default='./params/params.parquet', help='Add saving path for parameters in .parquet')
    parser.add_argument('--multimodel', action='store_true',help='Compute the main for mulimodel contained in multimodels.yaml')
    parser.add_argument('--save', action='store_true',help='Save the model in ./models')
    parser.add_argument('--no_plot', action='store_true',help='Use that to force the unplotting')
    parser.add_argument('--verbose', action='store_true', help='Ne pas afficher les infos générales')
    parser.add_argument('--all_verbose', action='store_true', help='Ne pas afficher les infos de training')
    args = parser.parse_args()

    params_frame = import_params(args.params_path, model=cols)
    save_path = "./models"
    os.makedirs(save_path, exist_ok=True)

    # === Chargement des configs des models === #

    if args.multimodel:
        CONFIG = import_config('./config/multimodels.yaml')
    else:
        CONFIG = import_config(args.config_path)
    ID = CONFIG["ID"]

    # === Chargement des données d'entraînement et de validation === #
        
    train_npz = np.load(CONFIG["dataset"]["train_path"])
    X_train = train_npz['X_train']
    y_train = train_npz['y_train']
    mask_train = train_npz['mask_train']
    train_data = (X_train, y_train, mask_train)

    test_data = None
    if CONFIG["dataset"]["test_path"] is not None:
        test_npz = np.load(CONFIG["dataset"]["test_path"])
        X_test = test_npz['X_val']
        y_test = test_npz['y_val']
        mask_test = test_npz['mask_val']
        test_data = (X_test, y_test, mask_test)

    # === Multimodel === #

    if args.multimodel:
        model_names  = ensure_list(CONFIG["model"]["name"])
        n_hids       = ensure_list(CONFIG["model"]["n_hid"])
        n_filts      = ensure_list(CONFIG["model"]["n_filt"])
        drop_probs   = ensure_list(CONFIG["model"]["drop_prob"])
        batch_sizes  = ensure_list(CONFIG["training"]["batch_size"])
        epochs_list  = ensure_list(CONFIG["training"]["epochs"])
        lrs          = ensure_list(CONFIG["training"]["learning_rate"])

        hyper_grid = list(itertools.product(model_names,batch_sizes,epochs_list,
            lrs,n_hids,n_filts,drop_probs))
        
        if args.save:
            with tqdm(total=len(hyper_grid), desc="[MULTIMODEL] Run for model n°", ncols=150) as pbar:
                for run_idx, (model_name, batch_size, epochs, lr, n_hid, n_filt, drop_prob) in enumerate(hyper_grid):

                    params = main(ID,model_name,batch_size,epochs,
                        lr,n_hid,n_filt,drop_prob,
                        train_data,test_data
                    )
                    file_path = os.path.join(save_path, f"{ID}_{CONFIG['model']['name']}.npz")
                    save_model(params, file_path)

                    ID = str(int(ID) + 1).zfill(len(ID))
                    pbar.update(1)

        else:
            with tqdm(total=len(hyper_grid), desc="[MULTIMODEL] Run for model - ", ncols=150) as pbar:
                for run_idx, (model_name, batch_size, epochs, lr, n_hid, n_filt, drop_prob) in enumerate(hyper_grid):

                    _ = main(ID,model_name,batch_size,epochs,
                        lr,n_hid,n_filt,drop_prob,
                        train_data,test_data
                    )

                    ID = str(int(ID) + 1).zfill(len(ID))
                    pbar.update(1)

    # === Single model === #

    else:
        if args.save:
            params = main(ID, CONFIG["model"]["name"], CONFIG["training"]["batch_size"], 
                CONFIG["training"]["epochs"], CONFIG["training"]["learning_rate"], 
                CONFIG["model"]["n_hid"], CONFIG["model"]["n_filt"], CONFIG["model"]["drop_prob"],
                train_data, test_data )
            
            file_path = os.path.join(save_path, f"{ID}_{CONFIG['model']['name']}.npz")
            save_model(params, file_path)
        else:
            _ = main(ID, CONFIG["model"]["name"], CONFIG["training"]["batch_size"], 
                CONFIG["training"]["epochs"], CONFIG["training"]["learning_rate"], 
                CONFIG["model"]["n_hid"], CONFIG["model"]["n_filt"], CONFIG["model"]["drop_prob"],
                train_data, test_data )


    