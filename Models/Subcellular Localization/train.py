import os
import sys
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,optimizer=None,device=cpu,floatX=float32"
sys.path = [p for p in sys.path if ".local/lib/python3.8/site-packages" not in p]
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import itertools
import lasagne

from mypackage.building_models import *
from mypackage.confusionmatrix import ConfusionMatrix
from mypackage.utils import iterate_minibatches, import_config, import_params, EarlyStopping, LSTMAttentionDecodeFeedbackLayer
from mypackage.metrics_mc import gorodkin, IC
from mypackage.plotting import *

##################################################### Helper Function #####################################################
def ensure_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    else:
        return [x]

def save_params(df, **params):
    df.loc[len(df)] = params
    return None

def save_model(params, path):
    values = [p.get_value() for p in params]
    np.savez(path, *values)

##################################################### Model Training #####################################################

def train_model(ID, model_name, 
                train_data,
                val_data=None,
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
    if val_data is not None:
        X_val, y_val, mask_val = val_data
    else:
        X_val = y_val = mask_val = None

    seq_len = X_train.shape[1]
    n_feat = X_train.shape[2]
    n_class = int(np.max(y_train) + 1)
    if verbose:
        tqdm.write(f"[INFO] Train shape: {X_train.shape}, n_class: {n_class}")
        if X_val is not None:
            tqdm.write(f"[INFO] Test shape: {X_val.shape}, n_class: {n_class}")

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
    cf_val = None

    stopper = None
    if early_stopping:
        stopper = EarlyStopping(patience=patience, min_delta=min_delta)
    
    best_score = np.inf
    best_record = None

    for epoch in range(1, num_epochs + 1):
        tqdm.write(
            f"[EPOCH {epoch}/{num_epochs}]\t{model_name} | "
            f"bs={batch_size} | lr={lr} | hid={n_hid} | filt={n_filt} | drop={drop_prob}"
        )
            
        # === TRAINING du model === #

        train_err = 0.0
        train_batches = 0
        confusion_train = ConfusionMatrix(n_class)

        train_iter = list(iterate_minibatches(
        X_train, y_train, mask_train,
        batchsize=batch_size,
        shuffle=True,
        sort_len=uses_mask
        ))

        with tqdm(total=len(train_iter), desc=f"[TRAINING] Traited batch", leave=False, ncols=90) as pbar:
            for inputs, targets, in_masks in train_iter:

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
                pbar.update(1)

        train_loss = train_err / max(1, train_batches)
        train_losses.append(train_loss)
        train_acc = confusion_train.accuracy()
        cf_train = confusion_train.ret_mat()

        # ===  PHASE VALIDATION ===
            
        if X_val is not None:
            val_err = 0.0
            val_batches = 0
            confusion_valid = ConfusionMatrix(n_class)

            val_iter = list(iterate_minibatches(
                X_val, y_val, mask_val,
                batchsize=batch_size,
                shuffle=False,
                sort_len=False
                ))
            
            with tqdm(total=len(val_iter), desc=f"[VALIDATION] Traited batch", leave=False, ncols=90) as pbar:
                for inputs, targets, in_masks in val_iter:
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
                    pbar.update(1)

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
                stop_now = stopper.step(val_loss, l_out, epoch)
                if stop_now:
                    if verbose or all_verbose:
                        tqdm.write(
                                f"[EARLY STOPPING] Stop at epoch {epoch}. "
                                f"Best epoch = {stopper.best_epoch}, best val_loss = {stopper.best_loss:.6f}\n"
                        )
                    if args.multimodel:
                        print("\033[2K\033[1G", end="", flush=True)
                    break

        else:
            # === Pas de validation === #
            val_loss = np.nan
            val_losses.append(np.nan)
            val_acc = np.nan
            cf_val = None
            if all_verbose:
                tqdm.write(
                        f"Epoch {epoch:02d}/{num_epochs} | "
                        f"train_loss {train_loss:.6f} | train_acc {train_acc*100:.2f}%"
                    )
                tqdm.write("  training Gorodkin:\t{:.2f}".format(gorodkin(cf_train)))
                tqdm.write("  training IC:\t\t{:.2f}".format(IC(cf_train)))
            if early_stopping:
                stop_now = stopper.step(train_loss, l_out, epoch)
                if stop_now:
                    if verbose or all_verbose:
                        tqdm.write(
                                f"[EARLY STOPPING] Stop at epoch {epoch}. "
                                f"Best epoch = {stopper.best_epoch}, best train_loss = {stopper.best_loss:.6f}\n"
                            )
                    if args.multimodel:
                        print("\033[2K\033[1G", end="", flush=True)
                    break
                # On choisit la loss à surveiller
        monitored_loss = val_loss if X_val is not None else train_loss

        
        if save_params_frame is not None:
            if monitored_loss < best_score - min_delta:
                best_score = monitored_loss

                g_train = gorodkin(cf_train)
                ic_train = IC(cf_train)

                g_val = gorodkin(cf_val) if X_val is not None else None
                ic_val = IC(cf_val) if X_val is not None else None

                best_record = dict(
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
        if args.multimodel:
            print("\033[2K\033[1G", end="", flush=True)

    if save_params_frame is not None and best_record is not None:
        save_params(save_params_frame, **best_record)      

    if early_stopping:
        stopper.restore_best_weights(l_out)

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'cf_val': cf_val         if X_val is not None else None,
        'cf_train': cf_train,
        'train_acc':train_acc,
        'val_acc': val_acc       if X_val is not None else None
    }

    return l_out, history

##################################################### Main #####################################################

def main(ID, model_name, batch_size, num_epochs, lr, n_hid, n_filt, drop_prob, train_data, test_data, early_stop, patience=None, min_delta=None):

    # === Entraînement du modèle choisi === #
    l_out, history = train_model(ID,
        model_name=model_name,
        train_data=train_data,
        val_data=test_data,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        n_hid=n_hid,
        n_filt=n_filt,
        drop_prob=drop_prob,
        save_params_frame=params_frame,
        verbose=args.verbose,
        all_verbose=args.all_verbose,
        early_stopping=early_stop,
        patience=patience,
        min_delta=min_delta)
    
    if args.verbose or args.all_verbose:
        tqdm.write(f"[INFO] Training finished\n")

    params_frame.to_parquet(args.params_path)

    if args.verbose or args.all_verbose:
        tqdm.write(f"[INFO] Parameters saved\n")

    if not args.no_plot:
        # === Training curve === #
        plot_training_curves(
            train_losses=history['train_losses'],
            val_losses=history['val_losses'],
            train_accs=history.get('train_accs', None),
            val_accs=history.get('val_accs', None),
            model_name=f'{model_name}_{ID}',
            verbose=args.all_verbose,
            ID=ID
        )
        
        # === Confusion matrix === #
        cf = history['cf_val'] if history['cf_val'] is not None else history['cf_train']
        classes = ['Nucleus','Cytoplasm','Extracellular','Mitochondrion','Cell membrane','ER',
           'Chloroplast','Golgi apparatus','Lysosome','Vacuole']
        plot_confusion_matrix(
                cf_matrix=cf,
                classes=classes,
                title=f"Matrice de confusion - {model_name}",
                model_name=f'{model_name}_{ID}',
                verbose=args.all_verbose,
                ID=ID
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
    parser.add_argument('--saving', action='store_true',help='Save the model in ./models/')
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

    val_data = None
    if CONFIG["dataset"]["val_path"] is not None:
        val_npz = np.load(CONFIG["dataset"]["val_path"])
        X_val = val_npz['X_val']
        y_val = val_npz['y_val']
        mask_val = val_npz['mask_val']
        val_data = (X_val, y_val, mask_val)

    early_stop = CONFIG['Early_stopping']['perform']
    patience = int(CONFIG['Early_stopping']['patience'])
    min_delta = float(CONFIG['Early_stopping']['min_delta'])
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

        if args.saving:
            with tqdm(total=len(hyper_grid), desc="[MULTIMODEL] Run for model n°", ncols=90) as pbar:
                for run_idx, (model_name, batch_size, epochs, lr, n_hid, n_filt, drop_prob) in enumerate(hyper_grid):

                    params = main(ID,model_name,batch_size,epochs,
                        lr,n_hid,n_filt,drop_prob,
                        train_data,val_data,
                        early_stop=early_stop, patience=patience, min_delta=min_delta
                    )
                    file_path = os.path.join(save_path, f"{ID}_{CONFIG['model']['name']}.npz")
                    save_model(params, file_path)
                    if args.verbose:
                        tqdm.write("[INFO] Model saved")

                    ID = str(int(ID) + 1).zfill(len(ID))
                    pbar.update(1)

        else:
            with tqdm(total=len(hyper_grid), desc="[MULTIMODEL] Run for model - ", ncols=90) as pbar:
                for run_idx, (model_name, batch_size, epochs, lr, n_hid, n_filt, drop_prob) in enumerate(hyper_grid):

                    _ = main(ID,model_name,batch_size,epochs,
                        lr,n_hid,n_filt,drop_prob,
                        train_data,val_data,
                        early_stop=early_stop, patience=patience, min_delta=min_delta
                    )

                    ID = str(int(ID) + 1).zfill(len(ID))
                    pbar.update(1)

    # === Single model === #

    else:
        if args.saving:
            params = main(ID, CONFIG["model"]["name"], CONFIG["training"]["batch_size"], 
                CONFIG["training"]["epochs"], CONFIG["training"]["learning_rate"], 
                CONFIG["model"]["n_hid"], CONFIG["model"]["n_filt"], CONFIG["model"]["drop_prob"],
                train_data, val_data, early_stop=early_stop, patience=patience, min_delta=min_delta)
            
            file_path = os.path.join(save_path, f"{ID}_{CONFIG['model']['name']}.npz")
            save_model(params, file_path)
            if args.verbose:
                tqdm.write("[INFO] Model saved")
        else:
            _ = main(ID, CONFIG["model"]["name"], CONFIG["training"]["batch_size"], 
                CONFIG["training"]["epochs"], CONFIG["training"]["learning_rate"], 
                CONFIG["model"]["n_hid"], CONFIG["model"]["n_filt"], CONFIG["model"]["drop_prob"],
                train_data, val_data , early_stop=early_stop, patience=patience, min_delta=min_delta)

