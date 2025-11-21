import numpy as np
from models import FFN, CNN_LSTM, CNN_LSTM_Att, CNN

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
