import numpy as np


def split_continuous(X, y, test_size=0.25, random_state=42, mask=True):
    rng=np.random.RandomState(random_state)
    val_size=int(test_size*X.shape[0])
    max_n_split=X.shape[0]-int(test_size*X.shape[0])
    ind=rng.randint(max_n_split-1)
    mask_val=np.arange(ind, ind+val_size+1)
    mask_train= np.array([n for n in range(X.shape[0]) if n not in mask_val])
    X_train=X[mask_train]
    y_train=y[mask_train]
    X_val=X[mask_val]
    y_val=y[mask_val]

    if mask:
        return X_train, X_val, y_train, y_val
    else:
        return X_train, X_val, y_train, y_val, mask_val

def split_continuous_ind(X, y, test_size=0.25, random_state=42):
    rng=np.random.RandomState(random_state)
    val_size=int(test_size*X.shape[0])
    max_n_split=X.shape[0]-int(test_size*X.shape[0])
    ind=rng.randint(max_n_split-1)
    mask_val=np.arange(ind, ind+val_size+1)
    mask_train= np.array([n for n in range(X.shape[0]) if n not in mask_val])



    return mask_train, mask_val
