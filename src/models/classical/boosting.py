# All boosting models

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import pickle, json, time
from pathlib import Path


def train_lightgbm(X_tr, y_tr, X_vl, y_vl, X_ts, y_ts,
                   n_estimators=2000, lr=0.03, num_leaves=255,
                   save_path=None, feature_names=None):
    import lightgbm as lgb

    params = {
        'objective':       'multiclass',
        'num_class':       len(set(y_tr)),
        'metric':          'multi_logloss',
        'learning_rate':   lr,
        'num_leaves':      num_leaves,
        'max_depth':       -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq':    5,
        'lambda_l2':       0.1,
        'verbose':         -1,
        'seed':            42,
        'num_threads':     -1,
    }

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
    dval   = lgb.Dataset(X_vl, label=y_vl, reference=dtrain)

    t0 = time.time()
    model = lgb.train(
        params, dtrain, num_boost_round=n_estimators,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)]
    )
    elapsed = time.time() - t0

    val_preds  = np.argmax(model.predict(X_vl), axis=1)
    test_preds = np.argmax(model.predict(X_ts), axis=1)
    val_acc  = accuracy_score(y_vl, val_preds)
    test_acc = accuracy_score(y_ts, test_preds)
    val_f1   = f1_score(y_vl, val_preds, average='macro')
    test_f1  = f1_score(y_ts, test_preds, average='macro')

    print(f"  LightGBM  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}  [{elapsed:.1f}s]")

    if save_path:
        model.save_model(save_path)

    return model, {'val_acc': val_acc, 'test_acc': test_acc, 'val_f1': val_f1, 'test_f1': test_f1}


def train_xgboost(X_tr, y_tr, X_vl, y_vl, X_ts, y_ts, save_path=None):
    import xgboost as xgb
    import torch

    use_gpu = torch.cuda.is_available()
    model = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.02, max_depth=8,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss',
        tree_method='gpu_hist' if use_gpu else 'hist',
        random_state=42, verbosity=0, use_label_encoder=False
    )
    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
              early_stopping_rounds=50, verbose=200)

    val_acc  = accuracy_score(y_vl, model.predict(X_vl))
    test_acc = accuracy_score(y_ts, model.predict(X_ts))
    val_f1   = f1_score(y_vl, model.predict(X_vl), average='macro')
    test_f1  = f1_score(y_ts, model.predict(X_ts), average='macro')
    print(f"  XGBoost  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")

    if save_path:
        with open(save_path, 'wb') as f: pickle.dump(model, f)

    return model, {'val_acc': val_acc, 'test_acc': test_acc, 'val_f1': val_f1, 'test_f1': test_f1}


def train_catboost(X_tr, y_tr, X_vl, y_vl, X_ts, y_ts, save_path=None):
    from catboost import CatBoostClassifier
    import torch

    model = CatBoostClassifier(
        iterations=2000, learning_rate=0.03, depth=8,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        task_type='GPU' if torch.cuda.is_available() else 'CPU',
        random_seed=42, verbose=200
    )
    model.fit(X_tr, y_tr, eval_set=(X_vl, y_vl),
              early_stopping_rounds=50, use_best_model=True)

    val_acc  = accuracy_score(y_vl, model.predict(X_vl).ravel())
    test_acc = accuracy_score(y_ts, model.predict(X_ts).ravel())
    val_f1   = f1_score(y_vl, model.predict(X_vl).ravel(), average='macro')
    test_f1  = f1_score(y_ts, model.predict(X_ts).ravel(), average='macro')
    print(f"  CatBoost  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")

    if save_path:
        model.save_model(save_path)

    return model, {'val_acc': val_acc, 'test_acc': test_acc, 'val_f1': val_f1, 'test_f1': test_f1}