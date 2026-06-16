import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import yaml
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score,accuracy_score,precision_score,recall_score,f1_score)
from xgboost import XGBClassifier

def setup_logger(log_dir: str, log_file: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger("model_training")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler   = logging.FileHandler(log_path)
        stream_handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    

def load_data(artifact_dir:str,target_col,logger:logging.Logger):
    train_path=os.path.join(artifact_dir,"train.csv")
    val_path=os.path.join(artifact_dir,"val.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found at: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"val.csv not found at: {val_path}")
    
    train_df=pd.read_csv(train_path)
    val_df=pd.read_csv(val_path)


    X_train = train_df.drop(target_col, axis=1).values
    y_train = train_df[target_col].values

    X_val = val_df.drop(target_col, axis=1).values
    y_val = val_df[target_col].values

    # X_val=val_df.iloc[:,:-1].values
    # y_val=val_df.iloc[:,-1].values

    logger.info(f"Train — X: {X_train.shape}, y: {y_train.shape}")
    logger.info(f"Val   — X: {X_val.shape},   y: {y_val.shape}")
    logger.info(
    f"Train Target Distribution: "
    f"{pd.Series(y_train).value_counts().to_dict()}")

    logger.info(
    f"Val Target Distribution: "
    f"{pd.Series(y_val).value_counts().to_dict()}")

    return X_train , y_train,X_val,y_val

# Building all models for training data 

def build_models (params:dict,logger:logging.Logger):
    

    models={
        "logistic_regression":LogisticRegression(
            max_iter=params["logistic_regression"]["max_iter"],
            C=params["logistic_regression"]["C"],
            random_state=params["logistic_regression"]["random_state"]),
        
        "random_forest":RandomForestClassifier(
            n_estimators=params["random_forest"]["n_estimators"],
            max_depth=params['random_forest']["max_depth"],
            min_samples_split= params["random_forest"]["min_samples_split"],
            min_samples_leaf = params["random_forest"]["min_samples_leaf"],
            random_state     = params["random_forest"]["random_state"]
        ),
        "xgboost":XGBClassifier(
            n_estimators      = params["xgboost"]["n_estimators"],
            max_depth         = params["xgboost"]["max_depth"],
            learning_rate     = params["xgboost"]["learning_rate"],
            subsample         = params["xgboost"]["subsample"],
            colsample_bytree  = params["xgboost"]["colsample_bytree"],
            scale_pos_weight  = params["xgboost"]["scale_pos_weight"],
            random_state      = params["xgboost"]["random_state"],
            eval_metric       = "logloss",
            verbosity         = 0
        ),

            
    }

    logger.info(f"Built {len(models)} models: {list(models.keys())}")
    return models

# training and evaluating on xtrain and x val respectivvely

def train_and_evaluate(
        models:dict,
        X_train:np.ndarray,
        y_train:np.ndarray,
        X_val:np.ndarray,
        y_val:np.ndarray,
        logger:logging.Logger):
    results={}

    for name , model in models.items():
        logger.info(f"training {name}...")

        model.fit(X_train,y_train)

        y_pred=model.predict(X_val)
        y_pred_prob=model.predict_proba(X_val)[:,1]

        # calculating all metrics on validation set 
        metrics={
            "accuracy"  : round(accuracy_score(y_val, y_pred),           4),
            "roc_auc"   : round(roc_auc_score(y_val, y_pred_prob),       4),
            "precision" : round(precision_score(y_val, y_pred),          4),
            "recall"    : round(recall_score(y_val, y_pred),             4),
            "f1"        : round(f1_score(y_val, y_pred),                 4),
        }

        results[name]={
            "model":model,
            "metrics":metrics
        }

        # log results clearly
        logger.info(f"  Results for {name}:")
        logger.info(f"  Accuracy  : {metrics['accuracy']}")
        logger.info(f"  AUC-ROC   : {metrics['roc_auc']}")
        logger.info(f"  Precision : {metrics['precision']}")
        logger.info(f"  Recall    : {metrics['recall']}")
        logger.info(f"  F1        : {metrics['f1']}")
        logger.info("-" * 40)
    return results

#  pick the best model

def pick_best_model(results:dict,metric:str,logger:logging.Logger):
    best_name=None
    best_score=-1

    for name, result in results.items():
        score=result["metrics"][metric]
        if score > best_score:
            best_score=score
            best_name=name
    logger.info(f"Best model : {best_name} ({metric} = {best_score})")
    return best_name

#  save  model
def save_artifacts(results:dict,best_name:str,X_train,y_train,X_val,y_val,artifact_dir:str,logger:logging.Logger):
    Path(artifact_dir).mkdir(parents=True,exist_ok=True)
    # save each model
    for name , result in results.items():
        model_path=os.path.join(artifact_dir,f"{name}.joblib")
        joblib.dump(result["model"],model_path)
        logger.info(f"Saved {name} → {model_path}")

    #  best model save here 
    best_model=results[best_name]["model"]
    logger.info("Retraining best model on train + validation date")
    X_train_full = np.vstack([
        X_train,X_val
    ])
    y_train_full=np.concatenate([
        y_train,y_val
    ])
    best_model.fit(X_train_full,y_train_full)
    best_model_path=os.path.join(artifact_dir,"best_model.joblib")

    joblib.dump(best_model,best_model_path)
    logger.info(f"Saved best model ({best_name}) → {best_model_path}")

    # save training results as json — for reference
    results_to_save = {
        name: result["metrics"]
        for name, result in results.items()
    }
    results_to_save["best_model"] = best_name

    results_path = os.path.join(artifact_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results_to_save, f, indent=4)

    logger.info(f"Saved training results → {results_path}")

def run():
    global_config = load_config("config/config.yaml")
    params        = load_config(
        "pipeline/stage_04_model_training/params.yaml"
    )

    logger = setup_logger(
        global_config["logs"]["log_dir"],
        global_config["logs"]["log_file"]
    )

    logger.info("=" * 50)
    logger.info("STAGE 04 — Model Training started")
    logger.info("=" * 50)

    try:
        # step 3 — load data from stage 03
        artifact_dir  = "pipeline/stage_03_preprocessing/artifacts"
        preprocess_cfg = load_config("pipeline/stage_03_preprocessing/config.yaml")
        target_col=preprocess_cfg["target_column"]
        X_train, y_train, X_val, y_val = load_data(artifact_dir,target_col, logger)

        # step 4 — build all models
        models = build_models(params, logger)

        # step 5 — train and evaluate all models
        results = train_and_evaluate(
            models,
            X_train, y_train,
            X_val,   y_val,
            logger
        )

        # step 6 — pick best model
        best_metric = params.get("best_model_metric", "roc_auc")
        best_name   = pick_best_model(results, best_metric, logger)

        # step 7 — save all models + best
        save_artifacts(
            results,
            best_name,
            X_train,
            y_train,
            X_val,
            y_val,
            artifact_dir = "pipeline/stage_04_model_training/artifacts",
            logger       = logger
        )

        logger.info("=" * 50)
        logger.info("STAGE 04 — Model Training completed successfully")
        logger.info("=" * 50)
        return True

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False
        

if __name__ == "__main__":
    run()
