import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import yaml
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path


from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)

def setup_logger(log_dir: str, log_file: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger("evaluation")
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
    


def load_test_data_and_model(
        preprocessing_dir:str,
        training_dir:str,
        logger: logging.Logger
):
    test_path=os.path.join(preprocessing_dir,"test.csv")
    model_path=os.path.join(training_dir,"best_model.joblib")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.csv not found: {test_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"best_model.joblib not found: {model_path}")
    
    test_df = pd.read_csv(test_path)
    X_test  = test_df.iloc[:, :-1].values
    y_test  = test_df.iloc[:, -1].values

    model = joblib.load(model_path)

    logger.info(f"Test set shape : {X_test.shape}")
    logger.info(f"Model loaded   : {model_path}")

    return X_test , y_test,model


def calculate_metrics(model:object,X_test:np.ndarray,y_test:np.ndarray,logger:logging.Logger):
    y_pred=model.predict(X_test)

    y_pred_prob=model.predict_proba(X_test)[:,1]

    metrics = {
        "accuracy"        : round(accuracy_score(y_test, y_pred),       4),
        "roc_auc"         : round(roc_auc_score(y_test, y_pred_prob),   4),
        "precision"       : round(precision_score(y_test, y_pred),      4),
        "recall"          : round(recall_score(y_test, y_pred),         4),
        "f1"              : round(f1_score(y_test, y_pred),             4),
    }


    logger.info("Final Test Set Metrics:")
    logger.info(f"  Accuracy  : {metrics['accuracy']}")
    logger.info(f"  AUC-ROC   : {metrics['roc_auc']}")
    logger.info(f"  Precision : {metrics['precision']}")
    logger.info(f"  Recall    : {metrics['recall']}")
    logger.info(f"  F1 Score  : {metrics['f1']}")


    # report 
    report=classification_report(y_test,y_pred)
    logger.info(f"Classification Report:\n{report}")

    if metrics["recall"] < 0.80:
        logger.warning("Recall below 0.80 . model not predicting too many sick patients . try to adjust threshold")

    return metrics , y_pred,y_pred_prob

def plot_confusion_matrix(
        y_test:np.ndarray,
        y_pred:np.ndarray,
        artifact_dir:str,
        logger:logging.Logger):
    cm=confusion_matrix(y_test,y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot      = True,
        fmt        = "d",
        cmap       = "Blues",
        xticklabels= ["No Disease (0)", "Disease (1)"],
        yticklabels= ["No Disease (0)", "Disease (1)"]
    )
    plt.title("Confusion Matrix — Test Set")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    save_path = os.path.join(artifact_dir, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved confusion matrix → {save_path}")

    tn, fp, fn, tp = cm.ravel()
    logger.info(f"True Negatives  (correctly said no disease) : {tn}")
    logger.info(f"False Positives (said disease, actually not): {fp}")
    logger.info(f"False Negatives (missed sick patients)      : {fn} ")
    logger.info(f"True Positives  (correctly caught disease)  : {tp}")

def plot_roc_curve(
        y_test:np.ndarray,
        y_pred_prob:np.ndarray,
        artifact_dir:str,
        logger:logging.Logger):
    fpr,tpr, _=roc_curve(y_test,y_pred_prob)
    auc_score=roc_auc_score(y_test,y_pred_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="blue", label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random baseline")
    plt.title("ROC Curve — Test Set")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(artifact_dir, "roc_curve.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved ROC curve → {save_path}")


def plot_shap(
    model:        object,
    X_test:       np.ndarray,
    artifact_dir: str,
    logger:       logging.Logger
):
    try:

        logger.info("Generating SHAP explanation...")
        expaliner=shap.TreeExplainer(model)
        shap_values=expaliner.shap_values(X_test)
        plt.figure()
        shap.summary_plot(
            shap_values,
            X_test,
            show = False
        )
        plt.tight_layout()

        save_path = os.path.join(artifact_dir, "shap_summary.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved SHAP summary → {save_path}")
    except Exception as e:
        logger.warning(f"SHAP SKIPPED: {e}")

def save_report(metrics:dict,
                artifact_dir:str,
                logger:logging.Logger):
    report = {
        "test_metrics"  : metrics,
        "status"        : "PASSED" if metrics["roc_auc"] >= 0.80 else "REVIEW",
        "note"          : "Recall is most important metric for medical use case"
    }

    report_path = os.path.join(artifact_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Saved evaluation report → {report_path}")


def run():
    global_config = load_config("config/config.yaml")

    logger = setup_logger(
        global_config["logs"]["log_dir"],
        global_config["logs"]["log_file"]
    )

    logger.info("=" * 50)
    logger.info("STAGE 05 — Evaluation started")
    logger.info("=" * 50)

    try:
        preprocessing_dir = "pipeline/stage_03_preprocessing/artifacts"
        training_dir      = "pipeline/stage_04_model_training/artifacts"
        artifact_dir      = "pipeline/stage_05_evaluation/artifacts"

        Path(artifact_dir).mkdir(parents=True, exist_ok=True)

        X_test, y_test, model = load_test_data_and_model(
            preprocessing_dir,
            training_dir,
            logger
        )

        metrics, y_pred, y_pred_prob = calculate_metrics(
            model, X_test, y_test, logger
        )


        plot_confusion_matrix(y_test, y_pred, artifact_dir, logger)

        plot_roc_curve(y_test, y_pred_prob, artifact_dir, logger)

        plot_shap(model, X_test, artifact_dir, logger)

        save_report(metrics, artifact_dir, logger)

        logger.info("=" * 50)
        logger.info("STAGE 05 — Evaluation completed successfully")
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