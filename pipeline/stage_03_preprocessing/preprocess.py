import os
import sys
import logging
import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def setup_logger(log_dir: str, log_file: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger("preprocessing")
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
    
#  load data
def load_data(source_path: str,logger:logging.Logger):
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Data not found: {source_path}")
    
    df=pd.read_csv(source_path)
    logger.info(f"Loaded data ")
    return df

#  now cleaning invalid values from stage 2 
def clean_invalid_values(df:pd.DataFrame,invalid_cfg:dict,logger:logging.Logger):
    original_len=len(df)
    for col , rules in invalid_cfg.items():
        invalid_vals=rules["invalid"]
        strategy=rules["strategy"]

        # rows with invalid values 
        mask=df[col].isin(invalid_vals)
        affected=int(mask.sum())

        if affected==0:
            logger.info(f"{col} - no invalid values found")
            continue

        if strategy == "drop":
            df = df[~mask].reset_index(drop=True)
            logger.info(
                f"  '{col}' — dropped {affected} rows with values {invalid_vals}"
            )
        elif strategy == "replace":
            valid_mode = df.loc[~mask, col].mode()[0]
            df.loc[mask, col] = valid_mode
            logger.info(
                f"  '{col}' — replaced {affected} rows "
                f"(values {invalid_vals}) with mode={valid_mode}"
            )
        else:
            logger.warning(f"{col} - unknown strategy {strategy} , skipped")
    
    rows_removed = original_len - len(df)
    logger.info(
        f"  Cleaning done — removed {rows_removed} rows, "
        f"{len(df)} rows remaining"
    )
    return df


# SPLIT IN FEATURES AND TARGET X,y

def split_features_target(df:pd.DataFrame,target_col:str,logger:logging.Logger):
    X=df.drop(columns=[target_col])
    y=df[target_col]
    logger.info(f"Features shape : {X.shape}")
    logger.info(f"Target shape   : {y.shape}")
    logger.info(f"Target balance : {y.value_counts().to_dict()}")
    return X,y

# split into train val test 1. seperate test set

def split_data(
        X:pd.DataFrame,
        y:pd.Series,
        split_cfg:dict,
        logger:logging.Logger):
    random_state=split_cfg["random_state"]
    val_size=split_cfg["val_size"]
    test_size=split_cfg["test_size"]

    X_temp ,X_test , y_temp , y_test =train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=y)

    # 2. seperate val from remaining and recalculate val size related to remaining data 
    
    val_size_adjusted=val_size/(1-test_size)

    X_train , X_val , y_train , y_val = train_test_split(X_temp,y_temp,test_size=val_size_adjusted,random_state=random_state,stratify=y_temp)


    logger.info(f"Train set : {X_train.shape[0]} rows ({split_cfg['train_size']*100:.0f}%)")
    logger.info(f"Val set   : {X_val.shape[0]} rows ({val_size*100:.0f}%)")
    logger.info(f"Test set  : {X_test.shape[0]} rows ({test_size*100:.0f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_preprocessor(numeric_cols:list,categorical_cols:list,binary_cols:list,logger:logging.Logger)->ColumnTransformer:
    logger.info("-"*35)
    logger.info("Build sklearn ColumnTransformer")

    numeric_pipeline=Pipeline([
        ("impputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])

    categorical_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))
    ])

    binary_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("encoder",OneHotEncoder(handle_unknown="ignore", sparse_output=False)
         )
    ])

    preprocessor=ColumnTransformer([
        ("numeric",numeric_pipeline,numeric_cols),
        ("categorical",categorical_pipeline,categorical_cols),
        ("binary",binary_pipeline,binary_cols)
    ])
    logger.info(f"  Numeric columns     : {numeric_cols}")
    logger.info(f"  Categorical columns : {categorical_cols}")
    logger.info(f"  Binary columns      : {binary_cols}")

    return preprocessor

# fit and transform
def fit_transform(preprocessor:ColumnTransformer,X_train:pd.DataFrame,X_val:pd.DataFrame,X_test:pd.DataFrame,logger:logging.Logger):
    logger.info("-" * 45)
    logger.info("FIT + TRANSFORM — fit on train only (no leakage)")
    X_train_p=preprocessor.fit_transform(X_train)
    X_val_p=preprocessor.transform(X_val)
    X_test_p=preprocessor.transform(X_test)

    logger.info(f"  Processed train shape : {X_train_p.shape}")
    logger.info(f"  Processed val shape   : {X_val_p.shape}")
    logger.info(f"  Processed test shape  : {X_test_p.shape}")
 
    return X_train_p, X_val_p, X_test_p

#  saving artifacts
def save_artifacts(
    X_train: np.ndarray, y_train: pd.Series,
    X_val:   np.ndarray, y_val:   pd.Series,
    X_test:  np.ndarray, y_test:  pd.Series,
    preprocessor: ColumnTransformer,
    artifact_dir: str,
    logger: logging.Logger,):


    logger.info("-" * 45)
    logger.info("STEP 6 — Save artifacts")

    Path(artifact_dir).mkdir(parents=True,exist_ok=True)

    try:
        feature_names=preprocessor.get_feature_names_out()
    except Exception:
        feature_names = None
    
    def _to_df(X:np.ndarray,y:pd.Series):
        df=pd.DataFrame(X,columns=feature_names)
        df["HeartDisease"]=y.values
        return df
    
    #  save train
    train_path=os.path.join(artifact_dir,"train.csv")
    _to_df(X_train,y_train).to_csv(train_path,index=False)
    logger.info(f"  Saved train.csv          → {train_path}")

    # save val
    val_path=os.path.join(artifact_dir,"val.csv")
    _to_df(X_val,y_val).to_csv(val_path,index=False)
    logger.info(f"  Saved val.csv            → {val_path}")

    # save test
    test_path = os.path.join(artifact_dir, "test.csv")
    _to_df(X_test, y_test).to_csv(test_path, index=False)
    logger.info(f"  Saved test.csv           → {test_path}")

    # save fitted preprocessor - used by inference API  at serving time
    preprocessor_path = os.path.join(artifact_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"  Saved preprocessor.joblib → {preprocessor_path}")


def run():
    global_config = load_config("config/config.yaml")
    stage_config  = load_config(
        "pipeline/stage_03_preprocessing/config.yaml"
    )
 
    logger = setup_logger(
        global_config["logs"]["log_dir"],
        global_config["logs"]["log_file"],
    )
 
    logger.info("=" * 50)
    logger.info("STAGE 03 — Preprocessing started")
    logger.info("=" * 50)
 
    try:
        artifact_dir = global_config["data_ingestion"]["artifact_dir"]
        file_name = global_config["data_ingestion"]["file_name"]

        data_path = os.path.join(
            artifact_dir,
            file_name)
        df = load_data(data_path, logger)

        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info(f"Removed {before - len(df)} duplicate rows")
        
        bp_zeros = (df["RestingBP"] == 0).sum()

        df["RestingBP"] = df["RestingBP"].replace(0,np.nan)

        logger.info(f"Converted {bp_zeros} RestingBP=0 values to NaN")
 
        df = clean_invalid_values(
            df,
            stage_config["invalid_values"],
            logger,
        )
 
        X, y = split_features_target(
            df,
            stage_config["target_column"],
            logger,
        )
 
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y,
            stage_config["split"],
            logger,
        )
 
        preprocessor = build_preprocessor(
            numeric_cols     = stage_config["numeric_columns"],
            categorical_cols = stage_config["categorical_columns"],
            binary_cols      = stage_config["binary_columns"],
            logger           = logger,
        )


        X_train_p, X_val_p, X_test_p = fit_transform(
            preprocessor,
            X_train, X_val, X_test,
            logger,
        )
 
        # ── STEP 6 — save everything ──────────────────────────────────────
        save_artifacts(
            X_train_p, y_train,
            X_val_p,   y_val,
            X_test_p,  y_test,
            preprocessor,
            artifact_dir = "pipeline/stage_03_preprocessing/artifacts",
            logger       = logger,
        )
 
        logger.info("=" * 50)
        logger.info("STAGE 03 — Preprocessing completed successfully")
        logger.info("=" * 50)
        return True
 
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return False
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return False
        
 
 
if __name__ == "__main__":
    run()
