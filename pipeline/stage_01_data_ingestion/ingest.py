import os
import sys
import logging
import pandas as pd 
import yaml
from pathlib import Path



# 1. Logger -> logs message to both terminal and log file .for tracking
def setup_logger(log_dir:str,log_file:str)-> logging.Logger:
    Path(log_dir).mkdir(parents=True,exist_ok=True)
    log_path=os.path.join(log_dir,log_file)

    #create logger with name 
    logger=logging.getLogger("Data Ingestion")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler=logging.FileHandler(log_path)
        terminal_handler=logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        terminal_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(terminal_handler)
    
    return logger

#  2 Load config

def load_config(config_path:str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found:{config_path}")
    
    with open(config_path,"r") as f:
        config=yaml.safe_load(f)
    
    return config

# 3 read csv
def read_and_save(source_path:str,artifact_dir:str,file_name:str,logger:logging.Logger)-> pd.DataFrame:
    
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Original file not found at: {source_path}")
    
    logger.info(f"Reading original file:{source_path}")
    
    df = pd.read_csv(source_path)
    logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    Path(artifact_dir).mkdir(parents=True,exist_ok=True)
    artifact_path=os.path.join(artifact_dir,file_name)
    df.to_csv(artifact_path,index=False)
    logger.info(f"Saved pipeline copy to {artifact_path}")

    return df

# 4 Sanity CHECKS = basic checks to make sure data looks right befor passing it to validation stage
def sanity_check(df:pd.DataFrame,stage_config:dict,logger:logging.Logger):
    logging.info("Running sanity check")
    errors=[]

    if df.empty:
        errors.append("Dataframe is empty ")
    min_rows=stage_config.get("expected_rows_min",100)
    if len(df)<min_rows:
        errors.append(f"few rows {len(df)}, need at least {min_rows}")

    exp_col=stage_config.get("expected_columns",14)
    if df.shape[1] != exp_col:
        errors.append(f"Wrong columns — got {df.shape[1]}, expected {exp_col}")

    target_col = stage_config.get("target_column", "target")
    if target_col not in df.columns:
        errors.append(f"Target column '{target_col}' not found")
    
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"{duplicates} duplicate rows found — worth checking")
    
    if errors:
        for err in errors:
            logger.error(f"FAILED: {err}")
        raise ValueError("Sanity checks failed — fix the errors above")
    
    logger.info("All checks passed")
    logger.info(f"Shape      : {df.shape}")
    logger.info(f"Columns    : {list(df.columns)}")
    logger.info(f"Missing    : {df.isnull().sum().sum()} total null values")
    logger.info(f"Duplicates : {duplicates}")

# data summary

def log_summary(df: pd.DataFrame, logger: logging.Logger) -> None:

    logger.info("--- Data Summary ---")
    logger.info(f"Column types:\n{df.dtypes.to_string()}")

    # how many 0s and 1s in the target column
    logger.info(f"Target counts:\n{df['target'].value_counts().to_string()}")


def run():
    global_config=load_config("config/config.yaml")
    stage_config=load_config("pipeline/stage_01_data_ingestion/config.yaml")

    logger=setup_logger(
        global_config["logs"]["log_dir"],
        global_config["logs"]["log_file"]
    )
    logger.info("=" * 25)
    logger.info("Stage 01 -Data ingestion started")
    logger.info("=" * 25)

    try:
        cfg=global_config["data_ingestion"]
        df= read_and_save(cfg["source_path"],cfg["artifact_dir"],cfg["file_name"],logger)
        sanity_check(df,stage_config,logger)
        log_summary(df,logger)
        logger.info("=" * 25)
        logger.info("STAGE 01 — Completed successfully")
        logger.info("=" * 24)
        return True

    except FileNotFoundError as e:
        logger.error(f"File not found:{e}")
        return False
    except ValueError as e:
        logger.error(f"data error {e}")
        return False
    except Exception as e:
        logger.error(f"unexpected error : {e}")
        return False

    
if __name__=="__main__":
    run()