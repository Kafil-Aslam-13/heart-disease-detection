import os
import sys
import json
import logging
import pandas as pd 
import yaml
from pathlib import Path

def setup_logger(log_dir:str,log_file:str)-> logging.Logger:
    Path(log_dir).mkdir(parents=True,exist_ok=True)
    log_path=os.path.join(log_dir,log_file)

    #create logger with name 
    logger=logging.getLogger("Data Validation")
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



def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def check_columns(df:pd.DataFrame,schema:dict,logger:logging.Logger):
    errors=[]
    expected=list(schema["columns"].keys())
    actual=list(df.columns)

    logger.info(f"Expected columns:{expected}")
    logger.info(f"Actual columns{actual}")

    for col in expected:
        if col in actual:
            logger.info(f"{col} is present ")
        else:
            logger.error(f"column: {col} is MISSING from the data")
            errors.append(f"MISSING COLUMN {col}")
    
    if not errors:
        logger.info("COLUMN CHECKED PASSED - all columns present")
    
    return errors

def check_missing(df:pd.DataFrame,logger:logging.Logger):
    errors=[]
    null_counts=df.isnull().sum()
    cols_with_nulls=null_counts[null_counts>0]

    if len(cols_with_nulls) >0:
        logger.warning(f"Null values Found:\n{cols_with_nulls.to_string()}")
    else:
        logger.info("Missing value check passed - ")
    
    return errors

#  check data types and value ranges
def check_values(df:pd.DataFrame,schema:dict,logger:logging.Logger):
    errors=[]
    for col, rules in schema["columns"].items():

        if col not in df.columns:
            continue

        col_values=df[col].dropna()

        if "allowed_values" in rules:
            allowed  = rules["allowed_values"]
            actual_vals = df[col].dropna().unique().tolist()
            invalid = [v for v in actual_vals if v not in allowed]

            if invalid:
                logger.warning(f"Column '{col}' has invalid values: {invalid} "
                               f"— will be handled in preprocessing")
            else:
                logger.info(f"Column '{col}' values OK — {actual_vals}")
            
        if "min" in rules:
            below_min=col_values[col_values < rules["min"]]
            if len(below_min)>0:
                errors.append(f"Column {col} has {len(below_min)} values below min {rules['min']}")
                logger.error(f"Column {col} below min ({rules['min']}): {below_min.values}")
            else:
                logger.info(f"column {col} min check ok")
        
        if "max" in rules:
            above_max=col_values[col_values > rules["max"]]
            if len(above_max)>0:
                errors.append(f"Column {col} has {len(above_max)} values above max {rules['max']}")
                logger.error(f"Column {col} above max ({rules['max']}): {above_max.values}")
            else:
                logger.info(f"column {col} max check ok")
    
    return errors

# checking if dataset is balanced or not
def check_class_balance(df: pd.DataFrame, logger: logging.Logger) -> list:
    errors = []
    counts = df["target"].value_counts(normalize=True) * 100

    logger.info(f"Target class balance:")
    for label, pct in counts.items():
        logger.info(f"  Class {label} → {pct:.1f}%")

    # warn if any class is below 20%
    for label, pct in counts.items():
        if pct < 20:
            logger.warning(f"Class {label} is only {pct:.1f}% — consider SMOTE in preprocessing")

    return errors

#  now for this particular dataset i will save this report 
def save_report(df:pd.DataFrame,errors:list,artifact_dir:str,logger:logging.Logger):
    Path(artifact_dir).mkdir(parents=True,exist_ok=True)
    report={
        "rows":int(df.shape[0]),
        "columns":int(df.shape[1]),
        "null_counts":df.isnull().sum().to_dict(),
        "errors_found":len(errors),
        "errors":errors,
        "status":"FAILED" if errors else "PASSED"
    }
    report_path=os.path.join(artifact_dir,"validation_report.json")
    with open(report_path,"w") as f:
        json.dump(report,f,indent=4)
    logger.info(f"Validation report saved to :{report_path}")


def  run():
    global_config=load_config("config/config.yaml")
    schema=load_config("pipeline/stage_02_data_validation/schema.yaml")

    logger = setup_logger(
        global_config["logs"]["log_dir"],
        global_config["logs"]["log_file"]
    )

    logger.info("=" * 50)
    logger.info("STAGE 02 — Data Validation started")
    logger.info("=" * 50)

    try:
        # read artifact from stage 1
        data_path=global_config["data_ingestion"]["source_path"]
        df=pd.read_csv(data_path)
        logger.info(f"loaded data from stage 1 : {data_path}")

        all_errors=[]

        all_errors+=check_columns(df,schema,logger)
        all_errors+=check_missing(df,logger)
        all_errors+=check_values(df,schema,logger)
        all_errors+=check_class_balance(df,logger)

        # save report
        artifact_dir="pipeline/stage_02_data_validation/artifacts"
        save_report(df,all_errors,artifact_dir,logger)

        if all_errors:
            logger.error(f"Validation Failed with {len(all_errors)} errors")
            # sys.exit(1)
            return False
        else:
            logger.info("=" * 25)
            logger.info("STAGE 02 — Validation passed successfully")
            logger.info("=" * 25)
            return True
    except FileNotFoundError as e:
        logger.error(f"file not found : {e}")
        return False
    except Exception as e:
        logger.error(f"unexpected error {e}")
        return False

if __name__=="__main__":
    run()


        

