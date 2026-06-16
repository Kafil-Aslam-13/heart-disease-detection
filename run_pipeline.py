import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pipeline.stage_01_data_ingestion.ingest    import run as ingest
from pipeline.stage_02_data_validation.validate import run as validate
from pipeline.stage_03_preprocessing.preprocess import run as preprocess
from pipeline.stage_04_model_training.train     import run as train
from pipeline.stage_05_evaluation.evaluate      import run as evaluate


def run_full_pipeline():
    print("=" * 60)
    print("HEART DISEASE DETECTION — FULL PIPELINE")
    print("=" * 60)

    print("\n[1/5] Running Data Ingestion...")
    if ingest() is False:
        print("Pipeline stopped — Data Ingestion failed")
        sys.exit(1)

    print("\n[2/5] Running Data Validation...")
    if validate() is False:
        print("Pipeline stopped — Data Validation failed")
        sys.exit(1)

    print("\n[3/5] Running Preprocessing...")
    if preprocess() is False:
        print("Pipeline stopped — Preprocessing failed")
        sys.exit(1)

    print("\n[4/5] Running Model Training...")
    if train() is False:
        print("Pipeline stopped — Model Training failed")
        sys.exit(1)

    print("\n[5/5] Running Evaluation...")
    if evaluate() is False:
        print("Pipeline stopped — Evaluation failed")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_full_pipeline()