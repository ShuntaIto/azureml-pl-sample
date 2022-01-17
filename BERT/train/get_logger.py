
import pytorch_lightning as pl
from azureml.core import Run

def get_logger():
    run = Run.get_context()
    ws = run.experiment.workspace
    mlflow_url = ws.get_mlflow_tracking_uri()

    mlflow_url = run.experiment.workspace.get_mlflow_tracking_uri()
    mlf_logger = pl.loggers.MLFlowLogger(
        experiment_name=run.experiment.name,
        tracking_uri=mlflow_url)
    mlf_logger._run_id = run.id
    return mlf_logger