from azureml.core import Experiment
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.webservice.aci import AciWebservice
from azureml.core import Workspace, Run
from azureml.data import TabularDataset
from azureml.exceptions import ComputeTargetException
from azureml.train.automl import AutoMLConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--primary_metric', dest='primary_metric', default='accuracy')
args = parser.parse_args()

experiment_name = 'udacity-instagram-users-train'
project_folder = './automl-pipeline-project'

ws = Workspace.from_config()
current_step_run = Run.get_context()

key = "instagram-fake-real-accounts"
dataset = ws.datasets[key]

automl_settings = {
    "max_concurrent_iterations": 3,
    "experiment_timeout_minutes": 25,
    "primary_metric": args.primary_metric, 
    "n_cross_validations": 5
}

automl_config = AutoMLConfig(task="classification", 
                             path = project_folder,
                             training_data = dataset,
                             label_column_name = "is_fake",
                             debug_log = "automl_errors.log",
                             **automl_settings)

experiment = Experiment(ws, experiment_name)
run = current_step_run.submit_child(automl_config, show_output=True)
logger.info(run)
print(run)
run.wait_for_completion()

