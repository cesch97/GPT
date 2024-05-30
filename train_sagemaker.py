import sagemaker
from sagemaker.pytorch import PyTorch
import json


sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()


with open('config/sagemaker_config.json') as f:
    config = json.load(f)


estimator = PyTorch(
    entry_point="main.py",
    instance_type="ml.p2.xlarge",
    instance_count=1,
    framework_version="1.8.0",
    py_version="py3",
    source_dir=".",
    hyperparameters=config,
    role=role
)

estimator.fit()
