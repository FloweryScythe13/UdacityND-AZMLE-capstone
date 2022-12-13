import json
from azureml.core.model import Model
import pickle
import numpy as np

def init():
    global my_model
    model_path = Model.get_model_path("ig-aml-classifier")
    my_model = pickle.load(model_path)


def run(request):
    req_data = np.array(json.loads(request)['data'])
    preds = my_model.predict(req_data)
    classnames = ["not-fake", "fake"]
    predicted_classes = [classnames[prediction] for prediction in preds]
    return json.dumps(predicted_classes)