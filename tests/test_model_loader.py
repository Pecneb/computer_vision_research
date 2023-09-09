from computer_vision_research.classifier import OneVSRestClassifierExtended
from computer_vision_research.dataManagementClasses import Detection, TrackedObject
from utils.models import load_model

class TestModelLoader:
    def test_loader(self):
        model = load_model("../research_data/0001_1_37min/models/binary_SVM_kernel_rbf_probability_True.joblib")
        assert type(model) == OneVSRestClassifierExtended 
        assert type(model.tracks[0]) == TrackedObject
        assert type(model.tracks[0].history[0]) == Detection