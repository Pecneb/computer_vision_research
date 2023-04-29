import processing_utils 
from classifier import OneVSRestClassifierExtended
from dataManagementClasses import Detection, TrackedObject

class TestModelLoader:
    def test_loader(self):
        model = processing_utils.load_model("../research_data/0001_1_37min/models/binary_SVM_kernel_rbf_probability_True.joblib")
        assert type(model) == OneVSRestClassifierExtended 
        assert type(model.tracks[0]) == TrackedObject
        assert type(model.tracks[0].history[0]) == Detection