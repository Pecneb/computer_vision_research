import numpy as np
from utils.dataset import detectionFactory, trackedObjectFactory
from computer_vision_research.classification import make_feature_vectors_version_six
from computer_vision_research.dataManagementClasses import TrackedObject, Detection

class TestFeatureVectorV6:
    trackedObjects = np.array([
        trackedObjectFactory((
            [detectionFactory(1, 1, "car", 0.75, 0,0,0,0,0,0,0,0) for i in range(100)], 
            np.linspace(0, 1.5, 100),
            np.linspace(0, 1.5, 100),
            np.linspace(0, 1.5, 100),
            np.linspace(0, 1.5, 100),
            np.linspace(0, 1.5, 100),
            np.linspace(0, 1.5, 100)
        ))
    ])

    def test_make_feature_vectors_version_six(self):
        wv = np.array([1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0])
        print(wv.shape)
        x, y, metadata = make_feature_vectors_version_six(self.trackedObjects, np.array([0]), 15, wv)
        x_expected_sample = np.array([self.trackedObjects[0].history_X[0],
                                 self.trackedObjects[0].history_Y[0],
                                 self.trackedObjects[0].history_VX_calculated[0],
                                 self.trackedObjects[0].history_VY_calculated[0],
                                 self.trackedObjects[0].history_X[6],
                                 self.trackedObjects[0].history_Y[6],
                                 self.trackedObjects[0].history_VX_calculated[6],
                                 self.trackedObjects[0].history_VY_calculated[6],
                                 self.trackedObjects[0].history_X[14],
                                 self.trackedObjects[0].history_Y[14],
                                 self.trackedObjects[0].history_VX_calculated[14],
                                 self.trackedObjects[0].history_VY_calculated[14]]) * wv
        np.testing.assert_equal(x.shape[1], x_expected_sample.shape[0])
        np.testing.assert_array_equal(x[0], x_expected_sample)