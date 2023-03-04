import numpy as np
from processing_utils import make_feature_vectors_version_five, detectionFactory, trackedObjectFactory
from dataManagementClasses import TrackedObject, Detection

class TestFeatureVectorV5:
    trackedObjects = np.array([
        trackedObjectFactory((
            [detectionFactory(1, 1, "car", 0.75, 0.5, 0.75, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0), detectionFactory(1, 2, "car", 0.75, 0.51, 0.75, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0), 
             detectionFactory(1, 3, "car", 0.75, 0.53, 0.75, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0), detectionFactory(1, 4, "car", 0.75, 0.5, 0.75, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0)], 
            np.array([0.5, 0.51, 0.53, 0.54, 0.56]), np.array([0.75, 0.75, 0.75, 0.75, 0.75]), np.array([0.0, 0.01, 0.02, 0.75, 0.0]), np.array([0.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.01, 0.01, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        ))
    ])

    def test_make_feature_vectors_version_five(self):
        X, y, metadata = make_feature_vectors_version_five(self.trackedObjects, np.array([0]), 3, 1)
        X_expected = np.array([[0.5, 0.75, 0.51, 0.75, 0.53, 0.75],
                               [0.51, 0.75, 0.53, 0.75, 0.54, 0.75]])
        y_expected = np.array([0,0])
        np.testing.assert_array_equal(X, X_expected)
        np.testing.assert_array_equal(y, y_expected)