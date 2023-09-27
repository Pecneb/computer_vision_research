import numpy as np
from utils.dataset import trackedObjectFactory, detectionFactory
from computer_vision_research.classification import make_feature_vectors_version_five
from computer_vision_research.dataManagementClasses import TrackedObject, Detection

class TestFeatureVectorV5:
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

    def test_make_feature_vectors_version_five(self):
        x, y, metadata = make_feature_vectors_version_five(self.trackedObjects, np.array([0]), 15, 5)
        x_expected_sample = np.array([self.trackedObjects[0].history_X[0],
                                 self.trackedObjects[0].history_Y[0],
                                 self.trackedObjects[0].history_VX_calculated[9],
                                 self.trackedObjects[0].history_VY_calculated[9],
                                 self.trackedObjects[0].history_VX_calculated[10],
                                 self.trackedObjects[0].history_VY_calculated[10],
                                 self.trackedObjects[0].history_VX_calculated[11],
                                 self.trackedObjects[0].history_VY_calculated[11],
                                 self.trackedObjects[0].history_VX_calculated[12],
                                 self.trackedObjects[0].history_VY_calculated[12],
                                 self.trackedObjects[0].history_VX_calculated[13],
                                 self.trackedObjects[0].history_VY_calculated[13],
                                 self.trackedObjects[0].history_X[9],
                                 self.trackedObjects[0].history_Y[9],
                                 self.trackedObjects[0].history_X[10],
                                 self.trackedObjects[0].history_Y[10],
                                 self.trackedObjects[0].history_X[11],
                                 self.trackedObjects[0].history_Y[11],
                                 self.trackedObjects[0].history_X[12],
                                 self.trackedObjects[0].history_Y[12],
                                 self.trackedObjects[0].history_X[13],
                                 self.trackedObjects[0].history_Y[13],
                                 self.trackedObjects[0].history_X[14],
                                 self.trackedObjects[0].history_Y[14],])
        np.testing.assert_equal(x.shape[1], x_expected_sample.shape[0])
        np.testing.assert_array_equal(x[0], x_expected_sample)

    def test_make_feature_vectors_version_five_50(self):
        x, y, metadata = make_feature_vectors_version_five(self.trackedObjects, np.array([0]), 15, 5)
        x_expected_sample = np.array([self.trackedObjects[0].history_X[50],
                                 self.trackedObjects[0].history_Y[50],
                                 self.trackedObjects[0].history_VX_calculated[59],
                                 self.trackedObjects[0].history_VY_calculated[59],
                                 self.trackedObjects[0].history_VX_calculated[60],
                                 self.trackedObjects[0].history_VY_calculated[60],
                                 self.trackedObjects[0].history_VX_calculated[61],
                                 self.trackedObjects[0].history_VY_calculated[61],
                                 self.trackedObjects[0].history_VX_calculated[62],
                                 self.trackedObjects[0].history_VY_calculated[62],
                                 self.trackedObjects[0].history_VX_calculated[63],
                                 self.trackedObjects[0].history_VY_calculated[63],
                                 self.trackedObjects[0].history_X[59],
                                 self.trackedObjects[0].history_Y[59],
                                 self.trackedObjects[0].history_X[60],
                                 self.trackedObjects[0].history_Y[60],
                                 self.trackedObjects[0].history_X[61],
                                 self.trackedObjects[0].history_Y[61],
                                 self.trackedObjects[0].history_X[62],
                                 self.trackedObjects[0].history_Y[62],
                                 self.trackedObjects[0].history_X[63],
                                 self.trackedObjects[0].history_Y[63],
                                 self.trackedObjects[0].history_X[64],
                                 self.trackedObjects[0].history_Y[64],])
        np.testing.assert_equal(x.shape[1], x_expected_sample.shape[0])
        np.testing.assert_array_equal(x[50], x_expected_sample)

    def test_make_feature_vectors_version_five_30(self):
        x, y, metadata = make_feature_vectors_version_five(self.trackedObjects, np.array([0]), 15, 5)
        x_expected_sample = np.array([self.trackedObjects[0].history_X[30],
                                 self.trackedObjects[0].history_Y[30],
                                 self.trackedObjects[0].history_VX_calculated[39],
                                 self.trackedObjects[0].history_VY_calculated[39],
                                 self.trackedObjects[0].history_VX_calculated[40],
                                 self.trackedObjects[0].history_VY_calculated[40],
                                 self.trackedObjects[0].history_VX_calculated[41],
                                 self.trackedObjects[0].history_VY_calculated[41],
                                 self.trackedObjects[0].history_VX_calculated[42],
                                 self.trackedObjects[0].history_VY_calculated[42],
                                 self.trackedObjects[0].history_VX_calculated[43],
                                 self.trackedObjects[0].history_VY_calculated[43],
                                 self.trackedObjects[0].history_X[39],
                                 self.trackedObjects[0].history_Y[39],
                                 self.trackedObjects[0].history_X[40],
                                 self.trackedObjects[0].history_Y[40],
                                 self.trackedObjects[0].history_X[41],
                                 self.trackedObjects[0].history_Y[41],
                                 self.trackedObjects[0].history_X[42],
                                 self.trackedObjects[0].history_Y[42],
                                 self.trackedObjects[0].history_X[43],
                                 self.trackedObjects[0].history_Y[43],
                                 self.trackedObjects[0].history_X[44],
                                 self.trackedObjects[0].history_Y[44],])
        np.testing.assert_equal(x.shape[1], x_expected_sample.shape[0])
        np.testing.assert_array_equal(x[30], x_expected_sample)

    def test_make_feature_vectors_version_five_stride_10(self):
        x, y, metadata = make_feature_vectors_version_five(self.trackedObjects, np.array([0]), 10, 5)
        x_expected_sample = np.array([self.trackedObjects[0].history_X[30],
                                 self.trackedObjects[0].history_Y[30],
                                 self.trackedObjects[0].history_VX_calculated[34],
                                 self.trackedObjects[0].history_VY_calculated[34],
                                 self.trackedObjects[0].history_VX_calculated[35],
                                 self.trackedObjects[0].history_VY_calculated[35],
                                 self.trackedObjects[0].history_VX_calculated[36],
                                 self.trackedObjects[0].history_VY_calculated[36],
                                 self.trackedObjects[0].history_VX_calculated[37],
                                 self.trackedObjects[0].history_VY_calculated[37],
                                 self.trackedObjects[0].history_VX_calculated[38],
                                 self.trackedObjects[0].history_VY_calculated[38],
                                 self.trackedObjects[0].history_X[34],
                                 self.trackedObjects[0].history_Y[34],
                                 self.trackedObjects[0].history_X[35],
                                 self.trackedObjects[0].history_Y[35],
                                 self.trackedObjects[0].history_X[36],
                                 self.trackedObjects[0].history_Y[36],
                                 self.trackedObjects[0].history_X[37],
                                 self.trackedObjects[0].history_Y[37],
                                 self.trackedObjects[0].history_X[38],
                                 self.trackedObjects[0].history_Y[38],
                                 self.trackedObjects[0].history_X[39],
                                 self.trackedObjects[0].history_Y[39],])
        np.testing.assert_equal(x.shape[1], x_expected_sample.shape[0])
        np.testing.assert_array_equal(x[30], x_expected_sample)

    def test_make_feature_vectors_version_five_stride_10_n_weights_3(self):
        x, y, metadata = make_feature_vectors_version_five(self.trackedObjects, np.array([0]), 10, 2)
        x_expected_sample = np.array([self.trackedObjects[0].history_X[30],
                                 self.trackedObjects[0].history_Y[30],
                                 self.trackedObjects[0].history_VX_calculated[36],
                                 self.trackedObjects[0].history_VY_calculated[36],
                                 self.trackedObjects[0].history_VX_calculated[38],
                                 self.trackedObjects[0].history_VY_calculated[38],
                                 self.trackedObjects[0].history_X[36],
                                 self.trackedObjects[0].history_Y[36],
                                 self.trackedObjects[0].history_X[38],
                                 self.trackedObjects[0].history_Y[38],
                                 self.trackedObjects[0].history_X[39],
                                 self.trackedObjects[0].history_Y[39],])
        np.testing.assert_equal(x.shape[1], x_expected_sample.shape[0])
        np.testing.assert_array_equal(x[30], x_expected_sample)