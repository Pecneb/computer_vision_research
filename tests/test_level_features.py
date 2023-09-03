from computer_vision_research.classification import level_features
import numpy as np

class TestLevelFeatures:
    def test_level_features(self):
        y_in = np.array([0,0,0,
                    1,1,1,1,
                    2,2,2,2,2,
                    3,3,3,3,3,3,
                    4,4,4,4,4,4,4])
        X_in = np.ones(shape=(y_in.shape[0], 6))
        y_true = np.array([0,0,0,
                        1,1,1,
                        2,2,2,
                        3,3,3,
                        4,4,4])
        X_true = np.ones(shape=(y_true.shape[0], 6))
        X_out, y_out = level_features(X_in, y_in, 1)
        assert (y_true.shape == y_out.shape)
        assert (X_true.shape == X_out.shape)
        assert (X_out.shape[0] == y_out.shape[0])
        np.testing.assert_array_equal(X_true, X_out)
        np.testing.assert_array_equal(y_true, y_out)