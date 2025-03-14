import numpy as np
from computer_vision_research.dataManagementClasses import insert_weights_into_feature_vector

class TestInsertWeightsFunc:
    X = np.array([1,2,3,4,5,6,7,8,9,10])
    Y = np.array([1,2,3,4,5,6,7,8,9,10])

    def test_insert_func_n_weights_2(self):
        expected = np.array([1,1,5,5,9,9,10,10])
        midx = 1
        endidx = 9 
        n_weights = 2
        feature_vector = np.array([self.X[0], self.Y[0],
                                   self.X[endidx], self.Y[endidx]])
        feature_vector = insert_weights_into_feature_vector(
            midx, endidx, n_weights, self.X, self.Y, 2, feature_vector
        )
        np.testing.assert_array_equal(feature_vector, expected)

    def test_insert_func_n_weights_4(self):
        expected = np.array([1,1,3,3,5,5,7,7,9,9,10,10])
        midx = 1
        endidx = 9 
        n_weights = 4 
        feature_vector = np.array([self.X[0], self.Y[0],
                                   self.X[endidx], self.Y[endidx]])
        feature_vector = insert_weights_into_feature_vector(
            midx, endidx, n_weights, self.X, self.Y, 2, feature_vector
        )
        np.testing.assert_array_equal(feature_vector, expected)

    def test_insert_func_n_weights_4_midx_5(self):
        expected = np.array([1,1,6,6,7,7,8,8,9,9,10,10])
        midx = 5
        endidx = 9 
        n_weights = 4 
        feature_vector = np.array([self.X[0], self.Y[0],
                                   self.X[endidx], self.Y[endidx]])
        feature_vector = insert_weights_into_feature_vector(
            midx, endidx, n_weights, self.X, self.Y, 2, feature_vector
        )
        np.testing.assert_array_equal(feature_vector, expected)