from computer_vision_research.dataManagementClasses import TrackedObject
import numpy as np

def downscaler_10D_features(featureVector, framewidth=1920, frameheight=1080):
    if featureVector is None:
                return None
    ratio = framewidth / frameheight

    """
    ret_featureVector = np.array([])
    for i in range(featureVector.shape[0]):
        if i == 0 or i % 2 == 0:
            ret_featureVector = np.append(ret_featureVector, [featureVector[i]/(framewidth*ratio)])
        else:
            ret_featureVector = np.append(ret_featureVector, [featureVector[i]/framewidth])
    print(ret_featureVector)
    return ret_featureVector

    if feature_v3:
        return np.array([featureVector[0] / framewidth*ratio, featureVector[1] / frameheight, 
                        featureVector[2] / framewidth*ratio, featureVector[3] / frameheight, 
                        featureVector[4] / framewidth*ratio, featureVector[5] / frameheight,])

    """
    return np.array([featureVector[0] / framewidth*ratio, featureVector[1] / frameheight, featureVector[2] / framewidth*ratio,
                    featureVector[3] / frameheight, featureVector[4] / framewidth*ratio, featureVector[5] / frameheight,
                    featureVector[6] / framewidth*ratio, featureVector[7] / frameheight, featureVector[8] / framewidth*ratio,
                    featureVector[9] / frameheight], dtype=float)

def dowscaler2test(featureVector, framewidth=1920, frameheight=1080):
    if featureVector is None:
                return None
    ratio = framewidth / frameheight
    ret_featureVector = np.array([], dtype=float)
    for i in range(featureVector.shape[0]):
        if i % 2 == 0:
            ret_featureVector = np.append(ret_featureVector, [featureVector[i]/framewidth*ratio])
        else:
            ret_featureVector = np.append(ret_featureVector, [featureVector[i]/frameheight])
    return ret_featureVector

class TestScalerFunctions:
    test_features = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]) * 108 
    def test_downscaler(self):
        print(self.test_features.shape)
        expected = np.apply_along_axis(downscaler_10D_features, 1, self.test_features)
        returned = np.apply_along_axis(TrackedObject.downscale_feature, 1, self.test_features)
        print(f"Expected output:\n {expected}\n")
        print(f"Returned output:\n {returned}\n")
        np.testing.assert_array_equal(expected, returned)
