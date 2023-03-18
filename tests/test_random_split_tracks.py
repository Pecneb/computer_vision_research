from processing_utils import random_split_tracks
import numpy as np

class TestRandomSplitFunc:
    example_dataset = np.linspace(0, 100, 1000)

    def test_train_not_equals_test(self):
        train, test = random_split_tracks(self.example_dataset, 0.75, 1)
        for e in train:
            for t in test:
                assert e != t

    def test_change_percentage(self):
        train, test = random_split_tracks(self.example_dataset, 0.75, 1)
        train_test_a = np.append(train, [test])
        changed = 0
        for i in range(len(train_test_a)):
            if train_test_a[i] != self.example_dataset[i]:
                changed += 1
        assert changed/len(train_test_a) > 0.75 