from typing import List
from computer_vision_research.dataManagementClasses import TrackedObject, Detection
from utils.dataset import detectionFactory, trackedObjectFactory


class TestDataManagementClasses:
    detection1 = Detection("car", 0.87, 0.670, 0.1023, 0.5, 0.5, 123)
    detection2 = Detection("person", 0.27, 0.170, 0.523, 0.2, 0.1, 163)
    track1 = TrackedObject(1, detection1)
    track2 = TrackedObject(1, detection1)
    track3 = TrackedObject(2, detection2)

    def test_eq(self):
        assert (self.track1 == self.track2) == True, "track1 and track2 should be equal"
        assert (self.track1 == self.track3) == False, "track1 and track3 should not be equal"
        assert (self.track2 == self.track3) == False, "track2 and track3 should not be equal"