from classifier import BinaryClassifier, OneVSRestClassifierExtended
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.estimator_checks import check_estimator
import pytest 

@parametrize_with_checks([OneVSRestClassifierExtended(DecisionTreeClassifier(), [])])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)

def main():
    test_sklearn_compatible_estimator(OneVSRestClassifierExtended(DecisionTreeClassifier(), []), check_estimator)

if __name__ == "__main__":
    main()