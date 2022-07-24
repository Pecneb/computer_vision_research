from tkinter import W
from cv2 import line
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

def polynom(X: np.ndarray):
    y = (-X**3 + -X**2 + -X)
    return y

def f(x):
    """Function to be approximated by polynomial interpolation."""
    return x * np.sin(x)


def main():
    # whole range we want to plot
    x_plot = np.linspace(10, 15, 100)
    x_train = np.linspace(0, 10, 100)
    rng = np.random.RandomState(0)
    x_train = np.sort(rng.choice(x_train, size=50, replace=False))
    y_train = polynom(x_train) 

    # create 2D-array versions of these arrays to feed to transformers
    X_train = x_train[:, np.newaxis]
    X_plot = x_plot[:, np.newaxis]

    polynom_model = make_pipeline(PolynomialFeatures(degree=3), Ridge(alpha=1e-3))
    polynom_model.fit(x_train.reshape(-1,1), y_train)
    y_plot = polynom_model.predict(x_plot.reshape(-1,1))

    # B-spline with 4 + 3 - 1 = 6 basis functions
    model = make_pipeline(SplineTransformer(n_knots=4, degree=3), Ridge(alpha=1e-3))
    model.fit(X_plot, y_plot) 

    y_spline  = model.predict(X_plot)
    print(y_spline)
    fig, ax= plt.subplots(1,1)
    ax.plot(x_plot, y_spline, label="B-spline")
    ax.scatter(x_train, y_train)
    ax.legend(loc="lower center")
    plt.show()

if __name__ == "__main__":
    main()