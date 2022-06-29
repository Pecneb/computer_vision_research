from cv2 import line
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

def main():
    X = np.linspace(0, 100, num=25)
    X_test = np.linspace(110, 210, num=25)
    y = np.sin(X)
    fig, ax = plt.subplots()
    regr = linear_model.LinearRegression()
    regr.fit(X.reshape(-1,1), y)
    y_pred = regr.predict(X_test.reshape(-1,1))
    print("Coefficients: \n", regr.coef_)
    ax.plot(X,y, color="black")
    ax.plot(X_test, y_pred, color="red")
    plt.show()

if __name__ == "__main__":
    main()