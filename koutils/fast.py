# Project
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self):
        pass
    
    def linear_regression(self, x: list, y: list, title: str = "Linear Regression", xlabel: str = "x", ylabel: str = "y"):
        """Linear regression plotter

        Args:
            x (list): The data for the x-axis.
            y (list): The data for the y-axis.
            title (str, optional): The title of the plot. Defaults to "Linear Regression".
            xlabel (str, optional): The label for the x-axis. Defaults to "x".
            ylabel (str, optional): The label for the y-axis. Defaults to "y".
        """
        df = pd.DataFrame({"x": x, "y": y})
        x = df[["x"]]
        y = df[["y"]]
        
        clf = LinearRegression()
        clf.fit(x, y)
        y_pred = clf.predict(x)
        m = clf.coef_
        b = clf.intercept_
        fig, ax = plt.subplots()
        ax.scatter(x, y, label="Original", color="red")
        ax.plot(x, y_pred, label="Regression", color="black")
        ax.set_title(f"y = {m[0][0]:.3} * x + {b[0]:.3}", loc='right', fontsize=10)
        fig.suptitle(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid()
        ax.legend()
        plt.show()

class Print:
    def scientific_print(value: float, unit: str = None, name: str = None):
        """Prints a value in scientific notation

        Args:
            value (float): The value to print.
            unit (str): The unit of the value.
            name (str): The name of the value.
        """
        if unit is None and name is not None:
            print(f"{name} = {value:.3e}")
            
        elif name is None and unit is not None:
            print(f"{value:.3e} {unit}")
            
        elif unit is None and name is None:
            print(f"{value:.3e}")
            
        else:
            print(f"{name} = {value:.3e} {unit}")
            
        
        
