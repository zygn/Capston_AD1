import numpy as np
from matplotlib import pyplot as plt

from sklearn import linear_model, datasets

class ObstacleDetective():
    
    def __init__(self):
        self.samples = 1000
        self.outliers = 50
        self.X = None
        self.Y = None
        self.coef = None
        self.line_X = None
        self.line_Y = None
        self.line_Y_ransac = None 

        self.load_list()
        self.fit()

    
    def load_list(self, type_check=None):
        if type_check == None: pass
        elif type(type_check) == str and type_check.split(".")[-1].lower() == "csv": pass 
        elif type(type_check) == list:
                

            self.X = X
            self.Y = Y
            self.coef = coef
        
    def fit(self):
        np.random.seed(0)
        self.X[:self.outliers] = 3 + 0.5 * np.random.normal(size=(self.outliers, 1))
        self.y[:self.outliers] = -3 + 10 * np.random.normal(size=self.outliers)

        lr = linear_model.LinearRegression()
        lr.fit()

        ransac = linear_model.RANSACRegressor()
        ransac.fit(self.X, self.Y)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)

        self.line_X = np.arange(self.X.min(), self.X.max())[:, np.newaxis]
        self.line_Y = lr.predict(self.line_X)
        self.line_y_ransac = ransac.predict(self.line_X)

        print("Estimated coefficients (true, linear regression, RANSAC):")
        print(self.coef, lr.coef_, ransac.estimator_.coef_)




    
        