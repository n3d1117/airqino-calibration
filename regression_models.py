from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR


def get_linear_model():
    return linear_model.LinearRegression(n_jobs=-1)


def get_huber_model():
    return linear_model.HuberRegressor(epsilon=3)


def get_polynomial_model():
    return make_pipeline(PolynomialFeatures(degree=2), linear_model.LinearRegression(n_jobs=-1))


def get_random_forest_model():
    return RandomForestRegressor(n_estimators=10, n_jobs=-1)


def get_gradient_boosting_model():
    params = {'learning_rate': 0.01, 'loss': 'squared_error', 'max_depth': 3, 'min_samples_split': 10,
              'n_estimators': 100}
    return GradientBoostingRegressor(**params)


def get_svr_linear_model():
    return SVR(kernel='linear', C=10, gamma='scale')


def get_svr_polynomial_model():
    return SVR(kernel='poly', C=10, gamma='scale', degree=2)


def get_svr_rbf_model():
    return SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
