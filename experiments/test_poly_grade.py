import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from regression.regression_summary import avg
from utils.utils import get_dataset, Dataset

if __name__ == '__main__':
    d = get_dataset(Dataset.SMART16_NEW_PM_8H)
    X = d['airqino_pm2.5'].values.reshape((-1, 1))
    y = d['arpat_pm2.5'].values

    line_x = np.arange(X.min(), X.max())[:, np.newaxis]

    # 2
    model1 = make_pipeline(StandardScaler(with_mean=False), PolynomialFeatures(degree=2, include_bias=False),
                           linear_model.LinearRegression(n_jobs=-1))
    model1.fit(X, y)

    # 3
    model2 = make_pipeline(StandardScaler(with_mean=False), PolynomialFeatures(degree=3, include_bias=False),
                           linear_model.LinearRegression(n_jobs=-1))
    model2.fit(X, y)

    # 4
    model3 = make_pipeline(StandardScaler(with_mean=False), PolynomialFeatures(degree=4, include_bias=False),
                           linear_model.LinearRegression(n_jobs=-1))
    model3.fit(X, y)

    r1, r2, r3 = [], [], []
    for _ in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        model3.fit(X_train, y_train)
        r1.append(model1.score(X_test, y_test))
        r2.append(model2.score(X_test, y_test))
        r3.append(model3.score(X_test, y_test))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X, y, label='Data', alpha=.5)
    ax.set_xlabel('AirQino ({})'.format('µg/m³'))
    ax.set_ylabel('ARPAT (µg/m³)')
    ax.set_title('SMART16_new | PM2.5 ({})'.format(len(X)))
    ax.plot(line_x, model1.predict(line_x), color='red', linewidth=2,
            label='Poly regressor\n(d=2, R²={})'.format(avg(r1)), alpha=.5)
    ax.plot(line_x, model2.predict(line_x), color='green', linewidth=2,
            label='Poly regressor\n(d=3, R²={})'.format(avg(r2)), alpha=.5)
    ax.plot(line_x, model3.predict(line_x), color='yellow', linewidth=2,
            label='Poly regressor\n(d=4, R²={})'.format(avg(r3)), alpha=.5)
    ax.legend()
    plt.show()
