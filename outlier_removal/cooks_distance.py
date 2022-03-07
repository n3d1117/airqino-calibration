import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from statsmodels.tools.eval_measures import rmse
from yellowbrick.regressor import CooksDistance

from regression.regression_summary import evaluate, print_annual_results
from utils.utils import get_dataset, Dataset


def annual_summary(X, y):
    results = evaluate(X, y)
    print_annual_results(results, title='All year | {n} | {c}'.format(c='NO2', n='SMART16-CAPANNORI'))


if __name__ == '__main__':
    dataset = get_dataset(Dataset.SMART16_NEW_PM_8H)
    # dataset = dataset.loc['2020-01-18': '2020-12-30']
    X = dataset['airqino_{}'.format('pm10')].values.reshape((-1, 1))
    y = dataset['arpat_{}'.format('pm10')].values

    model_result = sm.OLS(y, sm.add_constant(X)).fit()
    ypred = model_result.predict(sm.add_constant(X))
    print(model_result.rsquared)
    print(rmse(y, ypred))

    # ypred = model_result.predict(X)
    # print(model_result.summary())
    # annual_summary(X, y)

    infl = model_result.get_influence()
    soglia = 4 / len(X)
    (a, p) = infl.cooks_distance
    mask = a < soglia
    model_result = sm.OLS(y[mask], sm.add_constant(X[mask])).fit()
    ypred = model_result.predict(sm.add_constant(X[mask]))
    print(model_result.rsquared)
    print(rmse(y[mask], ypred))
    # annual_summary(X[mask], y[mask])

    print('totale senza outlier: {}, ovvero tolti {} punti (su {} totali, {}%)'.format(
        len(X[mask]), len(X) - len(X[mask]), len(X), round((100 / len(X)) * (len(X) - len(X[mask])), 2)
    ))

    outlier_mask = np.logical_not(mask)
    # plt.scatter(X, y, label='Data', alpha=.5)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('SMART16 | NO₂ | Rimozione di outlier con distanza di Cook ({}%)'.format(
        round((100 / len(X)) * (len(X) - len(X[mask])), 2)))
    ax.set_xlabel('AirQino ({})'.format('counts'))
    ax.set_ylabel('ARPAT (µg/m³)')
    ax.scatter(
        X[mask], y[mask], color=(0.2, 0.4, 0.6, 0.6), marker=".", label="Inliers"
    )
    ax.scatter(
        X[outlier_mask], y[outlier_mask], color="tab:red", alpha=.8, marker="x", label="Outliers"
    )
    # ax.plot(X, ypred, color='cyan', label='pred')
    ax.legend()
    # plt.savefig('../thesis_img/cook_no2.png', dpi=300)

    visualizer = CooksDistance()
    visualizer.fit(X, y)
    visualizer.show()
