import pandas as pd
from matplotlib import pyplot as plt
from pandas.tseries.offsets import MonthEnd
from sklearn.model_selection import train_test_split
from yellowbrick.regressor import ResidualsPlot

from regression_models import *
from utils import get_dataset, Dataset


def plot_residuals(model, X_train, X_test, y_train, y_test, station, chemical, period, filename):
    title = 'Residuals Plot for {m} model - {p} - {s} - {c}'.format(m=type(model).__name__, s=station, p=period,
                                                                    c=chemical.upper())
    plt.figure()
    visualizer = ResidualsPlot(model, title=title)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show(outpath='generated_data/plots/residuals/{}.png'.format(filename))


def get_models():
    return {
        'linear': get_linear_model(),
        'huber': get_huber_model(),
        'poly': get_polynomial_model(),
        'rf': get_random_forest_model(),
        'gb': get_gradient_boosting_model(),
        'svr_lin': get_svr_linear_model(),
        'svr_poly': get_svr_polynomial_model(),
        'svr_rbf': get_svr_rbf_model()
    }


def execute(ds, chemical):
    dataset = get_dataset(ds)
    X = dataset['airqino_{}'.format(chemical)].values.reshape((-1, 1))
    y = dataset['arpat_{}'.format(chemical)].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    for (name, model) in get_models().items():
        plot_residuals(model, X_train, X_test, y_train, y_test, station=ds.name, chemical=chemical, period='All year',
                       filename='{d}_{c}_{m}'.format(d=ds.name.lower(), c=chemical, m=name))

        for month in pd.date_range('2020-01-01', '2020-12-31', freq='MS'):
            month_str = month.strftime('%b')
            month_start = month.strftime('%Y-%m-%d')
            month_end = (month + MonthEnd(1)).strftime('%Y-%m-%d')
            month_dataset = dataset.loc[month_start: month_end]
            month_X = month_dataset['airqino_{}'.format(chemical)].values.reshape((-1, 1))
            month_y = month_dataset['arpat_{}'.format(chemical)].values
            if month_dataset.empty or len(month_dataset.index) < 30:
                continue
            month_X_train, month_X_test, month_y_train, month_y_test = train_test_split(month_X, month_y,
                                                                                        test_size=0.25, random_state=42)
            plot_residuals(model, month_X_train, month_X_test, month_y_train, month_y_test,
                           station=ds.name, chemical=chemical, period=month_str,
                           filename='{d}_{c}_{m}_{p}'.format(d=ds.name.lower(), c=chemical, m=name,
                                                             p=month_str.lower()))


if __name__ == '__main__':

    for ds in [Dataset.SMART16_NO2, Dataset.SMART24, Dataset.SMART25, Dataset.SMART26]:
        execute(ds, chemical='no2')

    for ds in [Dataset.SMART16_PM]:
        execute(ds, chemical='pm2.5')
        execute(ds, chemical='pm10')
