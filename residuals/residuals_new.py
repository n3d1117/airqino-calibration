import numpy as np
import pandas as pd
import seaborn
import statsmodels.api as sm
from matplotlib import pyplot as plt
from pandas import DataFrame
from pandas.tseries.offsets import MonthEnd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from regression.regression_models import get_polynomial_model
from utils.utils import Dataset, get_dataset


def do(X, y, title, filename):
    polynomial_features = PolynomialFeatures(degree=2)
    xp = polynomial_features.fit_transform(X)
    model_result = sm.OLS(y, xp).fit()

    mu, std = stats.norm.fit(model_result.resid)
    fig, ax = plt.subplots()
    seaborn.histplot(x=model_result.resid, ax=ax, stat="density", linewidth=0, kde=True)
    ax.set(title="Distribution of residuals | {}".format(title), xlabel="Residual")
    xmin, xmax = plt.xlim()  # the maximum x values from the histogram above
    x = np.linspace(xmin, xmax, 100)  # generate some x values
    p = stats.norm.pdf(x, mu, std)  # calculate the y values for the normal curve
    seaborn.lineplot(x=x, y=p, color="orange", ax=ax)
    plt.savefig('generated_data/plots/residuals_new/{}_distr.png'.format(filename))

    fig, ax = plt.subplots()
    sm.qqplot(model_result.resid, ax=ax, line='s')
    ax.set(title="Q-Q plot | {}".format(title))
    plt.savefig('generated_data/plots/residuals_new/{}_qq.png'.format(filename))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
    model = get_polynomial_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    residuals = DataFrame(y_test - y_pred)
    fig, ax = plt.subplots()
    ax.set(title='Residuals Plot | {}'.format(title), ylabel='Residuals')
    ax.plot(residuals)
    plt.savefig('generated_data/plots/residuals_new/{}_plot.png'.format(filename))

    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals)
    ax.set(title="Residuals scatterplot | {}".format(title), xlabel="y_pred", ylabel='Residuals')
    plt.savefig('generated_data/plots/residuals_new/{}_scatter.png'.format(filename))


if __name__ == '__main__':
    for ds in [Dataset.SMART16_NO2]:
        dataset = get_dataset(ds)
        X = dataset['airqino_{}'.format('no2')].values.reshape((-1, 1))
        y = dataset['arpat_{}'.format('no2')].values
        date_range = pd.date_range('2020-01-01', '2020-12-31', freq='MS')

        do(X, y, title='SMART16 | NO2 | All year ({})'.format(len(X)), filename='no2_allyear')

        for month in date_range:
            month_str = month.strftime('%B')
            month_start = month.strftime('%Y-%m-%d')
            month_end = (month + MonthEnd(1)).strftime('%Y-%m-%d')
            month_dataset = dataset.loc[month_start: month_end]
            month_X = month_dataset['airqino_{}'.format('no2')].values.reshape((-1, 1))
            month_y = month_dataset['arpat_{}'.format('no2')].values
            if month_dataset.empty:
                continue
            do(month_X, month_y, title='SMART16 | NO2 | {} ({})'.format(month_str, len(month_X)),
               filename='no2_{}'.format(month_str.lower()))

    for ds in [Dataset.SMART16_NEW_PM_8H]:
        dataset = get_dataset(ds)

        # todo seasons
        for chem in ['pm2.5', 'pm10']:
            X = dataset['airqino_{}'.format(chem)].values.reshape((-1, 1))
            y = dataset['arpat_{}'.format(chem)].values
            date_range = pd.date_range('2020-09-01', '2021-08-31', freq='MS')
            do(X, y, title='SMART16_new | {} | All year ({})'.format(chem.upper(), len(X)), filename=chem + '_allyear')

            for month in date_range:
                month_str = month.strftime('%B')
                month_start = month.strftime('%Y-%m-%d')
                month_end = (month + MonthEnd(1)).strftime('%Y-%m-%d')
                month_dataset = dataset.loc[month_start: month_end]
                month_X = month_dataset['airqino_{}'.format(chem)].values.reshape((-1, 1))
                month_y = month_dataset['arpat_{}'.format(chem)].values
                if month_dataset.empty:
                    continue
                do(month_X, month_y, title='SMART16 | {} | {} ({})'.format(chem.upper(), month_str, len(month_X)),
                   filename='{}_{}'.format(chem, month_str.lower()))
