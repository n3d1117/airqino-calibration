import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from regression.regression_models import get_polynomial_model
from regression.regression_summary import avg
from utils.utils import get_dataset, Dataset

if __name__ == '__main__':
    dataset_ = get_dataset(Dataset.SMART16_NEW_PM_8H)


    def seasons(df):
        df_autumn = df[df.index.map(lambda t: t.month in [9, 10, 11] and t.year == 2020)]
        df_winter = df[
            df.index.map(lambda t: (t.month == 12 and t.year == 2020) or (t.month in [1, 2] and t.year == 2021))
        ]
        df_spring = df[df.index.map(lambda t: t.month in [3, 4, 5] and t.year == 2021)]
        df_summer = df[df.index.map(lambda t: t.month in [6, 7, 8] and t.year == 2021)]
        return df_autumn, df_winter, df_spring, df_summer


    labels = ['Autumn', 'Winter', 'Spring', 'Summer']

    # Scatterplot by season
    for i, season in enumerate(seasons(dataset_)):
        X = season['airqino_{}'.format('pm2.5')].values.reshape((-1, 1))
        y = season['arpat_{}'.format('pm2.5')].values
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
        line_x = np.arange(X.min(), X.max())[:, np.newaxis]
        poly = get_polynomial_model()
        poly.fit(X, y)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(X, y, label='Data', alpha=.5)
        ax.set_xlabel('AirQino ({})'.format('µg/m³'))
        ax.set_ylabel('ARPAT (µg/m³)')
        ax.set_title('SMART16_new | PM2.5 | {} ({})'.format(labels[i], len(X)))

        r1 = []
        for _ in range(500):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
            poly = get_polynomial_model()
            poly.fit(X_train, y_train)
            r1.append(poly.score(X_test, y_test))
        ax.plot(line_x, poly.predict(line_x), color='red', linewidth=2,
                label='Poly regressor\n(d=2, R²={})'.format(avg(r1)), alpha=.5)
        ax.legend()
        plt.show()
