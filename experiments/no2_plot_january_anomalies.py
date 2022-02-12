import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt

from utils.utils import get_dataset, Dataset

if __name__ == '__main__':
    dataset = get_dataset(Dataset.SMART16_NO2)

    # d1 = dataset.loc['2020-01-18': '2020-12-31']
    # X = d1['airqino_no2'].values.reshape((-1, 1))
    # y = d1['arpat_no2'].values
    # plt.scatter(X, y)
    # X = d2['airqino_no2'].values.reshape((-1, 1))
    # y = d2['arpat_no2'].values
    # plt.scatter(X, y, color='red')
    # plt.show()

    X = dataset['airqino_no2'].values.reshape((-1, 1))
    y = dataset['arpat_no2'].values

    model_result = sm.OLS(y, sm.add_constant(X)).fit()
    infl = model_result.get_influence()
    soglia = 4 / len(X)
    (a, p) = infl.cooks_distance
    mask = a < soglia
    outlier_mask = np.logical_not(mask)
    int_mask = mask.astype(int)

    print(len([i for i in int_mask if i == 0]))
    print(len(int_mask))

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(int_mask)), int_mask)
    plt.show()

    plt.scatter(X[mask], y[mask], color='green')
    plt.scatter(X[outlier_mask], y[outlier_mask], color='red')
    plt.show()
