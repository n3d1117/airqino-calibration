import statsmodels.api as sm
from matplotlib import pyplot as plt

from regression.regression_models import get_linear_model
from utils.utils import get_dataset, Dataset


def execute(dataset, chemical, name):
    dataset = get_dataset(dataset)

    # Get X and y
    X = dataset['airqino_{}'.format(chemical)].values.reshape((-1, 1))
    y = dataset['arpat_{}'.format(chemical)].values

    # Get regression model and remove outliers using Cook's distance
    model = get_linear_model()
    m = sm.OLS(y, sm.add_constant(X)).fit()
    infl = m.get_influence()
    soglia = 4 / len(X)
    (a, p) = infl.cooks_distance
    mask = a < soglia

    # Fit and predict
    model.fit(X[mask], y[mask])
    y_pred = model.predict(X[mask])

    # Plot
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    ax.plot(y[mask], color='tab:blue', label='ARPAT (Reference)', alpha=.8, linewidth=1.2)
    ax.plot(X[mask], color='tab:gray', label='AirQino (Raw)', alpha=.4, linewidth=1)
    ax.plot(y_pred, color='tab:red', label='AirQino (Calibrated)', alpha=.8, linewidth=1)
    ax.set_title('SMART16 | {} | Reference vs Raw vs Calibrated'.format(chemical.upper()))
    ax.set_ylabel('µg/m³')
    limit = 25 if chemical == 'pm2.5' else 50
    ax.axhline(y=limit, color='tab:purple', linestyle='dashed', alpha=.6,
               label='Limite di riferimento\n({}µg/m³, D.Lgs.155/2010)'.format(limit))

    x_dates = ['', '01/09/20', '01/10/20', '01/11/20', '01/12/20', '01/01/21', '01/02/21', '01/03/21', '01/04/21',
               '01/05/21',
               '01/06/21', '01/07/21', '01/08/21', '01/09/21']
    ax.locator_params(axis='x', nbins=len(x_dates))
    ax.tick_params(axis='x', which='major', labelsize=9)
    ax.set_xticklabels(x_dates)
    fig.autofmt_xdate()

    ax.legend()
    plt.savefig('../generated_data/new/predictions/' + name + '.svg')


if __name__ == '__main__':
    execute(Dataset.SMART16_NEW_PM_24H, 'pm2.5', 'pred_24h_pm2.5')
    execute(Dataset.SMART16_NEW_PM_24H, 'pm10', 'pred_24h_pm10')
