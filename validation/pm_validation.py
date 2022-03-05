import matplotlib.dates as mdates
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

from regression.regression_models import get_linear_model
from utils.utils import get_dataset, Dataset


def execute(dataset, validation_dataset, chemical, name):
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

    # Test against validation dataset
    val_dataset = get_dataset(validation_dataset)
    X_val = val_dataset['airqino_{}'.format(chemical)].values.reshape((-1, 1))
    y_val = val_dataset['arpat_{}'.format(chemical)].values
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    limit = 25 if chemical == 'pm2.5' else 50

    # Plot
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    ax.plot(val_dataset.index, y_val, color='tab:blue', label='ARPAT (Reference)', alpha=.8, linewidth=1.2)
    ax.plot(val_dataset.index, X_val, color='tab:gray', label='AirQino (Raw)', alpha=.4, linewidth=1)
    ax.plot(val_dataset.index, y_pred, color='tab:red', label='AirQino (Calibrated)', alpha=.6, linewidth=1)
    ax.axhline(y=limit, color='tab:purple', linestyle='dashed', alpha=.6,
               label='Limite di riferimento\n({}µg/m³, D.Lgs.155/2010)'.format(limit))
    c = 'PM₂.₅' if chemical == 'pm2.5' else 'PM₁₀'
    ax.set_title(
        'SMART16 Validation | {} | Reference vs Raw vs Calibrated (R² = {})'.format(c, round(r2, 2)))
    ax.set_ylabel('µg/m³')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
    ax.tick_params(axis='x', which='major', labelsize=9)
    fig.autofmt_xdate()

    ax.legend()
    plt.savefig('../generated_data/new/validation/' + name + '.svg')


if __name__ == '__main__':
    execute(Dataset.SMART16_NEW_PM_12H, Dataset.SMART16_VAL_PM_12H, 'pm2.5', 'val_pm2.5')
    execute(Dataset.SMART16_NEW_PM_12H, Dataset.SMART16_VAL_PM_12H, 'pm10', 'val_pm10')
