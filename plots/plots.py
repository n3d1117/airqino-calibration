import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.tseries.offsets import MonthEnd

from regression.regression_models import get_polynomial_model
from utils.utils import get_dataset, Dataset


def scatterplot(X, y, u, title, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X, y, color='tab:blue', marker='.')
    ax.set_xlabel('AirQino ({})'.format(u))
    ax.set_ylabel('ARPAT (µg/m³)')
    ax.set_title(title)

    if len(X) > 0:
        line_x = np.arange(X.min(), X.max())[:, np.newaxis]
        poly = get_polynomial_model()
        poly.fit(X.reshape((-1, 1)), y)
        ax.plot(line_x, poly.predict(line_x), color='tab:red', linewidth=2,
                label='Poly regressor\n(d=2, R²=%0.3f)' % poly.score(X.reshape((-1, 1)), y), alpha=.3)
        plt.legend(loc='best')

    plt.savefig('generated_data/plots/scatterplot/{}.png'.format(filename))


def plot(df, column, ylabel, title, filename, color=(0.2, 0.4, 0.6, 0.6), month_freq=1):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(df.index, df[column], color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_freq))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
    ax.tick_params(axis='x', which='major', labelsize=9)
    fig.autofmt_xdate()
    plt.savefig('generated_data/plots/{}.png'.format(filename.lower()), dpi=300)


def plot_compare(ds1, ds2, label1, label2, title, filename):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    l1 = ax.plot(ds1, color='tab:red', label=label1)
    ax.tick_params(axis='y', labelcolor='tab:red')
    ax2 = ax.twinx()
    l2 = ax2.plot(ds2, color='tab:blue', label=label2)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc='upper right')
    plt.title(title)
    plt.savefig('generated_data/plots/compare/{}.png'.format(filename))


def get_smart_dataset(station):
    df = pd.read_csv('generated_data/smart/{}_resampled.csv'.format(station))
    df.set_index('data', inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def get_arpat_dataset(name):
    df = pd.read_csv('generated_data/arpat/{}.csv'.format(name))
    df.set_index('data', inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def plot_avg_count():
    # Data points
    for station in ['smart16', 'smart16_new', 'smart16_no2_new']:
        df = get_smart_dataset(station)
        plot(df=df, column='n_data_points', ylabel='# data points / h',
             title='{} -  Number of data points / hour'.format(station.upper()),
             filename='smart/{}_count'.format(station))

    # SMART NO2
    for station in ['smart16', 'smart16_no2_new']:
        df = get_smart_dataset(station)
        plot(df=df, column='no2', ylabel='counts', title='{} - NO2'.format(station.upper()),
             filename='smart/{}_no2'.format(station))

    # SMART PM2.5
    for station in ['smart16_new']:
        df = get_smart_dataset(station)
        plot(df=df, column='pm2_5', ylabel='µg/m³', title='{} - PM2.5'.format(station.upper()),
             filename='smart/{}_pm2.5'.format(station))

    # SMART PM10
    for station in ['smart16_new']:
        df = get_smart_dataset(station)
        plot(df=df, column='pm10', ylabel='µg/m³', title='{} - PM10'.format(station.upper()),
             filename='smart/{}_pm10'.format(station))

    # ARPAT NO2
    for name in ['lu-capannori_no2_2020_cleaned', 'lu-capannori_no2_new_cleaned']:
        df = get_arpat_dataset(name)
        plot(df=df, column='avg', ylabel='µg/m³', title='ARPAT - NO2'.format(name),
             filename='arpat/{}_no2'.format(name), color='#fe7b7c')

    # ARPAT PM
    for name in ['lu-capannori_pm_dati_orari_cleaned', 'lu-capannori_pm_dati_orari_cleaned_resampled_8h']:
        df = get_arpat_dataset(name)
        df = df.loc['2020-08-01':'2021-09-01']
        print(df)
        plot(df=df, column='pm2.5', ylabel='µg/m³', title='ARPAT - PM2.5'.format(name),
             filename='arpat/{}_pm2.5'.format(name), month_freq=1, color='#fe7b7c')
        plot(df=df, column='pm10', ylabel='µg/m³', title='ARPAT - PM10'.format(name),
             filename='arpat/{}_pm10'.format(name), month_freq=1, color='#fe7b7c')


def scatterplots():
    for dataset in [Dataset.SMART16_NO2, Dataset.SMART16_NO2_NEW]:
        df = get_dataset(dataset)
        X = df['airqino_no2'].values
        y = df['arpat_no2'].values
        scatterplot(X, y, u='counts', title='{} | NO2 ({})'.format(dataset.name, str(len(X))),
                    filename='{}'.format(dataset.name.lower()))
        date_range = pd.date_range('2020-01-01', '2020-12-31', freq='MS') \
            if dataset == Dataset.SMART16_NO2 else pd.date_range('2021-10-01', '2022-01-17', freq='MS')
        for month in date_range:
            month_str = month.strftime('%B').lower()
            month_start = month.strftime('%Y-%m-%d')
            month_end = (month + MonthEnd(1)).strftime('%Y-%m-%d')
            month_dataset = df.loc[month_start: month_end]
            X = month_dataset['airqino_no2'].values
            y = month_dataset['arpat_no2'].values
            scatterplot(X, y, u='counts', title='{} | NO2 | {} ({})'.format(dataset.name, month_str, str(len(X))),
                        filename='{}_{}'.format(dataset.name.lower(), month_str))

    for dataset in [Dataset.SMART16_NEW_PM_8H]:
        df = get_dataset(dataset)
        X = df['airqino_pm2.5'].values
        y = df['arpat_pm2.5'].values
        scatterplot(X, y, u='µg/m³', title='{} | PM2.5 ({})'.format(dataset.name, str(len(X))),
                    filename='{}2.5'.format(dataset.name.lower()))

        X = df['airqino_pm10'].values
        y = df['arpat_pm10'].values
        scatterplot(X, y, u='µg/m³', title='{} | PM10 ({})'.format(dataset.name, str(len(X))),
                    filename='{}10'.format(dataset.name.lower()))

        for month in pd.date_range('2020-09-01', '2021-08-31', freq='MS'):
            month_str = month.strftime('%B').lower()
            month_start = month.strftime('%Y-%m-%d')
            month_end = (month + MonthEnd(1)).strftime('%Y-%m-%d')
            month_dataset = df.loc[month_start: month_end]
            X = month_dataset['airqino_pm2.5'].values
            y = month_dataset['arpat_pm2.5'].values
            scatterplot(X, y, u='µg/m³',
                        title='{} | PM2.5 | {} ({})'.format(dataset.name, month_str, str(len(month_dataset.index))),
                        filename='{}2.5_{}'.format(dataset.name.lower(), month_str))

            X = month_dataset['airqino_pm10'].values
            y = month_dataset['arpat_pm10'].values
            scatterplot(X, y, u='µg/m³',
                        title='{} | PM10 | {} ({})'.format(dataset.name, month_str, str(len(month_dataset.index))),
                        filename='{}10_{}'.format(dataset.name.lower(), month_str))


def plot_compares():
    for dataset in [Dataset.SMART16_NO2, Dataset.SMART16_NO2_NEW]:
        df = get_dataset(dataset)
        plot_compare(df['airqino_no2'], df['arpat_no2'], label1='AirQino', label2='ARPAT',
                     title='{} | NO2'.format(dataset.name), filename='{}_no2'.format(dataset.name.lower()))
        date_range = pd.date_range('2020-01-01', '2020-12-31', freq='MS') \
            if dataset == Dataset.SMART16_NO2 else pd.date_range('2021-10-01', '2022-01-17', freq='MS')

        for month in date_range:
            month_str = month.strftime('%b')
            month_start = month.strftime('%Y-%m-%d')
            month_end = (month + MonthEnd(1)).strftime('%Y-%m-%d')
            month_dataset = df.loc[month_start: month_end]
            plot_compare(month_dataset['airqino_no2'], month_dataset['arpat_no2'], label1='AirQino', label2='ARPAT',
                         title='{d} | NO2 | {m}'.format(d=dataset.name, m=month_str),
                         filename='{d}_no2_{m}'.format(d=dataset.name.lower(), m=month_str.lower()))

    for dataset in [Dataset.SMART16_NEW_PM]:
        df = get_dataset(dataset)
        plot_compare(df['airqino_pm2.5'], df['arpat_pm2.5'], label1='AirQino', label2='ARPAT',
                     title='{} | PM2.5'.format(dataset.name), filename='{}2.5'.format(dataset.name.lower()))
        plot_compare(df['airqino_pm10'], df['arpat_pm2.5'], label1='AirQino', label2='ARPAT',
                     title='{} | PM10'.format(dataset.name), filename='{}10'.format(dataset.name.lower()))
        for month in pd.date_range('2020-09-01', '2021-08-31', freq='MS'):
            month_str = month.strftime('%b')
            month_start = month.strftime('%Y-%m-%d')
            month_end = (month + MonthEnd(1)).strftime('%Y-%m-%d')
            month_dataset = df.loc[month_start: month_end]
            plot_compare(month_dataset['airqino_pm2.5'], month_dataset['arpat_pm2.5'], label1='AirQino',
                         label2='ARPAT', title='{d} | PM2.5 | {m}'.format(d=dataset.name, m=month_str),
                         filename='{d}_pm2.5_{m}'.format(d=dataset.name.lower(), m=month_str.lower()))
            plot_compare(month_dataset['airqino_pm10'], month_dataset['arpat_pm10'], label1='AirQino', label2='ARPAT',
                         title='{d} | PM10 | {m}'.format(d=dataset.name, m=month_str),
                         filename='{d}_pm10_{m}'.format(d=dataset.name.lower(), m=month_str.lower()))


if __name__ == '__main__':
    plot_avg_count()
    scatterplots()
    plot_compares()
