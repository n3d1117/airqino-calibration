import matplotlib.dates as mdates
import pandas as pd
from matplotlib import pyplot as plt


def plot(df, column, ylabel, title, filename, month_freq=1):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(df.index, df[column])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_freq))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
    ax.tick_params(axis='x', which='major', labelsize=9)
    fig.autofmt_xdate()
    plt.savefig('generated_data/plots/{}.png'.format(filename.lower()))


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


if __name__ == '__main__':

    # SMART NO2 + Data Points
    for station in ['smart16', 'smart24', 'smart25', 'smart26']:
        df = get_smart_dataset(station)
        plot(df=df, column='no2', ylabel='counts', title='{} - NO2'.format(station.upper()),
             filename='smart/{}_no2'.format(station))
        plot(df=df, column='n_data_points', ylabel='# data points', title='Data points',
             filename='smart/{}_count'.format(station))

    # SMART PM2.5
    for station in ['smart16']:
        df = get_smart_dataset(station)
        plot(df=df, column='pm2_5', ylabel='counts', title='{} - PM2.5'.format(station.upper()),
             filename='smart/{}_pm2.5'.format(station))

    # SMART PM10
    for station in ['smart16']:
        df = get_smart_dataset(station)
        plot(df=df, column='pm10', ylabel='counts', title='{} - PM10'.format(station.upper()),
             filename='smart/{}_pm10'.format(station))

    # ARPAT NO2
    for name in ['lu-capannori_no2_2020_cleaned', 'lu-micheletto_no2_2020_cleaned',
                 'lu-san-concordio_no2_2020_cleaned']:
        df = get_arpat_dataset(name)
        plot(df=df, column='avg', ylabel='µg/m³', title='{} - NO2'.format(name), filename='arpat/{}_no2'.format(name))

    # ARPAT PM
    for name in ['lu-capannori_pm_dati_orari_cleaned']:
        df = get_arpat_dataset(name)
        plot(df=df, column='pm2.5', ylabel='µg/m³', title='{} - PM2.5'.format(name),
             filename='arpat/{}_pm2.5'.format(name), month_freq=6)
        plot(df=df, column='pm10', ylabel='µg/m³', title='{} - PM10'.format(name),
             filename='arpat/{}_pm10'.format(name), month_freq=6)
