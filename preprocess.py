import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def smart_stations_make_hourly_avg(station, columns):
    df = pd.read_csv('data/smart/{}.csv'.format(station))
    df.set_index('data', inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)

    df = df[columns]
    grouped = df[columns].groupby(pd.Grouper(freq='60Min'))
    df = grouped.mean().round(3)
    df['count'] = grouped[columns[0]].count()
    df.to_csv('generated_data/smart/{}_resampled.csv'.format(station.lower()))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(df.index, df['count'])
    ax.set_title('{} - Number of data points every hour'.format(station))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
    ax.tick_params(axis='x', which='major', labelsize=9)
    fig.autofmt_xdate()
    plt.savefig('generated_data/smart/{}_count_bar.png'.format(station.lower()))


if __name__ == '__main__':
    smart_stations_make_hourly_avg(station='SMART16', columns=['no2', 'pm2_5', 'pm10'])
    smart_stations_make_hourly_avg(station='SMART24', columns=['no2'])
    smart_stations_make_hourly_avg(station='SMART25', columns=['no2', 'pm10'])
    smart_stations_make_hourly_avg(station='SMART26', columns=['no2', 'pm10'])
