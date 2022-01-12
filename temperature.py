import matplotlib.dates as mdates
import pandas as pd
from matplotlib import pyplot as plt
from pandas.tseries.offsets import MonthEnd
from sklearn.linear_model import LinearRegression


def plot_tair():
    df = pd.read_csv('data/smart/{}.csv'.format('SMART16_new'))
    df.set_index('data', inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)

    def do_plot(dataset, locator, title, filename):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(dataset.index, dataset['tair'])
        ax.set_title(title)
        ax.set_ylabel('Degrees (°C)')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
        ax.tick_params(axis='x', which='major', labelsize=9)
        fig.autofmt_xdate()
        plt.savefig('generated_data/plots/tair/{}.png'.format(filename))

    do_plot(df.groupby(pd.Grouper(freq='3h')).mean().round(3), mdates.MonthLocator(interval=1),
            'SMART16_new | Temperature | All year', filename='tair')

    date_range = pd.date_range('2020-09-01', '2021-08-31', freq='MS')
    for month in date_range:
        month_str = month.strftime('%B')
        month_start = month.strftime('%Y-%m-%d')
        month_end = (month + MonthEnd(1)).strftime('%Y-%m-%d')

        month_dataset = df.loc[month_start: month_end]
        do_plot(month_dataset, mdates.DayLocator(interval=3), 'SMART16_new | Temperature | {}'.format(month_str),
                filename='tair_{}'.format(month_str.lower()))


def plot_tair_scatterplot():
    df = pd.read_csv('data/smart/{}.csv'.format('SMART16_new'))
    df.set_index('data', inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[['pm2_5', 'pm10', 'tair']]
    df = df.groupby(pd.Grouper(freq='3h')).mean().round(3).dropna()

    def seasons():
        df_autumn = df[df.index.map(lambda t: t.month in [9, 10, 11] and t.year == 2020)]
        df_winter = df[
            df.index.map(lambda t: (t.month == 12 and t.year == 2020) or (t.month in [1, 2] and t.year == 2021))
        ]
        df_spring = df[df.index.map(lambda t: t.month in [3, 4, 5] and t.year == 2021)]
        df_summer = df[df.index.map(lambda t: t.month in [6, 7, 8] and t.year == 2021)]
        return df_autumn, df_winter, df_spring, df_summer

    labels = ['Autumn', 'Winter', 'Spring', 'Summer']

    # Scatterplot by season
    for i, season in enumerate(seasons()):
        # PM2.5
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('PM2.5 (µg/m³)')
        ax.set_title('SMART16_new_3h | TAIR vs PM2.5 | {}'.format(labels[i]))
        X, y = season['tair'], season['pm2_5']
        ax.scatter(X, y, marker='.')
        linear_regressor = LinearRegression()
        linear_regressor.fit(X.values.reshape((-1, 1)), y.values)
        Y_pred = linear_regressor.predict(X.values.reshape((-1, 1)))
        ax.plot(X, Y_pred, color='tab:red', linewidth=2, alpha=.4)
        plt.savefig('generated_data/plots/tair/scatterplot_{}_pm2.5.png'.format(labels[i].lower()))

        # PM10
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('PM10 (µg/m³)')
        ax.set_title('SMART16_new_3h | TAIR vs PM10 | {}'.format(labels[i]))
        X, y = season['tair'], season['pm10']
        ax.scatter(X, y, marker='.')
        linear_regressor = LinearRegression()
        linear_regressor.fit(X.values.reshape((-1, 1)), y.values)
        Y_pred = linear_regressor.predict(X.values.reshape((-1, 1)))
        ax.plot(X, Y_pred, color='tab:red', linewidth=2, alpha=.4)
        plt.savefig('generated_data/plots/tair/scatterplot_{}_pm10.png'.format(labels[i].lower()))

    # Scatterplot all year PM2.5
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, season in enumerate(seasons()):
        ax.scatter(season['tair'], season['pm2_5'], marker='.', label=labels[i], alpha=.5)
    ax.legend(loc='upper right')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('PM2.5 (µg/m³)')
    ax.set_title('SMART16_new_3h | TAIR vs PM2.5')
    plt.savefig('generated_data/plots/tair/scatterplot_pm2.5.png')

    # Scatterplot all year PM10
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, season in enumerate(seasons()):
        ax.scatter(season['tair'], season['pm10'], marker='.', label=labels[i], alpha=.5)
    ax.legend(loc='upper right')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('PM10 (µg/m³)')
    ax.set_title('SMART16_new_3h | TAIR vs PM10')
    plt.savefig('generated_data/plots/tair/scatterplot_pm10.png')


def compare():
    df = pd.read_csv('data/smart/{}.csv'.format('SMART16_new'))
    df.set_index('data', inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)

    def compare_(df, start, end, period):
        month_dataset = df.loc[start: end]

        # PM2.5
        fig = plt.figure()
        ax = fig.add_subplot(111)
        l1 = ax.plot(month_dataset['tair'], color='tab:red', label='Temp (°C)', alpha=.7)
        ax.tick_params(axis='y', labelcolor='tab:red')
        ax2 = ax.twinx()
        l2 = ax2.plot(month_dataset['pm2_5'], color='tab:blue', label='PM2.5 (µg/m³)', alpha=.7)
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc='best')
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y %H:00'))
        fig.autofmt_xdate()
        plt.title('SMART16_new | TAIR vs PM2.5 | {}'.format(period))
        plt.savefig('generated_data/plots/tair/tair_pm2.5_compare_{}.png'.format(period.lower()))

        # PM10
        fig = plt.figure()
        ax = fig.add_subplot(111)
        l1 = ax.plot(month_dataset['tair'], color='tab:red', label='Temp (°C)', alpha=.7)
        ax.tick_params(axis='y', labelcolor='tab:red')
        ax2 = ax.twinx()
        l2 = ax2.plot(month_dataset['pm10'], color='tab:blue', label='PM10 (µg/m³)', alpha=.7)
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc='best')
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y %H:00'))
        fig.autofmt_xdate()
        plt.title('SMART16_new | TAIR vs PM10 | {}'.format(period))
        plt.savefig('generated_data/plots/tair/tair_pm10_compare_{}.png'.format(period.lower()))

    df = df[['pm2_5', 'pm10', 'tair']]
    grouped = df.groupby(pd.Grouper(freq='60Min'))
    df = grouped.mean().round(3)
    compare_(df, '2020-11-19', '2020-11-24', 'November')
    compare_(df, '2021-02-19', '2021-02-24', 'February')
    compare_(df, '2021-05-19', '2021-05-24', 'May')
    compare_(df, '2021-08-19', '2021-08-24', 'August')


if __name__ == '__main__':
    plot_tair_scatterplot()
    compare()
    plot_tair()
