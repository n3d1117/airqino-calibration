import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from pandas.tseries.offsets import *


def execute(chemical):
    data = pd.read_csv('data/smart/{}.csv'.format('SMART16_new'))
    data.set_index('data', inplace=True)
    data.index = pd.to_datetime(data.index, utc=True)
    data = data[['pm2_5', 'pm10', 'tair', 'rad']]
    data = data.groupby(pd.Grouper(freq='1h')).mean().round(3)

    date_range = pd.date_range('2020-08-28', '2021-08-28', freq='W')
    a, b, c = [], [], []
    for month in date_range:
        month_start = month.strftime('%Y-%m-%d')
        month_end = (month + Week(1)).strftime('%Y-%m-%d')
        month_dataset = data.loc[month_start: month_end]
        a.append(month_dataset['tair'].corr(month_dataset[chemical]))
        b.append(month_dataset['rad'].corr(month_dataset[chemical]))
        c.append(month_end)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.set_title('SMART16 | Andamento della correlazione vs {}'.format(chemical.upper()))
    ax.plot(c, a, label='Temperatura')
    ax.plot(c, b, label='Umidità')
    ax.axhline(y=0, color='tab:purple', linestyle='dashed', alpha=.6)
    fig.autofmt_xdate()
    ax.legend()

    every_nth = 3
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

    plt.show()

    plt.title('SMART16 | Correlation Matrix | ')
    seaborn.heatmap(data.corr(), cmap='vlag', xticklabels=data.corr().columns,
                    yticklabels=data.corr().columns, annot=True)
    plt.show()


if __name__ == '__main__':
    execute('pm2_5')
    execute('pm10')
