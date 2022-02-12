import matplotlib.dates as mdates
import pandas as pd
import seaborn
from matplotlib import pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('../data/smart/{}.csv'.format('SMART16_new'))
    df.set_index('data', inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.groupby(pd.Grouper(freq='8h')).mean().dropna()

    df = df.loc['2021-05-01': '2021-09-01']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(df.index, df['tair'], color='red')
    ax.plot(df.index, df['rad'], color='blue')
    ax.plot(df.index, df['pm10'], color='green')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
    ax.tick_params(axis='x', which='major', labelsize=9)
    fig.autofmt_xdate()
    plt.show()

    plt.title('SMART16 | Correlation Matrix | ')
    df = df[['tair', 'rad', 'pm10']]
    seaborn.heatmap(df.corr(), cmap='vlag', xticklabels=df.corr().columns,
                    yticklabels=df.corr().columns, annot=True)
    plt.show()
