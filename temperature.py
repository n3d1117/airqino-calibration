import matplotlib.dates as mdates
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    df = pd.read_csv('data/smart/{}.csv'.format('SMART16_new'))
    df.set_index('data', inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)

    # Resample every 24h
    df = df[['pm2_5', 'pm10', 'tair']]
    grouped = df.groupby(pd.Grouper(freq='24h'))
    df_resampled = grouped.mean().round(3)

    arpat = pd.read_csv('generated_data/arpat/lu-{}.csv'.format('capannori_pm_dati_orari_cleaned_resampled_24h'))
    arpat.set_index('data', inplace=True)
    arpat.index = pd.to_datetime(arpat.index, utc=True)
    arpat.rename(columns={'pm2.5': 'arpat_pm2.5', 'pm10': 'arpat_pm10'}, inplace=True)
    arpat_pm = arpat[['arpat_pm2.5', 'arpat_pm10']]


    def merge_datasets(ds1, ds2):
        return pd.merge(ds1, ds2, on='data', how='inner').dropna()


    merged = merge_datasets(df_resampled, arpat_pm)

    df_autumn = merged[merged.index.map(lambda t: t.month in [9, 10, 11] and t.year == 2020)]
    df_winter = merged[
        merged.index.map(lambda t: (t.month == 12 and t.year == 2020) or (t.month in [1, 2] and t.year == 2021))]
    df_spring = merged[merged.index.map(lambda t: t.month in [3, 4, 5] and t.year == 2021)]
    df_summer = merged[merged.index.map(lambda t: t.month in [6, 7, 8] and t.year == 2021)]
    labels = ['Autumn', 'Winter', 'Spring', 'Summer']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, season in enumerate([df_autumn, df_winter, df_spring, df_summer]):
        ax.scatter(season['tair'], season['pm2_5'], marker='.', label=labels[i])
        linear_regressor = LinearRegression()
        X = season['tair'].values.reshape(-1, 1)
        Y = season['pm2_5'].values.reshape(-1, 1)
        linear_regressor.fit(X, Y)
        Y_pred = linear_regressor.predict(X)
        plt.plot(X, Y_pred)
    ax.legend(loc='upper right')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('PM2.5 (µg/m³)')
    ax.set_title('SMART16 | TAIR vs PM2.5')
    plt.savefig('generated_data/plots/tair/scatterplot.png')


    def compare_(df, start, end, period):
        fig = plt.figure()
        month_dataset = df.loc[start: end]
        ax = fig.add_subplot(111)
        l1 = ax.plot(month_dataset['tair'], color='tab:red', label='temp (°C)', alpha=.7)
        ax.tick_params(axis='y', labelcolor='tab:red')
        ax2 = ax.twinx()
        l2 = ax2.plot(month_dataset['pm2_5'], color='tab:blue', label='pm2.5 (µg/m³)', alpha=.7)
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc='best')
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y %H:00'))
        fig.autofmt_xdate()
        plt.title('SMART16 | TAIR vs PM2.5 | {}'.format(period))
        plt.savefig('generated_data/plots/tair/tair_pm2.5_compare_{}.png'.format(period.lower()))


    df = df[['pm2_5', 'pm10', 'tair']]
    grouped = df.groupby(pd.Grouper(freq='60Min'))
    df = grouped.mean().round(3)
    compare_(df, '2020-11-19', '2020-11-24', 'November')
    compare_(df, '2021-02-19', '2021-02-24', 'February')
    compare_(df, '2021-05-19', '2021-05-24', 'May')
    compare_(df, '2021-08-19', '2021-08-24', 'August')
