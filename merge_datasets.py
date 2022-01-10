import pandas as pd


def merge_datasets(ds1, ds2):
    return pd.merge(ds1, ds2, on='data', how='inner').dropna()


def merge_and_save_no2(df1, df2):
    smart = pd.read_csv('generated_data/smart/{}_resampled.csv'.format(df1))
    smart.set_index('data', inplace=True)
    smart.index = pd.to_datetime(smart.index, utc=True)
    smart.rename(columns={'no2': 'airqino_no2'}, inplace=True)
    smart_no2 = smart[['airqino_no2']]

    arpat = pd.read_csv('generated_data/arpat/lu-{}_no2_2020_cleaned.csv'.format(df2))
    arpat.set_index('data', inplace=True)
    arpat.index = pd.to_datetime(arpat.index, utc=True)
    arpat.rename(columns={'avg': 'arpat_no2'}, inplace=True)
    arpat_no2 = arpat[['arpat_no2']]

    merged = merge_datasets(smart_no2, arpat_no2)
    merged.to_csv('generated_data/merged/{df1}-{df2}-no2.csv'.format(df1=df1, df2=df2))


def merge_and_save_pm(df1, df2, filename):
    smart = pd.read_csv('generated_data/smart/{}.csv'.format(df1))
    smart.set_index('data', inplace=True)
    smart.index = pd.to_datetime(smart.index, utc=True)
    smart.rename(columns={'pm2_5': 'airqino_pm2.5', 'pm10': 'airqino_pm10'}, inplace=True)
    smart_pm = smart[['airqino_pm2.5', 'airqino_pm10']]

    arpat = pd.read_csv('generated_data/arpat/lu-{}.csv'.format(df2))
    arpat.set_index('data', inplace=True)
    arpat.index = pd.to_datetime(arpat.index, utc=True)
    arpat.rename(columns={'pm2.5': 'arpat_pm2.5', 'pm10': 'arpat_pm10'}, inplace=True)
    arpat_pm = arpat[['arpat_pm2.5', 'arpat_pm10']]

    merged = merge_datasets(smart_pm, arpat_pm)
    merged.to_csv('generated_data/merged/{}-pm.csv'.format(filename))


if __name__ == '__main__':
    merge_and_save_no2(df1='smart16', df2='capannori')
    # merge_and_save_no2(df1='smart24', df2='micheletto')
    # merge_and_save_no2(df1='smart25', df2='san-concordio')
    # merge_and_save_no2(df1='smart26', df2='san-concordio')

    # merge_and_save_pm(df1='smart16', df2='capannori')
    merge_and_save_pm(df1='smart16_new_resampled', df2='capannori_pm_dati_orari_cleaned',
                      filename='smart16_new-capannori')
    merge_and_save_pm(df1='smart16_new_resampled_24h', df2='capannori_pm_dati_orari_cleaned_resampled_24h',
                      filename='smart16_new_resampled_24h-capannori')
