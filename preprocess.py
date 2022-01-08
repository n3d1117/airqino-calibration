import pandas as pd


def smart_stations_resample(station, columns):
    df = pd.read_csv('data/smart/{}.csv'.format(station))

    if station == 'SMART16':
        fix_dst_smart16(df)
    elif station == 'SMART16_new':
        fix_dst_smart16_new(df)
    elif station == 'SMART24':
        fix_dst_smart24(df)
    elif station == 'SMART25':
        fix_dst_smart25(df)

    df.set_index('data', inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)

    # Resample every 60min
    df = df[columns]
    grouped = df[columns].groupby(pd.Grouper(freq='60Min'))
    df = grouped.mean().round(3)
    df['n_data_points'] = grouped[columns[0]].count()
    df.to_csv('generated_data/smart/{}_resampled.csv'.format(station.lower()))


def fix_dst_smart16_new(df):
    shift_back = [
        '2021-03-28 03:01:12',
        '2021-03-28 03:02:46',
        '2021-03-28 03:04:20',
        '2021-03-28 03:05:54',
        '2021-03-28 03:07:28',
        '2021-03-28 03:09:02',
        '2021-03-28 03:10:36',
        '2021-03-28 03:12:10',
        '2021-03-28 03:13:44',
        '2021-03-28 03:15:18',
        '2021-03-28 03:16:52',
        '2021-03-28 03:18:26',
        '2021-03-28 03:20:00',
        '2021-03-28 03:21:34',
        '2021-03-28 03:23:08',
        '2021-03-28 03:24:42',
        '2021-03-28 03:26:16',
        '2021-03-28 03:27:50',
        '2021-03-28 03:29:24',
        '2021-03-28 03:30:58',
        '2021-03-28 03:32:32',
        '2021-03-28 03:34:06',
        '2021-03-28 03:35:40',
        '2021-03-28 03:37:14',
        '2021-03-28 03:38:48',
        '2021-03-28 03:40:22',
        '2021-03-28 03:41:56',
        '2021-03-28 03:43:30',
        '2021-03-28 03:45:04',
        '2021-03-28 03:46:38',
        '2021-03-28 03:48:12',
        '2021-03-28 03:49:46',
        '2021-03-28 03:51:20',
        '2021-03-28 03:52:54',
        '2021-03-28 03:54:28',
        '2021-03-28 03:55:34',
        '2021-03-28 03:57:08',
        '2021-03-28 03:58:42',
        '2021-03-28 04:00:16',
        '2021-03-28 04:03:25',
        '2021-03-28 04:04:59'
    ]
    shift_back_dict = {v: v.replace(' 03:', ' 02:') for v in shift_back}
    df.replace({'data': shift_back_dict}, inplace=True)


def fix_dst_smart16(df):
    shift_back = [
        '2020-03-29 03:00:33',
        '2020-03-29 03:02:07',
        '2020-03-29 03:03:41',
        '2020-03-29 03:05:16',
        '2020-03-29 03:06:50',
        '2020-03-29 03:08:24',
        '2020-03-29 03:09:58',
        '2020-03-29 03:11:32',
        '2020-03-29 03:13:06',
        '2020-03-29 03:14:40',
        '2020-03-29 03:16:14',
        '2020-03-29 03:17:48',
        '2020-03-29 03:19:22',
        '2020-03-29 03:20:56',
        '2020-03-29 03:23:38',
        '2020-03-29 03:25:38',
        '2020-03-29 03:27:12',
        '2020-03-29 03:28:46',
        '2020-03-29 03:30:20',
        '2020-03-29 03:31:54',
        '2020-03-29 03:33:28',
        '2020-03-29 03:35:02',
        '2020-03-29 03:36:36',
        '2020-03-29 03:38:10',
        '2020-03-29 03:39:44',
        '2020-03-29 03:41:18',
        '2020-03-29 03:42:53',
        '2020-03-29 03:44:27',
        '2020-03-29 03:46:01',
        '2020-03-29 03:47:35',
        '2020-03-29 03:49:09',
        '2020-03-29 03:50:43',
        '2020-03-29 03:53:51',
        '2020-03-29 03:55:25',
        '2020-03-29 03:56:59',
        '2020-03-29 03:58:33'
    ]
    shift_back_dict = {v: v.replace(' 03:', ' 02:') for v in shift_back}
    df.replace({'data': shift_back_dict}, inplace=True)


def fix_dst_smart24(df):
    shift_back = [
        '2020-03-29 03:01:09',
        '2020-03-29 03:03:30',
        '2020-03-29 03:06:38',
        '2020-03-29 03:09:46',
        '2020-03-29 03:11:20',
        '2020-03-29 03:12:54',
        '2020-03-29 03:14:28',
        '2020-03-29 03:16:02',
        '2020-03-29 03:17:36',
        '2020-03-29 03:19:37',
        '2020-03-29 03:21:11',
        '2020-03-29 03:22:45',
        '2020-03-29 03:24:19',
        '2020-03-29 03:25:53',
        '2020-03-29 03:27:27',
        '2020-03-29 03:30:08',
        '2020-03-29 03:31:42',
        '2020-03-29 03:33:16',
        '2020-03-29 03:34:50',
        '2020-03-29 03:36:24',
        '2020-03-29 03:37:58',
        '2020-03-29 03:39:32',
        '2020-03-29 03:41:06',
        '2020-03-29 03:42:40',
        '2020-03-29 03:44:14',
        '2020-03-29 03:45:48',
        '2020-03-29 03:47:22',
        '2020-03-29 03:48:56',
        '2020-03-29 03:50:30',
        '2020-03-29 03:52:04',
        '2020-03-29 03:53:38',
        '2020-03-29 03:55:12',
        '2020-03-29 03:56:46',
        '2020-03-29 03:58:20',
        '2020-03-29 03:59:54'
    ]
    shift_back_dict = {v: v.replace(' 03:', ' 02:') for v in shift_back}
    df.replace({'data': shift_back_dict}, inplace=True)


def fix_dst_smart25(df):
    shift_back = [
        '2020-03-29 03:00:48',
        '2020-03-29 03:01:49',
        '2020-03-29 03:02:45',
        '2020-03-29 03:03:41',
        '2020-03-29 03:04:36',
        '2020-03-29 03:05:34',
        '2020-03-29 03:06:29',
        '2020-03-29 03:07:26',
        '2020-03-29 03:08:29',
        '2020-03-29 03:09:27',
        '2020-03-29 03:10:23',
        '2020-03-29 03:11:17',
        '2020-03-29 03:12:15',
        '2020-03-29 03:13:11',
        '2020-03-29 03:14:07',
        '2020-03-29 03:15:02',
        '2020-03-29 03:15:55',
        '2020-03-29 03:16:49',
        '2020-03-29 03:17:50',
        '2020-03-29 03:18:44',
        '2020-03-29 03:19:38',
        '2020-03-29 03:20:39',
        '2020-03-29 03:21:35',
        '2020-03-29 03:22:30',
        '2020-03-29 03:23:23',
        '2020-03-29 03:24:22',
        '2020-03-29 03:25:16',
        '2020-03-29 03:26:19',
        '2020-03-29 03:27:18',
        '2020-03-29 03:28:13',
        '2020-03-29 03:29:07',
        '2020-03-29 03:30:06',
        '2020-03-29 03:31:02',
        '2020-03-29 03:31:56',
        '2020-03-29 03:32:51',
        '2020-03-29 03:33:48',
        '2020-03-29 03:34:49',
        '2020-03-29 03:35:43',
        '2020-03-29 03:36:37',
        '2020-03-29 03:37:32',
        '2020-03-29 03:38:27',
        '2020-03-29 03:39:22',
        '2020-03-29 03:40:17',
        '2020-03-29 03:41:14',
        '2020-03-29 03:42:09',
        '2020-03-29 03:43:04',
        '2020-03-29 03:43:58',
        '2020-03-29 03:44:54',
        '2020-03-29 03:45:52',
        '2020-03-29 03:46:46',
        '2020-03-29 03:47:40',
        '2020-03-29 03:48:35',
        '2020-03-29 03:49:30',
        '2020-03-29 03:50:25',
        '2020-03-29 03:51:25',
        '2020-03-29 03:52:22',
        '2020-03-29 03:53:23',
        '2020-03-29 03:54:18',
        '2020-03-29 03:55:19',
        '2020-03-29 03:56:14',
        '2020-03-29 03:57:09',
        '2020-03-29 03:58:04',
        '2020-03-29 03:58:04',
        '2020-03-29 03:58:59',
        '2020-03-29 03:59:53'
    ]
    shift_back_dict = {v: v.replace(' 03:', ' 02:') for v in shift_back}
    df.replace({'data': shift_back_dict}, inplace=True)


def arpat_clean_no2_dataset(name):
    csv = 'data/arpat_lucca/{}.csv'.format(name)
    df = pd.read_csv(csv, encoding='iso-8859-1', delimiter=';')

    # Rename columns
    df.columns = ['station', 'param', 'aaammgg', 'hour', 'avg', 'valid']

    # Convert data
    df['data'] = pd.to_datetime(df.aaammgg, format='%Y%m%d') + pd.to_timedelta(df.hour, unit='h')
    df.drop(columns=['aaammgg', 'hour', 'valid', 'station', 'param'], inplace=True)

    # Localize date and convert to UTC
    df.set_index('data', inplace=True)
    df.index = df.index.tz_localize('Europe/Rome', ambiguous='NaT', nonexistent='NaT').tz_convert('utc')

    # Drop NaN/NaT
    df.dropna(inplace=True)

    # Save
    df.to_csv('generated_data/arpat/{}_cleaned.csv'.format(name.lower()))


def arpat_clean_pm_dataset(name):
    csv = 'data/arpat_lucca/{}.csv'.format(name)
    df = pd.read_csv(csv)

    # Convert data
    df['data'] = pd.to_datetime(df.DATA, format='%d/%m/%Y') + pd.to_timedelta(df.ORA, unit='h')
    df.drop(columns=['DATA', 'ORA', 'NO_LU-CAPANNORI', 'NOX_LU-CAPANNORI'], inplace=True)

    # Localize date and convert to UTC
    df.set_index('data', inplace=True)
    df.index = df.index.tz_localize('Europe/Rome', ambiguous='NaT', nonexistent='NaT').tz_convert('utc')

    # Drop NaN/NaT
    df.dropna(inplace=True)

    # Rename columns
    df.columns = ['pm10', 'pm2.5']

    # Save
    df.to_csv('generated_data/arpat/{}_cleaned.csv'.format(name.lower()))


if __name__ == '__main__':
    smart_stations_resample(station='SMART16', columns=['no2', 'pm2_5', 'pm10'])
    # smart_stations_resample(station='SMART24', columns=['no2'])
    # smart_stations_resample(station='SMART25', columns=['no2', 'pm10'])
    # smart_stations_resample(station='SMART26', columns=['no2', 'pm10'])
    smart_stations_resample(station='SMART16_new', columns=['no2', 'pm2_5', 'pm10'])

    arpat_clean_no2_dataset(name='LU-CAPANNORI_NO2_2020')
    # arpat_clean_no2_dataset(name='LU-MICHELETTO_NO2_2020')
    # arpat_clean_no2_dataset(name='LU-SAN-CONCORDIO_NO2_2020')

    arpat_clean_pm_dataset(name='LU-CAPANNORI_PM_Dati_Orari')
