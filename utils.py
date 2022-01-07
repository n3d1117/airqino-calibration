from enum import Enum

import pandas as pd


class Dataset(Enum):
    SMART16_NO2 = 'smart16-capannori-no2'

    SMART16_NEW_PM = 'smart16_new-capannori-pm'

    SMART24 = 'smart24-micheletto-no2'
    SMART25 = 'smart25-san-concordio-no2'
    SMART26 = 'smart26-san-concordio-no2'


def get_dataset(dataset: Dataset):
    df = pd.read_csv('generated_data/merged/{}.csv'.format(dataset.value))
    df.set_index('data', inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df
