from enum import Enum

import pandas as pd


class Dataset(Enum):
    SMART16_NO2 = 'smart16-capannori-no2'
    SMART16_NO2_NEW = 'smart16_no2_new-capannori_no2_new-no2'

    SMART16_NEW_PM = 'smart16_new-capannori-pm'
    SMART16_NEW_PM_24H = 'smart16_new_resampled_24h-capannori-pm'
    SMART16_NEW_PM_8H = 'smart16_new_resampled_8h-capannori-pm'
    SMART16_NEW_PM_12H = 'smart16_new_resampled_12h-capannori-pm'

    SMART16_VAL_PM = 'smart16_val-capannori-pm'
    SMART16_VAL_PM_8H = 'smart16_val_resampled_8h-capannori-pm'
    SMART16_VAL_PM_12H = 'smart16_val_resampled_12h-capannori-pm'

    SMART24 = 'smart24-micheletto-no2'
    SMART25 = 'smart25-san-concordio-no2'
    SMART26 = 'smart26-san-concordio-no2'


def get_dataset(dataset: Dataset):
    df = pd.read_csv('../generated_data/merged/{}.csv'.format(dataset.value))
    df.set_index('data', inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df
