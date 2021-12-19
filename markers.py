import json

import pandas as pd


def save_coordinates():
    data = {'stations': []}

    for name in ['SMART16', 'SMART24', 'SMART25', 'SMART26']:
        df = pd.read_csv('data/smart/{}.csv'.format(name))
        unique_coords_df = df[['lat', 'long']].drop_duplicates()

        coords = []
        for row in unique_coords_df.itertuples():
            coords.append({
                'lat': row.lat,
                'long': row.long
            })
        data['stations'].append({'name': name, 'coords': coords})

    with open('map/locations.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    save_coordinates()
