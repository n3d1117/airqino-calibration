import json

import pandas as pd


def save_coordinates():
    data = {'stations': []}

    for name in ['SMART16', 'SMART24', 'SMART25', 'SMART26']:

        # Read CSV
        df = pd.read_csv('data/smart/{}.csv'.format(name))

        # Isolate lat long and data columns
        df = df[['lat', 'long', 'data']]

        # Add 'data_from' and 'data_to' columns on first/last occurrency of grouping by coord
        df['data_from'] = df.groupby(['lat', 'long'])['data'].transform('first')
        df['data_to'] = df.groupby(['lat', 'long'])['data'].transform('last')

        # Remove 'data' column and drop all lat/long duplicates
        df = df.drop(columns=['data'])
        df = df.drop_duplicates(subset=['lat', 'long'])

        # Add coordinates to JSON
        coords = []
        for row in df.itertuples():
            coords.append({
                'lat': row.lat,
                'long': row.long,
                'from': row.data_from,
                'to': row.data_to
            })
        data['stations'].append({'name': name, 'coords': coords})

    with open('map/locations.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    save_coordinates()
