import pandas as pd

def clean_car_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['year'].str.isnumeric()]
    df['year'] = df['year'].astype(int)

    df = df[df['Price'] != "Ask For Price"]
    df['Price'] = df['Price'].str.replace(',', '').astype(int)

    df['kms_driven'] = df['kms_driven'].str.split(' ').str.get(0)
    df = df[df['kms_driven'] != 'Petrol']
    df['kms_driven'] = df['kms_driven'].str.replace(',', '').astype(int)

    df.dropna(inplace=True)
    df['name'] = df['name'].str.split(' ').str[:3].str.join(' ')
    df = df[df['Price'] < 4e6]

    return df
