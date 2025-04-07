import pandas as pd
import numpy as np
from scipy import stats

def extract_data():
    industry_df = pd.read_csv("industry_emission.csv")
    socio_df = pd.read_excel("owid-covid-data.csv")
    with open("data/traffic_api_sample.json", "r") as file:
        traffic_json = file.read()
    traffic_df = pd.json_normalize(eval(traffic_json))  # Simulated API response
    return industry_df, socio_df, traffic_df

def clean_and_transform(industry_df, socio_df, traffic_df):
    # Convert dates
    for df in [industry_df, socio_df, traffic_df]:
        df['date'] = pd.to_datetime(df['date'])
    # Handle missing values
    industry_df = industry_df.interpolate(method='linear')
    traffic_df.fillna(traffic_df.mean(numeric_only=True), inplace=True)
    # Standardize emission units
    def convert_to_kg(row):
        if row.get('unit') == 'tons':
            return row['emission'] * 1000
        return row['emission']
    industry_df['emission'] = industry_df.apply(convert_to_kg, axis=1)
    # Z-score to detect and remove outliers
    industry_df['z_score'] = stats.zscore(industry_df['emission'])
    industry_df = industry_df[industry_df['z_score'].abs() < 3].drop(columns=['z_score'])
    # Merge datasets
    merged_df = industry_df.merge(traffic_df, on="date", how="left")
    merged_df = merged_df.merge(socio_df, on="date", how="left")

    return merged_df

def write_to_csv(df, file_name='clean_emission_data.csv'):
    # Write DataFrame to CSV
    df.to_csv(file_name, index=False)
    print(f"Data saved to: {file_name}")

if __name__ == "__main__":
    industry_df, socio_df, traffic_df = extract_data()
    clean_df = clean_and_transform(industry_df, socio_df, traffic_df)
    write_to_csv(clean_df)
