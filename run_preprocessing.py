import pandas as pd
import numpy as np

print('Loading Indian flight data...')
df = pd.read_csv('data/flights.csv', low_memory=False)
print(f'Loaded {len(df):,} rows')

df = df[df['CANCELLED'] == 0]
df = df[df['DIVERTED'] == 0]
df = df.dropna(subset=['DEPARTURE_DELAY', 'ARRIVAL_DELAY'])
for col in ['WEATHER_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','ATC_DELAY','TECHNICAL_DELAY']:
    if col in df.columns:
        df[col] = df[col].fillna(0)
df['DEPARTURE_DELAY'] = df['DEPARTURE_DELAY'].clip(-30, 300)
df['ARRIVAL_DELAY']   = df['ARRIVAL_DELAY'].clip(-30, 300)
print(f'Clean: {len(df):,} rows')

# Features
df['IS_DELAYED'] = (df['ARRIVAL_DELAY'] > 15).astype(int)
df['DEPARTURE_HOUR'] = (df['SCHEDULED_DEPARTURE'] // 100).astype(int).clip(0, 23)

def tod(h):
    if 5<=h<12:  return 'Morning'
    if 12<=h<17: return 'Afternoon'
    if 17<=h<21: return 'Evening'
    return 'Night'
df['TIME_OF_DAY'] = df['DEPARTURE_HOUR'].apply(tod)

season_map = {12:'Winter',1:'Winter',2:'Winter',
              3:'Spring',4:'Spring',5:'Spring',
              6:'Monsoon',7:'Monsoon',8:'Monsoon',9:'Monsoon',
              10:'Autumn',11:'Autumn'}
df['SEASON'] = df['MONTH'].map(season_map)
df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([6,7]).astype(int)

# Fog season flag (Jan/Feb at northern airports)
fog_airports = ['DEL','LKO','PAT','IXC']
df['IS_FOG_RISK'] = (df['MONTH'].isin([1,2]) & df['ORIGIN_AIRPORT'].isin(fog_airports)).astype(int)

df.to_csv('data/flights_processed.csv', index=False)
print(f'Delay rate: {df["IS_DELAYED"].mean()*100:.1f}%')
print('Saved to data/flights_processed.csv')
print('Done!')
