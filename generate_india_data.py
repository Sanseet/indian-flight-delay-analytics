import pandas as pd
import numpy as np
import os

os.makedirs('data', exist_ok=True)

rng = np.random.default_rng(42)
n = 500_000

# ── Real Indian Airlines (DGCA 2024 data) ──────────────────────────────
# OTP rates from DGCA: IndiGo 73.4%, Air India 67.6%, Akasa 62.7%, SpiceJet 48.6%
AIRLINES = {
    '6E': {'name': 'IndiGo',           'otp': 0.734, 'market_share': 0.619},
    'AI': {'name': 'Air India',         'otp': 0.676, 'market_share': 0.148},
    'QP': {'name': 'Akasa Air',         'otp': 0.627, 'market_share': 0.048},
    'SG': {'name': 'SpiceJet',          'otp': 0.486, 'market_share': 0.037},
    'I5': {'name': 'AIX Connect',       'otp': 0.730, 'market_share': 0.051},
    'IX': {'name': 'Air India Express', 'otp': 0.650, 'market_share': 0.065},
    'S5': {'name': 'Alliance Air',      'otp': 0.530, 'market_share': 0.032},
}

# ── Real Indian Airports ────────────────────────────────────────────────
AIRPORTS = {
    'DEL': {'name': 'Indira Gandhi Intl, Delhi',      'city': 'Delhi',     'delay_bias': 8.0},
    'BOM': {'name': 'Chhatrapati Shivaji Intl, Mumbai','city': 'Mumbai',   'delay_bias': 6.5},
    'BLR': {'name': 'Kempegowda Intl, Bengaluru',     'city': 'Bengaluru', 'delay_bias': 4.0},
    'MAA': {'name': 'Chennai Intl',                    'city': 'Chennai',  'delay_bias': 3.5},
    'HYD': {'name': 'Rajiv Gandhi Intl, Hyderabad',   'city': 'Hyderabad', 'delay_bias': 3.0},
    'CCU': {'name': 'Netaji Subhas Chandra Bose Intl, Kolkata','city':'Kolkata','delay_bias': 5.0},
    'COK': {'name': 'Cochin Intl',                    'city': 'Kochi',     'delay_bias': 2.0},
    'PNQ': {'name': 'Pune Airport',                   'city': 'Pune',      'delay_bias': 2.5},
    'GOI': {'name': 'Goa Intl (Dabolim)',              'city': 'Goa',      'delay_bias': 3.0},
    'AMD': {'name': 'Sardar Vallabhbhai Patel Intl, Ahmedabad','city':'Ahmedabad','delay_bias': 2.0},
    'JAI': {'name': 'Jaipur Intl',                    'city': 'Jaipur',    'delay_bias': 2.5},
    'LKO': {'name': 'Chaudhary Charan Singh Intl, Lucknow','city':'Lucknow','delay_bias': 4.0},
    'PAT': {'name': 'Jay Prakash Narayan Intl, Patna','city': 'Patna',     'delay_bias': 5.5},
    'IXC': {'name': 'Chandigarh Intl',                'city': 'Chandigarh','delay_bias': 3.0},
    'BBI': {'name': 'Biju Patnaik Intl, Bhubaneswar', 'city': 'Bhubaneswar','delay_bias':3.5},
}

airport_codes = list(AIRPORTS.keys())
airline_codes = list(AIRLINES.keys())
airline_weights = [AIRLINES[a]['market_share'] for a in airline_codes]

# Generate flights
al = rng.choice(airline_codes, n, p=airline_weights)
orig = rng.choice(airport_codes, n)
dest = rng.choice(airport_codes, n)
# ensure origin != destination
same = orig == dest
while same.any():
    dest[same] = rng.choice(airport_codes, same.sum())
    same = orig == dest

months   = rng.integers(1, 13, n)
days     = rng.integers(1, 29, n)
dow      = rng.integers(1, 8, n)
hours    = rng.integers(5, 23, n)

# Distance (km) — realistic India domestic routes
dist_km = rng.integers(200, 2800, n)

# ── Delay model based on real DGCA OTP ─────────────────────────────────
airline_delay_base = {code: (1 - info['otp']) * 40 for code, info in AIRLINES.items()}
airport_bias_arr   = np.array([AIRPORTS[a]['delay_bias'] for a in orig])

# Seasonal: fog in Jan/Feb (Delhi, Lucknow, Patna), monsoon Jul/Aug/Sep
fog_months    = np.isin(months, [1, 2])
monsoon_months = np.isin(months, [6, 7, 8, 9])
fog_airports  = np.isin(orig, ['DEL', 'LKO', 'PAT', 'IXC'])
fog_factor    = (fog_months & fog_airports).astype(float) * 25
monsoon_factor = monsoon_months.astype(float) * 8

# Evening peak delay
evening_factor = np.where(hours >= 18, 12, 0)
night_factor   = np.where(hours >= 21, 8, 0)

# Airline base delay
al_base = np.array([airline_delay_base[a] for a in al])

dep_delay = np.clip(
    rng.normal(al_base + airport_bias_arr + fog_factor + monsoon_factor + evening_factor + night_factor, 18),
    -20, 300
)
arr_delay = dep_delay + rng.normal(0, 6, n)

# Delay causes
weather_delay      = np.clip(rng.normal(fog_factor * 0.4 + monsoon_factor * 0.3, 5), 0, 120)
airline_delay_col  = np.clip(rng.normal(al_base * 0.5, 8), 0, 120)
reactionary_delay  = np.clip(rng.normal(al_base * 0.35, 10), 0, 120)  # late aircraft
atc_delay          = np.clip(rng.normal(3, 6), 0, 60)  # ATC / air traffic
technical_delay    = np.clip(rng.normal(2, 5), 0, 60)

df = pd.DataFrame({
    'AIRLINE':              al,
    'AIRLINE_NAME':         [AIRLINES[a]['name'] for a in al],
    'ORIGIN_AIRPORT':       orig,
    'ORIGIN_CITY':          [AIRPORTS[a]['city'] for a in orig],
    'DESTINATION_AIRPORT':  dest,
    'DESTINATION_CITY':     [AIRPORTS[a]['city'] for a in dest],
    'MONTH':                months,
    'DAY':                  days,
    'DAY_OF_WEEK':          dow,
    'SCHEDULED_DEPARTURE':  hours * 100 + rng.integers(0, 60, n),
    'DEPARTURE_DELAY':      dep_delay,
    'ARRIVAL_DELAY':        arr_delay,
    'DISTANCE':             dist_km,
    'WEATHER_DELAY':        weather_delay,
    'AIRLINE_DELAY':        airline_delay_col,
    'LATE_AIRCRAFT_DELAY':  reactionary_delay,
    'ATC_DELAY':            atc_delay,
    'TECHNICAL_DELAY':      technical_delay,
    'CANCELLED':            (rng.random(n) < 0.012).astype(int),
    'DIVERTED':             (rng.random(n) < 0.002).astype(int),
    'YEAR':                 2024,
})

df.to_csv('data/flights.csv', index=False)
print(f"Generated {len(df):,} Indian domestic flights")
print(f"Airlines: {df['AIRLINE_NAME'].unique().tolist()}")
print(f"Airports: {sorted(df['ORIGIN_AIRPORT'].unique().tolist())}")
print(f"Delay rate (>15 min): {(df['ARRIVAL_DELAY'] > 15).mean()*100:.1f}%")
print("Saved to data/flights.csv")
