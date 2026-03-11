import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os

os.makedirs('data/charts', exist_ok=True)

print('Loading data...')
df = pd.read_csv('data/flights_processed.csv', low_memory=False)
print(f'Loaded {len(df):,} rows')

BG = '#FFF8F0'; ACCENT = '#FF6B35'; BLUE = '#1B4F72'; GOLD = '#F39C12'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.grid': True, 'grid.color': '#E8E8E8',
    'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold'
})

# Chart 1 — Delay Overview
print('Chart 1: Delay Overview...')
delayed = df['IS_DELAYED'].sum(); on_time = len(df) - delayed
fig, ax = plt.subplots(figsize=(6,6))
ax.pie([on_time, delayed], labels=['On Time','Delayed (>15 min)'],
       colors=[BLUE, ACCENT], autopct='%1.1f%%', startangle=90,
       wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2),
       textprops=dict(fontsize=13))
ax.set_title('Indian Domestic Flight\nOn-Time Performance 2024', pad=20, fontsize=14)
ax.text(0, 0, f'{len(df)/1e6:.1f}M\nFlights', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#333')
fig.patch.set_facecolor(BG)
plt.savefig('data/charts/01_delay_overview.png', dpi=150, bbox_inches='tight'); plt.close()

# Chart 2 — Top Airports by Delay
print('Chart 2: Top Airports...')
top = (df.groupby(['ORIGIN_AIRPORT','ORIGIN_CITY'])['DEPARTURE_DELAY']
         .agg(['mean','count']).query('count > 300')
         .sort_values('mean', ascending=False).head(12).reset_index())
top['label'] = top['ORIGIN_AIRPORT'] + '\n(' + top['ORIGIN_CITY'] + ')'
fig, ax = plt.subplots(figsize=(11,6))
clrs = [ACCENT if v > top['mean'].median() else BLUE for v in top['mean']]
bars = ax.barh(top['label'][::-1], top['mean'][::-1], color=clrs[::-1], edgecolor='white', height=0.6)
for bar, v in zip(bars, top['mean'][::-1]):
    ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2, f'{v:.1f} min', va='center', fontsize=9)
ax.set_xlabel('Average Departure Delay (minutes)')
ax.set_title('Top Indian Airports — Highest Average Departure Delay')
ax.axvline(top['mean'].median(), color='gray', linestyle='--', linewidth=1.2, label='Median')
ax.legend(); plt.tight_layout()
plt.savefig('data/charts/02_top_airports.png', dpi=150, bbox_inches='tight'); plt.close()

# Chart 3 — Delay by Airline
print('Chart 3: Delay by Airline...')
al = (df.groupby('AIRLINE_NAME').agg(avg=('DEPARTURE_DELAY','mean'),
      otp=('IS_DELAYED','mean'), n=('IS_DELAYED','count'))
      .query('n > 1000').sort_values('avg', ascending=False).reset_index())
al['otp'] = (1 - al['otp']) * 100  # convert to OTP %
fig, axes = plt.subplots(1, 2, figsize=(14,6))
pal = sns.color_palette('RdYlGn_r', len(al))
axes[0].barh(al['AIRLINE_NAME'][::-1], al['avg'][::-1], color=pal[::-1], edgecolor='white')
axes[0].set_xlabel('Avg Departure Delay (min)'); axes[0].set_title('Average Departure Delay by Airline')
axes[1].barh(al['AIRLINE_NAME'][::-1], al['otp'][::-1], color=pal, edgecolor='white')
axes[1].xaxis.set_major_formatter(mtick.PercentFormatter())
axes[1].set_xlabel('On-Time Performance (%)'); axes[1].set_title('On-Time Performance by Airline')
# Add DGCA benchmark line
axes[1].axvline(70, color='navy', linestyle='--', linewidth=1.5, label='DGCA 70% benchmark')
axes[1].legend()
plt.suptitle('Indian Airline Performance — DGCA 2024 Data', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('data/charts/03_delay_by_airline.png', dpi=150, bbox_inches='tight'); plt.close()

# Chart 4 — Delay by Hour
print('Chart 4: Delay by Hour...')
h = df.groupby('DEPARTURE_HOUR').agg(avg=('DEPARTURE_DELAY','mean'),
    rate=('IS_DELAYED','mean')).reset_index()
h['rate'] *= 100
fig, ax1 = plt.subplots(figsize=(12,5))
ax2 = ax1.twinx()
ax1.bar(h['DEPARTURE_HOUR'], h['avg'], color=BLUE, alpha=0.75, width=0.7, label='Avg Delay')
ax2.plot(h['DEPARTURE_HOUR'], h['rate'], color=ACCENT, linewidth=2.5, marker='o', markersize=5)
ax1.set_xlabel('Hour of Day (IST)'); ax1.set_ylabel('Avg Departure Delay (min)', color=BLUE)
ax2.set_ylabel('Delay Rate (%)', color=ACCENT)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax1.set_xticks(range(5,23)); ax1.set_title('Flight Delay Patterns by Hour of Day (IST)')
ax1.axvspan(18, 22, alpha=0.07, color=ACCENT, label='Evening peak')
plt.tight_layout()
plt.savefig('data/charts/04_delay_by_hour.png', dpi=150, bbox_inches='tight'); plt.close()

# Chart 5 — Monthly Trends
print('Chart 5: Monthly Trends...')
m = df.groupby('MONTH').agg(avg=('DEPARTURE_DELAY','mean'),
    rate=('IS_DELAYED','mean')).reset_index()
m['rate'] *= 100
mnames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
fig, axes = plt.subplots(2, 1, figsize=(12,8), sharex=True)
axes[0].fill_between(m['MONTH'], m['avg'], alpha=0.3, color=BLUE)
axes[0].plot(m['MONTH'], m['avg'], color=BLUE, linewidth=2.5, marker='o')
axes[0].set_ylabel('Avg Delay (min)'); axes[0].set_title('Monthly Average Departure Delay')
# Shade fog & monsoon seasons
axes[0].axvspan(0.5, 2.5, alpha=0.1, color='lightblue', label='Fog Season')
axes[0].axvspan(5.5, 9.5, alpha=0.1, color='lightgreen', label='Monsoon Season')
axes[0].legend()
axes[1].fill_between(m['MONTH'], m['rate'], alpha=0.3, color=ACCENT)
axes[1].plot(m['MONTH'], m['rate'], color=ACCENT, linewidth=2.5, marker='o')
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[1].set_ylabel('Delay Rate (%)'); axes[1].set_title('Monthly Delay Rate (>15 min)')
axes[1].set_xticks(m['MONTH']); axes[1].set_xticklabels(mnames)
axes[1].axvspan(0.5, 2.5, alpha=0.1, color='lightblue')
axes[1].axvspan(5.5, 9.5, alpha=0.1, color='lightgreen')
plt.suptitle('Seasonal Delay Trends — Indian Domestic Aviation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('data/charts/05_monthly_trends.png', dpi=150, bbox_inches='tight'); plt.close()

# Chart 6 — Delay Causes
print('Chart 6: Delay Causes...')
causes = {
    'Weather / Fog':    df['WEATHER_DELAY'].sum(),
    'Airline / Ops':    df['AIRLINE_DELAY'].sum(),
    'Reactionary\n(Late Aircraft)': df['LATE_AIRCRAFT_DELAY'].sum(),
    'ATC / Air Traffic': df['ATC_DELAY'].sum(),
    'Technical':        df['TECHNICAL_DELAY'].sum(),
}
cs = pd.Series(causes).sort_values()
fig, ax = plt.subplots(figsize=(9,5))
clrs = sns.color_palette('Set2', len(cs))
bars = ax.barh(cs.index, cs.values, color=clrs, edgecolor='white')
total = cs.sum()
for bar, v in zip(bars, cs.values):
    ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
            f'{v/total*100:.1f}%', va='center', fontsize=10)
ax.set_xlabel('Total Delay Minutes')
ax.set_title('Flight Delay Causes — Indian Domestic Aviation')
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f'{x/1e6:.1f}M'))
plt.tight_layout()
plt.savefig('data/charts/06_delay_causes.png', dpi=150, bbox_inches='tight'); plt.close()

# Chart 7 — DOW x Hour Heatmap
print('Chart 7: Heatmap...')
pivot = df.groupby(['DAY_OF_WEEK','DEPARTURE_HOUR'])['IS_DELAYED'].mean().unstack() * 100
pivot.index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
fig, ax = plt.subplots(figsize=(14,4))
sns.heatmap(pivot, cmap='RdYlGn_r', linewidths=0.2,
            cbar_kws={'label':'Delay Rate (%)'}, ax=ax)
ax.set_xlabel('Hour of Day (IST)'); ax.set_title('Delay Rate Heatmap — Day of Week vs Hour (IST)')
fig.patch.set_facecolor(BG); plt.tight_layout()
plt.savefig('data/charts/07_heatmap.png', dpi=150, bbox_inches='tight'); plt.close()

# Chart 8 — Fog vs Non-Fog Delay
print('Chart 8: Fog Impact...')
fog_comp = df.groupby('IS_FOG_RISK')['DEPARTURE_DELAY'].agg(['mean','median']).reset_index()
fog_comp['IS_FOG_RISK'] = fog_comp['IS_FOG_RISK'].map({0:'Normal Season', 1:'Fog Season\n(Jan/Feb, North India)'})
fig, ax = plt.subplots(figsize=(7,5))
x = range(len(fog_comp))
bars1 = ax.bar([i-0.2 for i in x], fog_comp['mean'], width=0.35, color=BLUE, label='Mean Delay', edgecolor='white')
bars2 = ax.bar([i+0.2 for i in x], fog_comp['median'], width=0.35, color=ACCENT, label='Median Delay', edgecolor='white')
ax.set_xticks(list(x)); ax.set_xticklabels(fog_comp['IS_FOG_RISK'])
ax.set_ylabel('Departure Delay (min)'); ax.set_title('Impact of Fog Season on Flight Delays\n(North Indian Airports)')
ax.legend(); plt.tight_layout()
plt.savefig('data/charts/08_fog_impact.png', dpi=150, bbox_inches='tight'); plt.close()

print('\nAll 8 charts saved to data/charts/')
print('Done!')
