import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os, joblib

st.set_page_config(page_title="🇮🇳 Indian Flight Delay Analytics", page_icon="✈️", layout="wide")

ACCENT='#FF6B35'; BLUE='#1B4F72'; BG='#FFF8F0'
AIRLINE_NAMES={'6E':'IndiGo','AI':'Air India','QP':'Akasa Air','SG':'SpiceJet','I5':'AIX Connect','IX':'Air India Express','S5':'Alliance Air'}
AIRPORT_CITIES={'DEL':'Delhi','BOM':'Mumbai','BLR':'Bengaluru','MAA':'Chennai','HYD':'Hyderabad','CCU':'Kolkata','COK':'Kochi','PNQ':'Pune','GOI':'Goa','AMD':'Ahmedabad','JAI':'Jaipur','LKO':'Lucknow','PAT':'Patna','IXC':'Chandigarh','BBI':'Bhubaneswar'}

@st.cache_data(show_spinner='Loading Indian flight data...')
def load_data():
    path='data/flights_processed.csv'
    if os.path.exists(path): return pd.read_csv(path,low_memory=False)
    st.error('Run run_preprocessing.py first!'); st.stop()

df=load_data()
st.sidebar.title('🇮🇳 Indian Flight\nDelay Analytics')
st.sidebar.caption('Based on DGCA 2024 Data')
st.sidebar.markdown('---')
st.sidebar.header('🔍 Filters')
all_airlines=sorted(df['AIRLINE_NAME'].dropna().unique().tolist())
sel_airlines=st.sidebar.multiselect('Airline',all_airlines,default=all_airlines)
month_names={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
sel_months=st.sidebar.multiselect('Month',list(month_names.keys()),format_func=lambda m:month_names[m],default=list(month_names.keys()))
all_tod=['Morning','Afternoon','Evening','Night']
sel_tod=st.sidebar.multiselect('Time of Day (IST)',all_tod,default=all_tod)
filtered=df[df['AIRLINE_NAME'].isin(sel_airlines)&df['MONTH'].isin(sel_months)&df['TIME_OF_DAY'].astype(str).isin(sel_tod)]

st.markdown("<h1 style='text-align:center;color:#1B4F72;'>🇮🇳 Indian Domestic Flight Delay Analytics</h1>",unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#666;font-size:15px;'>DGCA 2024 On-Time Performance | IndiGo · Air India · SpiceJet · Akasa · AIX Connect · Air India Express</p>",unsafe_allow_html=True)
st.markdown('---')

total=len(filtered); delayed=filtered['IS_DELAYED'].sum()
d_rate=delayed/total*100 if total else 0
avg_dep=filtered[filtered['DEPARTURE_DELAY']>0]['DEPARTURE_DELAY'].mean()
avg_arr=filtered['ARRIVAL_DELAY'].mean(); otp=100-d_rate

c1,c2,c3,c4,c5=st.columns(5)
c1.metric('✈️ Total Flights',f'{total:,.0f}')
c2.metric('⚠️ Delayed',f'{delayed:,.0f}',f'{d_rate:.1f}%')
c3.metric('✅ OTP',f'{otp:.1f}%')
c4.metric('🕐 Avg Dep Delay',f'{avg_dep:.1f} min')
c5.metric('🛬 Avg Arr Delay',f'{avg_arr:.1f} min')
st.markdown('---')

tab1,tab2,tab3,tab4=st.tabs(['📅 Delay Trends','🏢 By Airport','🛫 By Airline','🤖 Predict Delay'])

with tab1:
    st.subheader('📅 Monthly Delay Trends')
    monthly=(filtered.groupby('MONTH').agg(avg=('DEPARTURE_DELAY','mean'),rate=('IS_DELAYED','mean')).reset_index())
    monthly['rate']*=100; monthly['mn']=monthly['MONTH'].apply(lambda m:month_names[m])
    col1,col2=st.columns(2)
    with col1:
        fig,ax=plt.subplots(figsize=(7,4)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        ax.fill_between(monthly['MONTH'],monthly['avg'],alpha=0.3,color=BLUE)
        ax.plot(monthly['MONTH'],monthly['avg'],color=BLUE,linewidth=2.5,marker='o')
        ax.axvspan(0.5,2.5,alpha=0.1,color='lightblue',label='Fog Season')
        ax.axvspan(5.5,9.5,alpha=0.1,color='lightgreen',label='Monsoon')
        ax.legend(fontsize=8); ax.set_xticks(monthly['MONTH']); ax.set_xticklabels(monthly['mn'])
        ax.set_ylabel('Avg Delay (min)'); ax.set_title('Monthly Average Delay')
        st.pyplot(fig); plt.close()
    with col2:
        fig,ax=plt.subplots(figsize=(7,4)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        ax.fill_between(monthly['MONTH'],monthly['rate'],alpha=0.3,color=ACCENT)
        ax.plot(monthly['MONTH'],monthly['rate'],color=ACCENT,linewidth=2.5,marker='o')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.axvspan(0.5,2.5,alpha=0.1,color='lightblue'); ax.axvspan(5.5,9.5,alpha=0.1,color='lightgreen')
        ax.set_xticks(monthly['MONTH']); ax.set_xticklabels(monthly['mn'])
        ax.set_ylabel('Delay Rate (%)'); ax.set_title('Monthly Delay Rate')
        st.pyplot(fig); plt.close()
    st.subheader('🗓️ Delay Heatmap — Day of Week × Hour (IST)')
    pivot=(filtered.groupby(['DAY_OF_WEEK','DEPARTURE_HOUR'])['IS_DELAYED'].mean().unstack()*100)
    pivot.index=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    fig,ax=plt.subplots(figsize=(14,4))
    sns.heatmap(pivot,cmap='RdYlGn_r',linewidths=0.2,cbar_kws={'label':'Delay Rate (%)'},ax=ax)
    ax.set_title('Delay Rate Heatmap'); fig.patch.set_facecolor(BG); st.pyplot(fig); plt.close()

with tab2:
    st.subheader('🏢 Delay by Airport')
    top_n=st.slider('Show top N airports',5,15,10)
    ap=(filtered.groupby('ORIGIN_AIRPORT').agg(avg=('DEPARTURE_DELAY','mean'),rate=('IS_DELAYED','mean'),n=('IS_DELAYED','count')).query('n>100').sort_values('avg',ascending=False).head(top_n).reset_index())
    ap['rate']*=100; ap['label']=ap['ORIGIN_AIRPORT']+' ('+ap['ORIGIN_AIRPORT'].map(AIRPORT_CITIES).fillna('')+')'
    col1,col2=st.columns(2)
    with col1:
        fig,ax=plt.subplots(figsize=(8,5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        clrs=[ACCENT if v>ap['avg'].median() else BLUE for v in ap['avg']]
        ax.barh(ap['label'][::-1],ap['avg'][::-1],color=clrs[::-1],edgecolor='white')
        ax.set_xlabel('Avg Delay (min)'); ax.set_title('Avg Delay by Airport'); st.pyplot(fig); plt.close()
    with col2:
        fig,ax=plt.subplots(figsize=(8,5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        ax.barh(ap['label'][::-1],ap['rate'][::-1],color=ACCENT,edgecolor='white',alpha=0.85)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_xlabel('Delay Rate (%)'); ax.set_title('Delay Rate by Airport'); st.pyplot(fig); plt.close()

with tab3:
    st.subheader('🛫 Airline On-Time Performance')
    al=(filtered.groupby('AIRLINE_NAME').agg(avg=('DEPARTURE_DELAY','mean'),otp=('IS_DELAYED','mean'),n=('IS_DELAYED','count')).query('n>500').sort_values('avg',ascending=False).reset_index())
    al['otp']=(1-al['otp'])*100
    col1,col2=st.columns(2); pal=sns.color_palette('RdYlGn_r',len(al))
    with col1:
        fig,ax=plt.subplots(figsize=(8,5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        ax.barh(al['AIRLINE_NAME'][::-1],al['avg'][::-1],color=pal[::-1],edgecolor='white')
        ax.set_xlabel('Avg Departure Delay (min)'); ax.set_title('Average Delay by Airline'); st.pyplot(fig); plt.close()
    with col2:
        fig,ax=plt.subplots(figsize=(8,5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        ax.barh(al['AIRLINE_NAME'][::-1],al['otp'][::-1],color=pal,edgecolor='white')
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.axvline(70,color='red',linestyle='--',linewidth=1.5,label='DGCA 70% target')
        ax.legend(); ax.set_xlabel('On-Time Performance (%)'); ax.set_title('OTP by Airline'); st.pyplot(fig); plt.close()
    st.info('📊 DGCA 2024: IndiGo leads at 73.4% OTP | SpiceJet lowest at 48.6% | Akasa Air 62.7%')

with tab4:
    st.subheader('🤖 Flight Delay Prediction Tool')
    st.markdown('Enter Indian flight details to predict delay probability.')
    col1,col2,col3=st.columns(3)
    with col1:
        airline=st.selectbox('Airline',sorted(df['AIRLINE'].dropna().unique()),format_func=lambda x:AIRLINE_NAMES.get(x,x))
        dep_hour=st.slider('Departure Hour (IST)',5,23,9)
        month=st.selectbox('Month',list(range(1,13)),format_func=lambda m:month_names[m])
    with col2:
        origin=st.selectbox('Origin Airport',sorted(df['ORIGIN_AIRPORT'].dropna().unique()),format_func=lambda x:f'{x} — {AIRPORT_CITIES.get(x,"")}')
        dow=st.selectbox('Day of Week',[1,2,3,4,5,6,7],format_func=lambda d:['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][d-1])
        dist=st.number_input('Distance (km)',200,3000,1200)
    with col3:
        dep_delay=st.number_input('Current Departure Delay (min)',-20,300,0)
        is_fog_risk=1 if(month in[1,2] and origin in['DEL','LKO','PAT','IXC']) else 0
        if is_fog_risk: st.warning('⚠️ Fog Risk: High fog delay risk in Jan/Feb!')
    if st.button('🔮 Predict Delay Probability',type='primary'):
        def get_tod(h):
            if 5<=h<12: return 'Morning'
            if 12<=h<17: return 'Afternoon'
            if 17<=h<21: return 'Evening'
            return 'Night'
        season_map={1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',6:'Monsoon',7:'Monsoon',8:'Monsoon',9:'Monsoon',10:'Autumn',11:'Autumn',12:'Winter'}
        model_path='data/models/random_forest.joblib'; le_path='data/models/label_encoders.joblib'
        if os.path.exists(model_path):
            rf_model=joblib.load(model_path); le=joblib.load(le_path)
            def safe_encode(enc,val):
                try: return enc.transform([str(val)])[0]
                except: return 0
            features={'DEPARTURE_DELAY':dep_delay,'DEPARTURE_HOUR':dep_hour,'DISTANCE':dist,'DAY_OF_WEEK':dow,'MONTH':month,'AIRLINE':safe_encode(le['AIRLINE'],airline),'ORIGIN_AIRPORT':safe_encode(le['ORIGIN_AIRPORT'],origin),'TIME_OF_DAY':safe_encode(le['TIME_OF_DAY'],get_tod(dep_hour)),'SEASON':safe_encode(le['SEASON'],season_map[month]),'IS_FOG_RISK':is_fog_risk,'IS_WEEKEND':1 if dow in[6,7] else 0}
            X=pd.DataFrame([features]); prob=rf_model.predict_proba(X)[0][1]
        else:
            score=(max(0,dep_delay)*0.4+(dep_hour>18)*15+is_fog_risk*20+(month in[6,7,8,9])*8+(airline in['SG','S5'])*10)
            prob=min(0.95,max(0.05,score/100))
        pred='Delayed' if prob>=0.5 else 'On Time'
        st.markdown('---')
        r1,r2=st.columns([1,2])
        with r1:
            if pred=='Delayed': st.error(f'## ⚠️ {pred}')
            else: st.success(f'## ✅ {pred}')
            st.metric('Delay Probability',f'{prob*100:.1f}%')
            st.caption(f'**{AIRLINE_NAMES.get(airline,airline)}** from **{origin} ({AIRPORT_CITIES.get(origin,"")})**')
        with r2:
            fig,ax=plt.subplots(figsize=(5,2))
            ax.barh([''],[prob],color=ACCENT,height=0.4)
            ax.barh([''],[1-prob],left=[prob],color=BLUE,height=0.4)
            ax.axvline(0.5,color='white',linestyle='--',linewidth=2)
            ax.set_xlim(0,1); ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
            ax.set_title(f'Delay Probability: {prob*100:.1f}%')
            fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.set_yticks([])
            st.pyplot(fig); plt.close()

st.markdown('---')
