import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.graph_objs as go
import plotly.express as px
import re

#DATA
DATA_URL = (
"Сухой Лог_ИГИ полевые работы.csv"
)
DATA_URL_PROJECT = (
"Приложение 3 Каталог горных выработок.xls"
)

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        padding-right: 3rem;
        padding-left: 4rem;
    }}
    .svg-container{{

    }} 

</style>
""",
        unsafe_allow_html=True,
    )


mydateparser = lambda x: pd.datetime.strptime(x, "%d.%m.%y")
сomma2dot = lambda x: x.replace(',', '.')

def change_date_str(x):
    if isinstance(x, str):
        if (x.replace(' ', '') == 'невскрыты') or (x == '-'):
            d = np.NAN
        else:
            d = float(x.replace(',', '.'))
    else:
        d = float(x)
    return d

def dms2dec(dms_str):
    """Return decimal representation of DMS

    #>>> dms2dec(utf8(48°53'10.18"N))
    48.8866111111F

    #>>> dms2dec(utf8(2°20'35.09"E))
    2.34330555556F

    #>>> dms2dec(utf8(48°53'10.18"S))
    -48.8866111111F

    #>>> dms2dec(utf8(2°20'35.09"W))
    -2.34330555556F

    """

    dms_str = re.sub(r'\s', '', dms_str)

    sign = -1 if re.search('[swSW]', dms_str) else 1

    numbers = [*filter(len, re.split('\D+', dms_str, maxsplit=4))]

    degree = numbers[0]
    minute = numbers[1] if len(numbers) >= 2 else '0'
    second = numbers[2] if len(numbers) >= 3 else '0'
    frac_seconds = numbers[3] if len(numbers) >= 4 else '0'

    second += "." + frac_seconds
    return sign * (int(degree) + float(minute) / 60 + float(second) / 3600)

def dm2dec(dms_str):
    """Return decimal representation of DMS

    #>>> dm2dec(utf8(48°53.18'N))

    """

    dms_str = re.sub(r'\s', '', dms_str)

    sign = -1 if re.search('[swSW]', dms_str) else 1

    numbers = [*filter(len, re.split('\D+', dms_str, maxsplit=4))]

    degree = numbers[0]
    minute = numbers[1] if len(numbers) >= 2 else '0'
    min_dec = numbers[2] if len(numbers) >= 3 else '0'

    return sign * (int(degree) + (float(minute) + (float(min_dec) / 1000)) / 60)

@st.cache(persist=True)
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows, skiprows=10, header=0, index_col='1', decimal=",", parse_dates=['8'], date_parser=mydateparser)
    data = data.drop(['Unnamed: 0'], axis=1)
    data.rename(columns={'1': 'Number', '2': 'type', '3': 'name',
                         'Unnamed: 4': 'object',
                         '4': 'latitude', '5': 'longitude', '6': 'depth_pr', '7': 'depth_f',
                         'Unnamed: 9': 'depth_rock', '8': 'date_start',
                         '9': 'date_finish', '10': 'termo', '11': 'samples',
                         '12': 'Geologist', '13': 'note'}, inplace=True)
    data.dropna(subset=['latitude','longitude'], inplace=True)

    #Преобразование данных
    data['date_finish'] = pd.to_datetime(data['date_finish'], format="%d.%m.%y")

    data['latitude'] = data['latitude'].apply(dm2dec)
    data['longitude'] = data['longitude'].apply(dm2dec)

    data['depth_rock'] = data['depth_rock'].apply(change_date_str).astype(float)

    return data

@st.cache(persist=True)
def load_data_project(nrows):
    data = pd.read_excel(DATA_URL_PROJECT, nrows=nrows, skiprows=6, header=0, parse_dates=[31])
    data.rename(columns={2: 'name', 6: 'latitude', 7: 'longitude', 3: 'depth_pr', 30: 'depth_f',
                                 31: 'Вскрытие скальных грунтов, м', 8: 'Дата начала бурения',
                                 9: 'date_finish'}, inplace=True)
    data.dropna(subset=['latitude','longitude'], inplace=True)
    return data


st.markdown("<h1 style='text-align: center;'>Интерактивная сводка</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Сухой Лог</h2>", unsafe_allow_html=True)

data = load_data(100000)
original_data = data
data_pr = load_data_project(100000)
original_data_pr = data_pr

st.markdown("**Объект:**\n Золотодобывающее предприятие на месторождении «Сухой Лог»")
st.markdown("**Заказчик:** ООО \"CЛ Золото\"")
st.markdown("**Подрядчик:** ООО \"ИнжГео\"")

today = pd.to_datetime('today')
st.markdown("**Сегодня:** " + str(today.date()))
st.markdown("**Дата начала бурения:** " + str(data['date_start'].dt.date.min()))

bhf = st.sidebar.checkbox("Пробуренные скважины", True)

start_date = st.sidebar.date_input('Начало интервала', pd.to_datetime(data['date_finish'].dt.date.min()))
end_date = st.sidebar.date_input('Конец интервала', today)


df = data[(data['date_finish'].dt.date >= start_date) & (data['date_finish'].dt.date < end_date)]


st.markdown("Пробурено с **" + str(start_date) + "** по **" + str(end_date) +"** : **" + "{:.1f}".format(
    df['depth_f'].sum()) + " п.м.**")

bhp = st.sidebar.checkbox("Проектные скважины", False)

st.sidebar.markdown("**Сведения о вскрытии скальных грунтов**")
bh_rock = st.sidebar.checkbox("Отобразить информацию о вскрытии скальных грунтов ", True)

df_f = df[['name', 'latitude', 'longitude']]
df_pr = data_pr[['name', 'latitude', 'longitude']]
df_rock = df[['depth_rock', 'latitude', 'longitude']].dropna(how='any')

st.markdown("<h2 style='text-align: center;'>Карта фактического материала</h1>", unsafe_allow_html=True)

map_layers=[]

#Добавление проектных скважин
if bhp:
    map_layers.append(pdk.Layer(
             'ScatterplotLayer',
             data=df_pr,
             get_position='[longitude, latitude]',
             get_color='[200, 0, 0, 100]',
             get_radius=10))
    map_layers.append(pdk.Layer(
        'TextLayer',
        data=df_pr,
        get_position='[longitude, latitude]',
        get_text='name',
        get_size=23,
        get_color=[200, 50, 0],
        get_angle=0,
    ))
#Добавление пробуренных скважин
if bhf:
    map_layers.append(pdk.Layer(
             'ScatterplotLayer',
             data=df_f,
             get_position='[longitude, latitude]',
             get_color='[0, 200, 0, 150]',
             get_radius=10))
    map_layers.append(pdk.Layer(
        'TextLayer',
        data=df_f,
        get_position='[longitude, latitude]',
        get_text='name',
        get_size=23,
        get_color=[0, 200, 100],
        get_angle=0,
    ))

if bh_rock:
    map_layers.append(pdk.Layer(
    "ColumnLayer",
    data=df_rock,
    opacity=0.2,
    get_position='[longitude, latitude]',
    threshold=0.75,
    get_elevation="depth_rock",
    elevation_scale=30,
    radius=25,
    pickable=True,
    auto_highlight=True,
    get_fill_color=["depth_rock * 50", "250 - depth_rock * 25", "depth_rock * 0", "depth_rock * 25"]
    ))

tooltip = {
    "html": "<b>{depth_rock}</b> метров до скальных грунтов",
    "style": {"background": "grey", "color": "white", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"},
}

st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/satellite-v9',
    initial_view_state=pdk.ViewState(
         latitude=np.average(data['latitude']).item(),
         longitude=np.average(data['longitude']).item(),
         zoom=12,
         #pitch=50,
     ),
     layers=map_layers,
    tooltip=tooltip,
 ))

st.markdown("<p style='text-align: center;'>Список скважин пробуренных с <b>" + str(start_date) + "</b> по <b>" + str(end_date) + "</b> </p>", unsafe_allow_html=True)
st.dataframe(df)

res = df.groupby(['date_finish']).sum()
res.reset_index(inplace=True)

#Проходка за сутки-------------------------------------------------------------------------------------------

st.markdown("<br><br><p style='text-align: center;'>Проходка за сутки в период с <b>" + str(start_date) + "</b> по <b>" + str(end_date) + "</b> </p>", unsafe_allow_html=True)
sl_width = st.slider("Ширина графиков", 220, 1000, step=20)

fig = px.line(res, x='date_finish', y='depth_f')
fig.update_layout(xaxis_title="Дата",
                  yaxis_title="Метраж, п.м.",
                  width=sl_width,
                  height=sl_width-int(0.2*sl_width),
                  margin=dict(l=20, r=20, t=0, b=0))
st.write(fig)

acc_date = {}

for i in res['date_finish']:
    acc_date[i] = np.around(res[res['date_finish']<=i]['depth_f'].sum(), 2)
acc_date = pd.Series(acc_date)
acc_date = pd.DataFrame(acc_date)
acc_date.reset_index(inplace=True)
acc_date.rename(columns={'index': 'date', 0: 'depth'}, inplace=True)

data_smg = pd.read_excel('СМГ_по объектам.xls', skiprows=3, header=0)
data_smg.drop(['Unnamed: '+ str(i) for i in range(10)], axis=1, inplace=True)
data_smg = data_smg.iloc[4].transpose()
data_smg.dropna(inplace=True)
data_smg = pd.DataFrame(data_smg)
data_smg.reset_index(inplace=True)
data_smg.rename(columns={'index': 'date', 4: 'depth'}, inplace=True)

acc_date_pr = {}
for i in data_smg['date']:
    acc_date_pr[i] = np.around(data_smg[data_smg['date']<=i]['depth'].sum(), 2)
acc_date_pr = pd.Series(acc_date_pr)
acc_date_pr = pd.DataFrame(acc_date_pr)
acc_date_pr.reset_index(inplace=True)
acc_date_pr.rename(columns={'index': 'date', 0: 'depth'}, inplace=True)


st.markdown("<br><br><p style='text-align: center;'>Куммулятивная кривая метража с <b>" + str(start_date) + "</b> по <b>" + str(end_date) + "</b> </p>", unsafe_allow_html=True)
fig = go.Figure()
fig.add_trace(go.Scatter(x=acc_date['date'], y=acc_date['depth'],  name='Фактический'))
fig.add_trace(go.Scatter(x=acc_date_pr['date'], y=acc_date_pr['depth'],  name='Проектный'))
fig.update_layout(legend_orientation="h",
                  legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.05),
                  xaxis_title="Дата",
                  yaxis_title="Метраж, п.м.",
                  margin=dict(l=20, r=20, t=0, b=0),
                  width=sl_width,
                  height=sl_width-int(0.2*sl_width),
                  xaxis_range=[acc_date['date'].iloc[0], acc_date['date'].iloc[acc_date.shape[0]-1]])
st.write(fig)


#Количество буровой технике
data_tech = pd.read_excel('СМГ_по объектам.xls',skiprows=3, header=0)
data_tech = data_tech.iloc[0].transpose()
data_tech.dropna(inplace=True)
data_tech.drop(['Unnamed: 0', 'Unnamed: 1'], inplace=True)
data_tech = pd.DataFrame(data_tech)
data_tech.reset_index(inplace=True)
data_tech.rename(columns={'index': 'date', 0: 'tech'}, inplace=True)

st.markdown("<br><br><p style='text-align: center;'><b>Проектное количество буровой техники, шт.</b> </p>", unsafe_allow_html=True)
fig = go.Figure()
fig.add_trace(go.Scatter(x=data_tech['date'], y=data_tech['tech'],  name='Количество буровой технике, шт'))
fig.update_layout(legend_orientation="h",
                  legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.98),
                  xaxis_title="Дата",
                  yaxis_title="Метраж, п.м.",
                  width=sl_width,
                  height=sl_width-int(0.2*sl_width),
                  margin=dict(l=20, r=20, t=0, b=0),
                  )
st.write(fig)
