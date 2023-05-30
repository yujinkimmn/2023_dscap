import streamlit as st
import pandas as pd
import time
from datetime import datetime
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
from streamlit_modal import Modal
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


fp1 = "/Users/stella/github/2023_dscap/EDA/Region_analysis/지역별마약류빈도_위경도포함_최종.csv"
fp2 = "/Users/stella/github/2023_dscap/twitterdata/labeling/total_labeling_preprocessed.csv"
fp3 = "/Users/stella/github/2023_dscap/twitterdata/preprocessing/preprocessed/total_preprocessed_name_revise.csv"  #이름 특수기호 처리

@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def popup_html():
    html = '''<!DOCTYPE html>
<html>
<head>
  <title>Twitter Design</title>
  <style>
    /* CSS styles */
    body {
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 20px;
    }
    
    .tweet {
      margin-bottom: 20px;
      border: 1px solid #ccc;
      padding: 10px;
      background-color: #f9f9f9;
    }
    
    .username {
      font-weight: bold;
      color: #1da1f2;
    }
    
    .content {
      margin-top: 5px;
    }
    
    .userid {
      font-size: 0.8em;
      color: #777;
    }
    /* Custom modal size */
    .custom-modal {
        max-width: 800px;
        height: 400px;
        overflow-y: scroll;
        scrollbar-width: thin;
    }
  </style>
</head>
<body>
  <div class="tweet">
    <div class="username">JohnDoe</div>
    <div class="userid">@johndoe123</div>
    <div class="content">
      Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis gravida lobortis aliquet.
    </div>
  </div>
  
  <div class="tweet">
    <div class="username">JaneSmith</div>
    <div class="userid">@janesmith456</div>
    <div class="content">
      Nulla nec ex vitae nisi feugiat rhoncus id non neque.
    </div>
  </div>
</body>
</html>
'''

    return html


def map_visualization():
    global fp1
    data = load_data(fp1)

    location = [[lat, lng] for lat, lng, region in zip(data['위도'], data['경도'], data['지역명']) if region == region_option]
    lat = location[0][0]
    lng = location[0][1]
    zoom = 9
    if region_option == "서울특별시":
        zoom = 12

    m = folium.Map(location=[lat, lng], zoom_start=zoom, tiles='OpenStreetMap')
    # Convert data to proper types
    lats = data['위도'].values.astype(float)
    lngs = data['경도'].values.astype(float)
    freqs = data['빈도'].values.astype(float)
    # Create a HeatMap layer
    heat_data = [[lat, lng, freq] for lat, lng, freq in zip(lats, lngs, freqs)]
    HeatMap(heat_data).add_to(m)

    # Create a MarkerCluster layer
    marker_cluster = MarkerCluster().add_to(m)

    # Add markers to the MarkerCluster layer
    for index, row in data.iterrows():
        lat = row['위도']
        lng = row['경도']
        freq = row['빈도']
        drug = row['마약류']
        popup_content = folium.Popup(f'마약류: {drug}', max_width=300)

        # Create a marker and add it to the MarkerCluster layer
        marker = folium.Marker(location=[lat, lng], popup=popup_content, tooltip='마약류 확인')
        marker.add_to(marker_cluster)

    st_data = st_folium(m, width=725)


# Set up the layout
st.set_page_config(page_title="Dashboard", page_icon="📊", 
                   layout='wide', initial_sidebar_state='expanded')

with st.sidebar:
    with st.spinner("Loading..."):
        time.sleep(0.5)
    
    st.subheader("Parameter")
    # 해당 기간의 시계열 결과를 보여주고 싶었음! 빼도 상관없어유~
    time_range = st.slider("날짜를 설정하시오", 
                           value=(datetime(2021, 1, 1), datetime(2023, 3, 31)),
                           format="YYYY/MM/DD")
    
    region_option = st.selectbox("지역을 선택하세요", 
                          ("서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시",
                           "대전광역시", "울산광역시", "세종특별자치시", "경기도", "강원도", 
                           "충청북도", "충청남도", "전라북도", "전라남도", "경상북도", "경상남도", "제주특별자치도"))
    
    
col1, col2 = st.columns([3,2])
with col1:
    st.title("Map")
    map_visualization()

    
with col2:
    st.title("Series")
    # Add code


    tab1, tab2, tab3= st.tabs(['Clustering' , 'Wordcloud', 'Histogram'])
    
    with tab1:
        st.subheader("Clustering")
        # Add code
    
    with tab2: 
        st.subheader("Wordcloud")
        # Add code

    with tab3:
        st.subheader("Histogram")
        df = pd.read_csv(fp3)
        df.drop_duplicates(['content', 'user.username', 'date'], inplace=True)  #중복으로 수집된 트윗 제거
        df_grouped = pd.DataFrame(df.groupby('user.displayname')['content'].count())  #닉네임 기준으로 개수 세기
        df_grouped.reset_index(inplace = True)
        df_grouped.sort_values('content', inplace=True, ascending=False)
        df_new = df_grouped[df_grouped['content'] >= 20]  #20번 이상 등장한 닉네임

        #히스토그램
        plt.figure(figsize = (20,15))
        sns.set(font = 'NanumGothic', font_scale = 1.5, rc = {'axes.unicode_minus': False}, style = 'darkgrid')
        fig = sns.barplot(
            x = 'content', y = 'user.displayname', data = df_new, width = 0.8)
        fig.set_yticks(np.arange(0, len(df_new)+1, 2))
        fig.set_title('User Name Appeared More Than 20 Times', fontsize = 20)
        fig.set_xlabel('Frequency', size = 15)
        fig.set_ylabel('Name', size = 15)
        fig = fig.figure
        st.pyplot(fig)
