import streamlit as st
import altair as alt
import pandas as pd
import time
import re
from vega_datasets import data
from datetime import datetime
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
from streamlit_modal import Modal
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from wordcloud import WordCloud


fp1 = "/Users/yeon/Documents/2023/ds_capstone/2023_dscap/EDA/Region_analysis/지역별마약류빈도_위경도포함_최종.csv"
fp2 = "/Users/yeon/Documents/2023/ds_capstone/2023_dscap/twitterdata/labeling/total_labeling_preprocessed.csv"
fp3 = "/Users/yeon/Documents/2023/ds_capstone/2023_dscap/twitterdata/preprocessing/preprocessed/total_preprocessed_name_revise.csv"  #이름 특수기호 처리

file_timeseries = "./files/total_preprocessed.csv"

file_clustering = "./files/total_tokenized_mecab.csv"

@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path, encoding = "utf-8")
    return data

@st.cache_data(ttl=60 * 60 * 24)
def get_timechart(data):
    hover = alt.selection_single(
        fields=["date"],
        nearest=True,
        on="mouseover",
        empty="none",
    )
    
    
    lines = (
        alt.Chart(data, height=500, title="마약 거래 게시물 추이")
        .mark_bar()
        .encode(
            x="date",
            y="count",
            color=alt.Color("count"),
        )
    )
    
    points = lines.transform_filter(hover).mark_circle(color="red", size=300)

    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="date",
            y="count",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("date", title="Date"),
                alt.Tooltip("count", title="Count"),
            ],
        )
        .add_selection(hover)
    )

    return (lines + points + tooltips).interactive()
    
def timeseries(file_path):
    global df
    df = load_data(file_path)
    global flowday
    
    for i in range(len(df['date'])):
        df['date'][i] = str(df['date'][i]).split(" ")[0]
    
    df['year'] = 0
    df['month'] = 0
    df['day'] = 0

    for i in range(len(df['date'])):
      if (str(df['date'][i])[4] == '.') :
        df['date'][i] = str(df['date'][i].split('.')[0]) + "-" + str(df['date'][i].split('.')[1]) + "-" + str(df['date'][i].split('.')[2])
      else:
        df['date'][i] = str(df['date'][i].split('-')[0]) + "-" + str(df['date'][i].split('-')[1]) + "-" + str(df['date'][i].split('-')[2])


    df['date'] = pd.to_datetime(df['date'])

    print(df['date'][0])

    for i in range(len(df['date'])):
      df['year'][i] = str(df['date'][i]).split("-")[0]
      df['month'][i] = str(df['date'][i]).split("-")[1]
      df['day'][i] = (str(df['date'][i]).split("-")[2]).split(" ")[0]

    flowday = df.groupby(['year', 'month', 'day']).count().reset_index()
    flowday = flowday.iloc[0:, :4]
    flowday['date'] = 0
    
    for i in range(len(flowday)):
        flowday['date'][i] = str(flowday['year'][i]) + "-" + str(flowday['month'][i]) + "-" + str(flowday['day'][i])

    flowday['date'] = pd.to_datetime(flowday['date'])
    flowday = flowday.iloc[0:, 3:5]
    flowday = flowday.rename(columns={'type1':'count'})
    
    print(flowday)
    # 기본 line chart 형태
    #st.line_chart(flowday, x="date", y="count")
    chart = get_timechart(flowday)
    
    ANNOTATIONS = [
    ("2021-11-03", "일 5개 이내 유지"),
    ("2022-08-19", "마약 거래 게시물 증가 추세"),
    ("2022-12-25", "마약 거래 게시물 감소 추세"),
    ("2022-10-28", "마약 거래 게시물 급증"),
    ("2023-03-28", "일 1000개 돌파"),]
    
    annotations_df = pd.DataFrame(ANNOTATIONS, columns=["date", "event"])
    annotations_df.date = pd.to_datetime(annotations_df.date)
    annotations_df["y"] = 10
    
    annotation_layer = (
    alt.Chart(annotations_df)
    .mark_text(size=20, text="⬇", dx=-8, dy=-10, align="left")
    .encode(
        x="date:T",
        y=alt.Y("y:Q"),
        tooltip=["event"],
    )
    .interactive())

    st.altair_chart((chart + annotation_layer), use_container_width=True)
    
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data
    

def visualize_clusters(df, n_clusters):
    graph = alt.Chart(df.reset_index()).mark_point(filled=True, size=60).encode(
        x=alt.X('x_values'),
        y=alt.Y('y_values'),
        color=alt.Color('labels'),
        tooltip=['word', 'labels']
    ).interactive()
    st.altair_chart(graph, use_container_width=True)
    

def clustering(file_path):
    global df
    df = load_data(file_path)
    
    print(df)
    
    for i in range(len(df['mecab'])):
        df['mecab'].loc[i] = eval(df['mecab'].loc[i])
        
    total = []
    for idx in range(len(df['mecab'])):
      word = []
      for i, j in df['mecab'].loc[idx]:
        if j == "NNP" or j == "SL" or j == "NNG":
          word.append(i)
      total.append(word)
    
    df = df.assign(word = total)
    
    corpus = []
    for i in df['word']:
      corpus.append(i)

    corpus_total = []
    for i in df['word']:
      for j in i:
        corpus_total.append(j)
    
    model = Word2Vec(corpus, vector_size=100, window = 4, min_count = 1, workers = 4, sg = 1)
    
    model_result = model.wv.most_similar('작대기')
    print(model_result)

    # fit a 2d PCA model to the vectors

    vectors = model.wv.vectors
    words = list(model.wv.key_to_index.keys())
    pca = PCA(n_components=2)
    PCA_result = pca.fit_transform(vectors)

    # prepare a dataframe
    words = pd.DataFrame(words)
    PCA_result = pd.DataFrame(PCA_result)
    PCA_result['x_values'] =PCA_result.iloc[0:, 0]
    PCA_result['y_values'] =PCA_result.iloc[0:, 1]
    PCA_final = pd.merge(words, PCA_result, left_index=True, right_index=True)
    PCA_final['word'] =PCA_final.iloc[0:, 0]
    PCA_data_complet =PCA_final[['word','x_values','y_values']]

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(PCA_result.iloc[0:,:2])
    PCA_data_complet['labels'] = kmeans.predict(PCA_result.iloc[0:,:2])
    
    # 단어 빈도 수 열 추가
    PCA_data_complet['counts'] = 0

    for i, word in enumerate(PCA_data_complet['word']):
      PCA_data_complet['counts'][i] = corpus_total.count(word)
    
    global PCA_drug
    PCA_drug = PCA_data_complet[PCA_data_complet['labels']==PCA_data_complet.loc[0].labels]
    
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(PCA_drug.iloc[0:,1:3])
    PCA_drug['labels'] = kmeans.predict(PCA_drug.iloc[0:,1:3])
    
    print(PCA_drug)
    
    visualize_clusters(PCA_drug, 2)


def w_cloud():
    PCA_drug_removed = PCA_drug

    keywords = ['스틸녹스', '신의눈물', '에토미데이트', '옥시코돈', '졸피뎀', '트라마돌', '캔디케이', '케타민',
                '러쉬파퍼', '랏슈', '정글주스', '엘에스디', '엑스터시', '마법의버섯', '환각버섯', 
                '떨액', '북한산아이스', '빙두', '삥두', '사끼', '샤부', '시원한술', '아이스술', '액상떨', '작대기',
                '히로뽕', '크리스탈', '차가운술', '아이스', '찬술', '드라퍼', '브액', '아이스드랍', '클럽약', 
                '텔레', '파티약', '패치', '후리베이스', '주사기', '허브', '물뽕', '발정제', '최음제', '사티바',
                '인디카', '합성대마', '해시시', '대마초']

    for i in keywords:
        PCA_drug_removed = PCA_drug_removed[~PCA_drug_removed['word'].str.contains(i)]  #키워드 제외한 결과

    words_rev = []
    counts_rev = []
    words_list_rev = list(PCA_drug_removed['word'])
    counts_list_rev = list(PCA_drug_removed['counts'])

    for i in range(len(words_list_rev)):
        if len(words_list_rev[i]) >= 2:  #2글자 이상인 단어만
            words_rev.append(words_list_rev[i])
            counts_rev.append(counts_list_rev[i])
    word_count_removed = dict(zip(words_rev, counts_rev))    

    #워드클라우드 그리기
    wordcloud = WordCloud(font_path = 'NanumBarunGothic', background_color = 'white', colormap = 'rainbow_r',
                     width = 4000, height = 3000).generate_from_frequencies(word_count_removed)
    fig = plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    st.pyplot(fig)   



def histogram():
    global fp3
    df = pd.read_csv(fp3)

    df.drop_duplicates(['content', 'user.username', 'date'], inplace=True)  #중복으로 수집된 트윗 제거
    df_grouped = pd.DataFrame(df.groupby('user.displayname')['content'].count())  #닉네임 기준으로 개수 세기
    df_grouped.reset_index(inplace = True)
    df_grouped.sort_values('content', inplace=True, ascending=False)
    df_new = df_grouped[df_grouped['content'] >= 20]  #20번 이상 등장한 닉네임

    #히스토그램
    plt.figure(figsize = (20,15))
    sns.set(font = 'NanumBarunGothic', font_scale = 1.5, rc = {'axes.unicode_minus': False}, style = 'darkgrid')
    fig = sns.barplot(
        x = 'content', y = 'user.displayname', data = df_new, width = 0.8)
    fig.set_yticks(np.arange(0, len(df_new)+1, 2))
    fig.set_title('User Name Appeared More Than 20 Times', fontsize = 20)
    fig.set_xlabel('Frequency', size = 15)
    fig.set_ylabel('Name', size = 15)
    fig = fig.figure
    st.pyplot(fig)

    
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
    flowday = pd.DataFrame()
    timeseries(file_timeseries)


    tab1, tab2, tab3= st.tabs(['Clustering' , 'Wordcloud', 'Histogram'])
    
    with tab1:
        st.subheader("Clustering")
        clustering(file_clustering)
        
    
    with tab2: 
        st.subheader("Wordcloud")
        w_cloud()

    with tab3:
        st.subheader("Histogram")
        histogram()
