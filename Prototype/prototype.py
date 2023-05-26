import streamlit as st
import pandas as pd
import time
from datetime import datetime

file_path = ""

@st.cache
def load_data():
    data = pd.read_csv(file_path)
    return data

# Set up the layout
st.set_page_config(layout="wide", initial_sidebar_state='auto')

with st.sidebar:
    with st.spinner("Loading..."):
        time.sleep(1)
    st.metric("새로 업로드된 게시글", "100 개", "15 개") # 정적임..
    
    st.divider()
    
    st.subheader("Parameter")
    # 해당 기간의 시계열 결과를 보여주고 싶었음! 빼도 상관없어유~
    time_range = st.slider("날짜를 설정하시오", 
                           value=(datetime(2021, 1, 1), datetime(2023, 3, 31)),
                           format="MM/DD/YY")
    
    region = st.selectbox("지역을 선택하세요", 
                          ("서울", "부산", "대구"))
    



col1, col2 = st.columns([3,2])
with col1:
    st.title("Map")
    # Add code

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
        # Add code
