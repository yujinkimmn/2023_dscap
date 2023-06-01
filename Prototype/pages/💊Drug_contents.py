import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import time
from datetime import datetime, date
import pytz
import html
import query 
import re

st.set_page_config(page_title="Check Drug Contents", page_icon="💊", 
                   layout='wide', initial_sidebar_state='collapsed')

@st.cache_data

# Twitter style content
def drug_html(username, userid, content, date, probability, image_url=None):
    html_code = f'''
    <style>
        /* CSS styles */
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f8fa;
        }}
        
        .tweet {{
            margin-bottom: 20px;
            border: 1px solid #ccc;
            padding: 10px;
            background-color: #f9f9f9;
        }}
        
        .username {{
            font-weight: bold;
            color: #1da1f2;
        }}
        
        .content {{
            margin-top: 5px;
        }}
        
         .image {{
            max-width: 30%;
            margin-top: 10px;
        }}
        
        .userid {{
            font-size: 0.8em;
            color: #777;
        }}
        
        .date {{
            font-size: 0.8em;
            color: #777;
        }}
        
        .probability {{
            font-size: 0.8em;
            color: #ff0000;
        }}
    </style>
    
    <div class="tweet">
        <div class="username">{username}</div>
        <div class="userid">@{userid}</div>
        <div class="content">
            {content}
        </div>
        <div class="date">{date}</div>
        <div class="probability"> 마약 거래 게시글일 확률이 {probability}% 입니다. </div>
        {'<img class="image" src="' + image_url + '">' if image_url else ''}
    </div>
    '''
    return html_code

with st.sidebar:
    st.subheader("Parameter")
    # 오늘 날짜로 슬라이더 끝을 설정
    today = date.today()
    today_datetime = datetime.combine(today, datetime.min.time())
    time_range = st.slider("날짜를 설정하시오", 
                            value=(datetime(2021, 1, 1), today_datetime),
                            format="YYYY/MM/DD")
# Match timezone
timezone = pytz.UTC
# start date랑 end date 타입은 datetime
# replace로 timezone offset은 지우기 
start_date = timezone.localize(time_range[0]).replace(tzinfo=None)
end_date = timezone.localize(time_range[1]).replace(tzinfo=None)

# Custom CSS styles
css = '''
<style>
.twitter-icon {
    display: inline-block;
    vertical-align: middle;
    margin-right: 10px;
}
</style>
'''

# Twitter icon HTML code 
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">', unsafe_allow_html=True)
twitter_icon = '<i class="fab fa-twitter"></i>'

# Twitter 아이콘이랑 같이 title 설정하기
st.markdown(f"{css}<h1>{twitter_icon} 마약 거래 Tweet</h1>", unsafe_allow_html=True)

# Loading
with st.spinner("Loading..."):
        time.sleep(0.5)

# DB에서 데이터 가져오기
data = pd.DataFrame(query.select_from_db_to_df('SELECT * FROM classified'))
print("가져온 데이터: ", data)

# Find corresponding contents within time range that user processes
count = 0
for index, row in data.iterrows():
    # row['date]의 타입은 Timestamp -> datetime object로 바꾸기
    tweet_date = row['date'].to_pydatetime()
    
    if start_date <= tweet_date <= end_date:
        username = html.escape(row['user_displayname']) if isinstance(row['user_displayname'], str) else ''
        userid = row['user_usename']
        content = row['content']
        prob = row['prob']
        tweet_date = row['date']
        url = None
        
        # url이 존재하면 가져오기
        if isinstance(row['media'], str):
            url_match = re.search(r"fullUrl='(.*?)'", row['media'])
            if url_match:
                url = url_match.group(1)

        # Generate the HTML using the drug_html function
        html_code = drug_html(username, userid, content, tweet_date, prob, url)
        print('html_code: ', html_code)
        # Display the HTML in Streamlit
        st.markdown(html_code, unsafe_allow_html=True)

        count += 1
        if count > 1000:
            break
