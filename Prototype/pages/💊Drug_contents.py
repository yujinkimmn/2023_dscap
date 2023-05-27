import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import time
from datetime import datetime
import pytz
import html

fp = "/Users/stella/github/2023_dscap/twitterdata/preprocessing/preprocessed/total_preprocessed_name_revise.csv"

st.set_page_config(page_title="Check Drug Contents", page_icon="💊", 
                   layout='wide', initial_sidebar_state='collapsed')

@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Twitter style content
def drug_html(username, userid, content, date):
    html_code = f'''
    <style>
        /* CSS styles */
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
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
        
        .userid {{
            font-size: 0.8em;
            color: #777;
        }}
        
        .date {{
            font-size: 0.8em;
            color: #777;
        }}
    </style>
    
    <div class="tweet">
        <div class="username">{username}</div>
        <div class="userid">@{userid}</div>
        <div class="content">
            {content}
        </div>
        <div class="date">{date}</div>
    </div>
    '''
    return html_code

with st.sidebar:
    st.subheader("Parameter")
    time_range = st.slider("날짜를 설정하시오", 
                            value=(datetime(2021, 1, 1), datetime(2023, 3, 31)),
                            format="YYYY/MM/DD")

# Match timezone
timezone = pytz.UTC
start_date = timezone.localize(time_range[0])
end_date = timezone.localize(time_range[1])

# Set title
st.title('마약 거래 게시글')

# Loading
with st.spinner("Loading..."):
        time.sleep(0.5)

# Data load
data = load_data(fp)

# Find corresponding contents within time range that user processes
count = 0
for index, row in data.iterrows():
    date = pd.to_datetime(row['date']).tz_localize(None).tz_localize(timezone)
    # try:
    #     date = pd.to_datetime(row['date']).tz_convert(timezone)
    # except:
    #     date = pd.to_datetime(row['date']).tz_localize(None).tz_convert(timezone)
    #date = pd.to_datetime(row['date']).tz_convert(timezone)

    if start_date <= date <= end_date:
        username = html.escape(row['user.displayname']) if isinstance(row['user.displayname'], str) else ''
        userid = row['user.username']
        content = row['content']
        date = row['date']

        # Generate the HTML using the drug_html function
        html_code = drug_html(username, userid, content, date)

        # Display the HTML in Streamlit
        st.markdown(html_code, unsafe_allow_html=True)

        count += 1
        if count > 1000:
            break
