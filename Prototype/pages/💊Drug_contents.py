import streamlit as st
import pandas as pd
import time

fp = "/Users/stella/github/2023_dscap/twitterdata/labeling/total_labeling_preprocessed.csv"

st.set_page_config(page_title="Check Drug Contents", page_icon="💊", 
                   layout='wide', initial_sidebar_state='collapsed')

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

st.title("마약 거래 게시글")