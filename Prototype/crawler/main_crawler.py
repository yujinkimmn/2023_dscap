import pandas as pd
import re
from tqdm.notebook import tqdm
import snscrape.modules.twitter as sntwitter 
import pymysql
from datetime import datetime, timedelta
import schedule
import time
import query 
import classify_with_model
import preprocessing

# 단어 리스트
search_keyword_list = {
            '기타': ['드라퍼', '브액', '아이스드랍', '주사기', '클럽약', '텔레', '파티약', '허브', '후리베이스'],
            '대마': ['대마초', '사티바', '인디카', '합성대마', '해시시'],
            '메스암페타민': ['사끼', '차가운술', '작대기', '떨액', '크리스탈', '삥두', '시원한술', '필로폰', '아이스술', '히로뽕', '액상떨', '아이스', '북한산아이스', '빙두', '찬술', '샤부'],
            '사일로시빈': ['마법의버섯', '환각버섯'],
            '아편': ['스틸녹스', '신의눈물', '에토미데이트', '옥시코돈', '졸피뎀', '트라마돌'],
            '알킬니트라이트': ['러쉬파퍼', '랏슈', '정글주스'],
            '케타민': ['캔디케이', '케타민'],
            '코카인': ['서울코카인', '충북코카인', '충남코카인', '강원코카인', '경기코카인', '전북코카인', '전남코카인', '경북코카인', '경남코카인', '제주코카인', '강남코카인', '부산코카인', '인천코카인'],
            'GHB': ['물뽕', '발정제', '최음제'],
            'LSD': ['엘에스디'],
            'MDMA': ['엑스터시', '서울도리도리', '충북도리도리', '충남도리도리', '강원도리도리', '경기도리도리', '전북도리도리', '전남도리도리', '경북도리도리', '경남도리도리', '제주도리도리', '강남도리도리', '부산도리도리', '인천도리도리', '서울몰리', '충북몰리', '충남몰리', '강원몰리', '경기몰리', '전북몰리', '전남몰리', '경북몰리', '경남몰리', '제주몰리', '강남몰리', '부산몰리', '인천몰리']
             }
    
except_words = ['허브맛', '허브맛쿠키', '허브솔트', '스파허브', '아이허브', '미국', '대회', 'F1', '유아인', '휘성', '검찰', '해시브라운', '시간', '웃어', '웃으', '시시해', \
            '에프엑스', 'fx', '정수정', '크리스탈라이트', '제시카', \
            '아이스베어', '아이스탕후루', '아이스만주', '아이스만쥬', '아메리카노', '얼죽아', '블랙아이스', '아이스크림', '초코', '커피', '카페', '아이스께끼', '찰떡', '아이스티', '겨울', '라떼', '에스프레소', '하키', '팝업', '주문', '당첨', '블렌드', '블렌디드', '바닐라', '헤이즐넛', '모찌', '케이크', '음료', '콜드브루', '프라푸치노', '엔시티', '스톰', '아이스맨', '매브', '매버릭', \
            '남경필', '한서희', '브레이킹 배드', '돈스파이크', \
            '브레이킹 배드', \
            '샤브샤브', '샤브', \
            '오마이걸 유빈', \
            'PD수첩', '히어로물뽕', '홍준표', '돼지', \
            '몰리면', '홀리몰리', '홀리 몰리', '과카몰리', '몰리게', '내몰리', '몰리는', '미스몰리', \
            '엑스토시움', \
            '유아인', '허성태', '코카인댄스', \
            '머쉬룸 스프', '머쉬룸스프', '수프', '버거', '파스타', '맛집', '표고버섯', '치즈', '피자', \
            '양지원', \
            '의사', '병원', '처방받', '처방 받', '졸피뎀과 나', '처벌', '구속', '불면', \
            '정글쥬스', \
            '전두환 손자', '전우원', '돈스파이크', '유아인', \
            '병원','여드름','뾰루지','얼굴','흉터','흉','상처',\
            '라이브','북클럽','콘서트','팬미팅','팬클럽','공연','대리', '음향', '춤', '50억', '비리', '수사', '대리티켓팅', '50억클럽', '멜론', '수작', '냄새', '웹툰', '게임'
            ]

#####################################################################################

def crawl_for_period(
    type: str,
    search_query: str,
    start_date: str,
    end_date: str,
    except_words: list  # 제외어 리스트 
    ):
    
    query = str(search_query) + " since:" + str(start_date) + " until:" + str(end_date)
    print(f"검색 query: {query}")

    # 트위터 데이터 저장할 리스트
    tweets_list = []
    
    for i, tweet in (enumerate(sntwitter.TwitterSearchScraper(query).get_items())): 
        # 수집할 데이터 컬럼
        data = [
            type, 
            search_query,
            tweet.date, 
            tweet.id,
            tweet.user.username,
            tweet.user.displayname,
            tweet.place,
            tweet.user.location,
            tweet.content,     
            tweet.likeCount,
            tweet.retweetCount, 
            tweet.viewCount,
            tweet.hashtags,
            tweet.media, 
            tweet.sourceLabel
        ]

        # 트윗 내용에 제외어 하나라도 포함시 제외하기
        if any(words in tweet.content for words in except_words):
            continue
        
        # 리트윗 데이터는 제외하기 (ex. @닉네임)
        regex = re.compile("@[\w_]+")
        if regex.search(tweet.content):
            continue 
        
        # tweet.content 전처리
        data[8] = preprocessing.preprocessing_data(tweet.content)
        
        # media, hashtag 리스트를 string으로 변환해서 저장하기
        if isinstance(data[-2], list): 
            data[-2] = str(data[-2])
        if isinstance(data[-3], list): 
            data[-3] = str(data[-3])
            
        # 모델이 content에 대해서 classification 수행
        pred_label, prediction, _ = classify_with_model.test_sentences(data[8])
        
        # 마약 거래 게시글이 맞으면 tweets_list에 넣기
        if pred_label == 1: 
            data.append(pred_label)
            data.append(prediction)
            tweets_list.append(data)        
    print(f'[{search_query}] 키워드에 대해서 분류된 트윗 개수: {len(tweets_list)}')
    return tweets_list

def search_twitter():
    print(f'{datetime.today()} 작업 실행')
    
    # DB에 3일간의 데이터만 저장하기 위해 기존 데이터 지우기
    query.delete_from_db('DELETE FROM classified')
    search_query = ''
    
    # 3일간의 데이터 추적
    start_date= (datetime.today() -timedelta(3)).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    
    # 키워드 하나씩 db에 저장
    for type1, type2 in search_keyword_list.items():
        for t in type2:
            print(f'type1: {type1}, type2: {t}')
            search_query = t
            tweets_list = crawl_for_period(type1, search_query, start_date, end_date, except_words)            
            # mysql DB에 저장
            if len(tweets_list) > 0:
                query.save_to_db(tweets_list)
                print(f"\n ------------------- 총 {len(tweets_list)}개 게시글 저장 완료 -------------------\n\n")
# main 실행
search_twitter()

