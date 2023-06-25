import pymysql

# db 연결 정보
host = '43.200.122.248'
user = 'dscap_db_user'
password='2023Dscyjline!'
db='DScapDB'
port = 56560

# DELETE문
def delete_from_db(sql, val=None):
    conn = pymysql.connect(host=host, port=port, user=user, password=password, db=db, charset='utf8')
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            # 변경사항 저장
            conn.commit()
    finally:
            conn.close()

# SELECT문으로 가져와서 데이터프레임 형태로 저장하기
def select_from_db_to_df(sql, val=None):
    conn = pymysql.connect(host=host, port=port, user=user, password=password, db=db, charset='utf8', cursorclass=pymysql.cursors.DictCursor)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)   # SELECT sql 구문 삽입
            result = cur.fetchall()
    finally:
        conn.close()

    return result

# SELECT문
def select_from_db(sql, val=None):
    conn = pymysql.connect(host=host, port=port, user=user, password=password, db=db, charset='utf8')
    try:
        with conn.cursor() as cur:
            cur.execute(sql, val)   # SELECT sql 구문 삽입
            result = cur.fetchall()
    finally:
        conn.close()

    return result
    
# 중복검사 & INSERT문
def save_to_db(datalist): 
    # DB에서 중복 체크
    for type1, type2, date, id, user_name, user_displayname, place, user_location, content, likeCount, retweetCount, viewCount, hashtags, media, sourceLabel, label, prob in datalist:
        sql = "SELECT EXISTS (SELECT * FROM classified WHERE date=%s AND id=%s)"
        is_exists = select_from_db(sql, (date, id))[0][0]
        
        if is_exists:   # 중복되는 트윗이면 pass
            print(f"중복트윗 발생\n{date}\n{id}\n{content}\n")
            continue
        
        else: 
            conn = pymysql.connect(host=host, port=port, user=user, password=password, db=db, charset='utf8')
            sql = "INSERT INTO classified (type1, type2, date, id, user_usename, user_displayname, place, user_location, content, likeCount, retweetCount, viewCount, hashtags, media, sourceLabel, label, prob) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            try:
                with conn.cursor() as cur:
                    # db에 저장
                    cur.execute(sql, (type1, type2, date, id, user_name, user_displayname, place, user_location, content, likeCount, retweetCount, viewCount, hashtags, media, sourceLabel, label, prob))
                    print(f"is_exists: {is_exists}, DB에 데이터 저장")
                    print(f"새로운 트윗 저장 완료\ntype1: {type1}\ntype2: {type2}\n{date}\n{id}\n{content}\n")
                    # 변경사항 저장
                    conn.commit()
            except pymysql.err.DataError:
                for i in (type1, type2, date, id, user_name, user_displayname, place, user_location, content, likeCount, retweetCount, viewCount, hashtags, media, sourceLabel, label, prob):
                    if type(i) == list: 
                        print(f"DataError 발생\n{i}\n")
            except pymysql.err.OperationalError:
                for i in (type1, type2, date, id, user_name, user_displayname, place, user_location, content, likeCount, retweetCount, viewCount, hashtags, media, sourceLabel, label, prob):
                    if type(i) == list: 
                        print(f"OperationalError 발생\n{i}\n")
            finally:
                conn.close()