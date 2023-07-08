import re

# 전처리 함수       
def preprocessing_data(content):
    pattern = '[^가-힣A-Za-z0-9.-/:\']'
    content = re.sub(pattern=pattern, repl = ' ', string=content)      # 이모지, 특수기호 제거
    content = content.replace('\s+', ' ')   # 중복 띄어쓰기, 줄바꿈 제거
    content = re.sub('ه ه [-=+,#\?^*\"※`~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]ܤه  ه ', ' ', content)
    content = content.replace("'", "")
    content = ' '.join(content.split())
    content = re.sub(u"[àáâãäåạ]", 'a', content)
    content = re.sub(u"[èéêë]", 'e', content)
    content = re.sub(u"[ìíîï]", 'i', content)
    content = re.sub(u"[òóôõö]", 'o', content)
    content = re.sub(u"[ùúûü]", 'u', content)
    content = re.sub(u"[ýÿ]", 'y', content)
    content = re.sub(u"[ß]", 'ss', content)
    content = re.sub(u"[ñ]", 'n', content)
    content = re.sub(u"[➍④]", '4', content)
    content = re.sub(u"𝒂𝒕𝒐", 'ato', content)
    content = re.sub(u"Аҳŗịṅḡ", 'axring', content)
    content = re.sub(u"аҳŗịṅḡ", 'axring', content)
    pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)' # E-mail제거
    content = re.sub(pattern=pattern, repl='', string=content)
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+' # URL제거
    content = re.sub(r'http\S+', repl='', string=content)
    pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
    content = re.sub(pattern=pattern, repl='', string=content)
    pattern = '<[^>]*>'         # HTML 태그 제거
    content = re.sub(pattern=pattern, repl='', string=content)
    pattern = re.compile(r"(?<=\b\w)\s(?=\w\b)") # 띄어쓰기 된  아이디 결합
    content = re.sub(pattern=pattern, repl='', string=content)
    
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', } 
    searchfilter_mapping = {"ㅌ ㄹ ㄱ ㄹ": "텔레그램", "ㅌ ㄹ ㄱ ㄹ": "텔레그램","ㅌㄹ": "텔레", "ㅌ ㄹ": "텔레", "ㅌㄹㄱㄹ": "텔레그램", "텔 레": "텔레", "그 램":"그램", "텔레 그램":"텔레그램","문 의":"문의","ka톡":"카톡", "ka 톡": "카톡", "ｋa톡":"카톡", "카 카 오 톡":"카카오톡", "gt":"", "lt":"", "p i n k m a n 1 1 4": "pinkman114", "F a s t i c e119": "Fastice119"}
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': '', 'ه ':'', '∞': ''}
    for p in punct_mapping:
        content = content.replace(p, punct_mapping[p])
    for s in searchfilter_mapping:
        content = content.replace(s, searchfilter_mapping[s])
    for s in specials:
        content = content.replace(s, specials[s])
        
    return content