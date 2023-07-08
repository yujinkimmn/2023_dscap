import torch
import io
import joblib     
import pymysql
import numpy as np
from transformers import BertTokenizer
from transformers import BertTokenizerFast, AlbertModel
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorflow.keras.utils import pad_sequences
import query
from train_albert import CustomAlbertModel
import torch.nn.functional as F

# 디바이스 설정
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')
    
# 학습된 모델 체크포인트로 불러오기
CHECKPOINT_NAME = 'kykim/albert-kor-base'   # hugging face에서 사용할 모델 이름
model_directory = 'ver5_albert-kor-base_34.pth'  # 사용할 모델 체크포인트 경로

loaded_model = CustomAlbertModel(CHECKPOINT_NAME).to(device)
model_state_dict = torch.load(model_directory, map_location=device)
loaded_model.load_state_dict(model_state_dict)


# 분류 위한 클래스
class CustomPredictor():
    def __init__(self, model, tokenizer, labels: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels
        
    def predict(self, sentence):
        # 토큰화 처리
        tokens = self.tokenizer(
            sentence,                # 1개 문장 
            return_tensors='pt',     # 텐서로 반환
            truncation=True,         # 잘라내기 적용
            padding='max_length',    # 패딩 적용
            add_special_tokens=True  # 스페셜 토큰 적용
        )
        tokens.to(device)
        prediction = self.model(**tokens)
        prediction = F.softmax(prediction, dim=1)
        label = prediction.argmax(dim=1).item()
        prob, result = prediction.max(dim=1)[0].item(), self.labels[label]
        #print(f'입력 데이터: [{sentence}\n[{result}]\n확률은 {prob*100:.2f}% 입니다.')

        return label, round(prob*100,2), result

labels = {
    0: '마약 거래 게시글이 아닙니다.',
    1: '마약 거래 게시글이 입니다.'
}

# BERT 토크나이저
tokenizer = BertTokenizerFast.from_pretrained(CHECKPOINT_NAME)
# 클래스 인스턴스 생성
predictor = CustomPredictor(loaded_model, tokenizer, labels)

def test_sentences(sentences):
    loaded_model.eval()
    label, prob, result = predictor.predict(sentences)
    return label, prob, result
    
# # 입력 데이터 변환
# def convert_input_data(sentences):
    
#     # BERT의 토크나이저로 문장을 토큰으로 분리
#     tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    
#     # 입력 토큰의 최대 시퀀스 길이
#     MAX_LEN = 128

#     # 토큰을 숫자 인덱스로 변환
#     input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    
#     # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
#     input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

#     # 어텐션 마스크 초기화
#     attention_masks = []

#     # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
#     # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
#     for seq in input_ids:
#         seq_mask = [float(i>0) for i in seq]
#         attention_masks.append(seq_mask)

#     # 데이터를 파이토치의 텐서로 변환
#     inputs = torch.tensor(input_ids)
#     masks = torch.tensor(attention_masks)

#     return inputs, masks


# # 문장 테스트
# def test_sentences(sentences):

#     # 평가모드로 변경
#     loaded_model.eval()

#     # 문장을 입력 데이터로 변환
#     inputs, masks = convert_input_data(sentences)

#     # 데이터를 GPU에 넣음
#     b_input_ids = inputs.to(device)
#     b_input_mask = masks.to(device)
            
#     # 그래디언트 계산 안함
#     with torch.no_grad():     
#         # Forward 수행
#         outputs = loaded_model(b_input_ids, 
#                         token_type_ids=None, 
#                         attention_mask=b_input_mask)

#     # 로스 구함
#     logits = outputs[0]
    
#     # 확률 구하기
#     # prediction = F.softmax(outputs, dim=1)
#     # print(f'prediction: {prediction}')
#     # prediction = prediction.max(dim=1)[0].item()
    
#     # CPU로 데이터 이동
#     logits = logits.detach().cpu().numpy()
#     # print(f'logits 값: {logits}')
#     #print(f'입력 데이터: {sentences}')
    
#     # argmax로 확률 가장 높은 class 저장
#     pred_label = np.argmax(logits)
#     if pred_label == 1:
#         result = f'{prediction*100:.2f}% 확률로 마약 거래 게시글입니다.'
#     else:
#         result = f'{prediction*100:.2f}% 확률로 마약 거래 게시글이 아닙니다.'

#     return pred_label, round(prediction*100,2), result


# pred_label, result = test_sentences(['필로폰삽니다 아이스작대기종류 필로폰삽니다 아이스작대기종류 TEL : MT2438 약사입나다 필독 채널사기조심 꼭 아이디로 문의 최고의 제품 및 서비스로 모시겠습니다 hRHL  '])
# print(pred_label)
# print(result)
# # print(f'logit 결과: {logits}\n')
# # print(f'np.argmax(logit) 결과: {np.argmax(logits)}\n')  # axis=1이면 가로축에서 최대값 인덱스 

############################################################### 
# tweets = query.select_from_db(sql = 'SELECT * FROM twitterdata')

# for _, _, _, _, _, _, _, _, content, _, _, _, _, _, _ in tweets:
#     print(f'입력 데이터: [{content}]')
#     pred_label, prob, result = test_sentences([content])
#     print(f'pred_label: {pred_label}')
#     print(f'{result}\n\n')
