import pandas as pd
import numpy as np
from collections import Counter    # 배열에서 각 원소 몇번 나오는지 알려줌
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, AlbertModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# training set, test set 분리
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm  # Progress Bar 출력


# 디바이스 설정
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')
    
# 데이터 불러오기
data = pd.read_csv('/home/pbl13/team_yj/total_labeling_preprocessed.csv', encoding='utf-8')
print('데이터프레임 csv 불러오기 완료\n')

# 텐서보드 사용하기 
writer = SummaryWriter()

# 하이퍼파라미터
batch_size = 32
num_workers = 8
learning_rate = 1e-3
num_epochs = 50
model_name = 'albert-kor-base'     # checkpoint로 저장할 모델의 이름
CHECKPOINT_NAME = 'kykim/albert-kor-base'
ckpt_path = 'ckpt_ver5'
print(f'\nTraining Option: \nLearning Rate: {learning_rate}\nBATCH_SIZE: {batch_size}\nMODEL NAME: {CHECKPOINT_NAME}\nEpochs: {num_epochs}\nCheckpoint directory:{ckpt_path}\n')
data_size = len(data)
train_size = int(data_size * 0.7)
test_size = data_size - train_size
train_set, test_set = train_test_split(data, test_size=0.3, shuffle=True, stratify=data['label'])

print(f"전체 데이터셋 크기: {data_size}")


# 토크나이저 관련 경고 무시하기 위하여 설정
os.environ["TOKENIZERS_PARALLELISM"] = 'true'

class TokenDataset(Dataset):
    def __init__(self, dataframe, tokenizer_pretrained):
        # sentence, label 컬럼으로 구성된 데이터프레임 전달
        self.data = dataframe        
        # Huggingface 토크나이저 생성
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_pretrained)
        
    def __len__(self):
        return len(self.data) 
        
    def __getitem__(self, idx):
        sentence = str(self.data.iloc[idx]['content'])
        label = self.data.iloc[idx]['label'] 
        
        # 토큰화 처리
        tokens = self.tokenizer(
            sentence,                # 1개 문장 
            return_tensors='pt',     # 텐서로 반환
            truncation=True,         # 잘라내기 적용
            padding='max_length',    # 패딩 적용
            add_special_tokens=True  # 스페셜 토큰 적용
        )

        input_ids = tokens['input_ids'].squeeze(0)           # 2D -> 1D
        attention_mask = tokens['attention_mask'].squeeze(0) # 2D -> 1D
        token_type_ids = torch.zeros_like(attention_mask)

        # input_ids, attention_mask, token_type_ids 이렇게 3가지 요소를 반환하도록 합니다.
        # input_ids: 토큰
        # attention_mask: 실제 단어가 존재하면 1, 패딩이면 0 (패딩은 0이 아닐 수 있습니다)
        # token_type_ids: 문장을 구분하는 id. 단일 문장인 경우에는 전부 0
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask, 
            'token_type_ids': token_type_ids,
        }, torch.tensor(label)
        
train_set = TokenDataset(train_set, CHECKPOINT_NAME)
test_set = TokenDataset(test_set, CHECKPOINT_NAME)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
print('데이터 토크나이징 완료\n')
# 텍스트 분류 모델 생성하기
class CustomAlbertModel(nn.Module):
    def __init__(self, CHECKPOINT_NAME, dropout_rate=0.5):
        super(CustomAlbertModel, self).__init__()

        # 모델 지정
        self.bert = AlbertModel.from_pretrained(CHECKPOINT_NAME)
        
        #dropout 설정
        self.dropout = nn.Dropout(p=dropout_rate)
        
        #최종 출력층
        self.fc = nn.Linear(768, 2)
        
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 결과의 last_hidden_state 가져옴
        last_hidden_state = output['last_hidden_state']
        # last_hidden_state[:, 0, :]는 [CLS] 토큰을 가져옴
        x = self.dropout(last_hidden_state[:, 0, :])
        # FC 을 거쳐 최종 출력
        x = self.fc(x)
        return x
    
# 모델 클래스 인스턴스 생성
albert_model = CustomAlbertModel(CHECKPOINT_NAME)
albert_model.to(device)

# loss랑 optimizer 정의
loss = nn.CrossEntropyLoss()
opt = optim.Adam(albert_model.parameters(), lr=learning_rate)

# train 함수 정의
def model_train(model, data_loader, loss_fn, optimizer, device):
    # 모델을 훈련모드로 설정합니다. training mode 일 때 Gradient 가 업데이트 됩니다. 반드시 train()으로 모드 변경을 해야 합니다.
    model.train()
    
    # loss와 accuracy 계산을 위한 임시 변수 입니다. 0으로 초기화합니다.
    running_loss = 0
    corr = 0
    counts = 0
    
    # 예쁘게 Progress Bar를 출력하면서 훈련 상태를 모니터링 하기 위하여 tqdm으로 래핑합니다.
    prograss_bar = tqdm(data_loader, unit='batch', total=len(data_loader), mininterval=1)
    
    # mini-batch 학습을 시작합니다.
    for idx, (inputs, labels) in enumerate(prograss_bar):
        # inputs, label 데이터를 device 에 올립니다. (cuda:0 혹은 cpu)
        inputs = {k:v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        output = model(**inputs)
        
        loss = loss_fn(output, labels)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # output의 max(dim=1)은 max probability와 max index를 반환합니다.
        # max probability는 무시하고, max index는 pred에 저장하여 label 값과 대조하여 정확도를 도출합니다.
        _, pred = output.max(dim=1)
        
        # pred.eq(lbl).sum() 은 정확히 맞춘 label의 합계를 계산합니다. item()은 tensor에서 값을 추출합니다.
        # 합계는 corr 변수에 누적합니다.
        corr += pred.eq(labels).sum().item()
        counts += len(labels)
        
        # loss 값은 1개 배치의 평균 손실(loss) 입니다. img.size(0)은 배치사이즈(batch size) 입니다.
        # loss 와 img.size(0)를 곱하면 1개 배치의 전체 loss가 계산됩니다.
        # 이를 누적한 뒤 Epoch 종료시 전체 데이터셋의 개수로 나누어 평균 loss를 산출합니다.
        running_loss += loss.item() * labels.size(0)
        
        # 프로그레스바에 학습 상황 업데이트
        prograss_bar.set_description(f"training loss: {running_loss/(idx+1):.5f}, training accuracy: {corr / counts:.5f}")
        
    # 누적된 정답수를 전체 개수로 나누어 주면 정확도가 산출됩니다.
    acc = corr / len(data_loader.dataset)
    
    # 평균 손실(loss)과 정확도를 반환합니다.
    # train_loss, train_acc
    return running_loss / len(data_loader.dataset), acc

# evaluation 함수 정의 

def model_evaluate(model, data_loader, loss_fn, device):
    # model.eval()은 모델을 평가모드로 설정을 바꾸어 줍니다. 
    # dropout과 같은 layer의 역할 변경을 위하여 evaluation 진행시 꼭 필요한 절차 입니다.
    model.eval()
    
    with torch.no_grad():
        # loss와 accuracy 계산을 위한 임시 변수 입니다. 0으로 초기화합니다.
        corr = 0
        running_loss = 0
        
        # 배치별 evaluation을 진행합니다.
        for inputs, labels in data_loader:
            # inputs, label 데이터를 device 에 올립니다. (cuda:0 혹은 cpu)
            inputs = {k:v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            
            # 모델에 Forward Propagation을 하여 결과를 도출합니다.
            output = model(**inputs)
            
            # output의 max(dim=1)은 max probability와 max index를 반환합니다.
            # max probability는 무시하고, max index는 pred에 저장하여 label 값과 대조하여 정확도를 도출합니다.
            _, pred = output.max(dim=1)
            
            # pred.eq(lbl).sum() 은 정확히 맞춘 label의 합계를 계산합니다. item()은 tensor에서 값을 추출합니다.
            # 합계는 corr 변수에 누적합니다.
            corr += torch.sum(pred.eq(labels)).item()
            
            # loss 값은 1개 배치의 평균 손실(loss) 입니다. img.size(0)은 배치사이즈(batch size) 입니다.
            # loss 와 img.size(0)를 곱하면 1개 배치의 전체 loss가 계산됩니다.
            # 이를 누적한 뒤 Epoch 종료시 전체 데이터셋의 개수로 나누어 평균 loss를 산출합니다.
            running_loss += loss_fn(output, labels).item() * labels.size(0)
        
        # validation 정확도를 계산합니다.
        # 누적한 정답숫자를 전체 데이터셋의 숫자로 나누어 최종 accuracy를 산출합니다.
        acc = corr / len(data_loader.dataset)
        
        # 결과를 반환합니다.
        # val_loss, val_acc
        return running_loss / len(data_loader.dataset), acc
    
min_loss = np.inf

# Epoch 별 훈련 및 검증을 수행합니다.
for epoch in range(num_epochs):
    torch.save(albert_model.state_dict(), f'{ckpt_path}/{model_name}_initial.pth')

    train_loss, train_acc = model_train(albert_model, train_loader, loss, opt, device)

    # 검증 손실과 검증 정확도를 반환 받습니다.
    val_loss, val_acc = model_evaluate(albert_model, test_loader, loss, device)   
    
    # val_loss 가 개선되었다면 min_loss를 갱신하고 model의 가중치(weights)를 저장합니다.
    if val_loss < min_loss:
        print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!')
        min_loss = val_loss
        torch.save(albert_model.state_dict(), f'{ckpt_path}/{model_name}_{epoch}.pth')
    
    
    # Epoch 별 결과를 출력합니다. 
    print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/validation",val_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/validation", val_acc, epoch)

print("")
print("Training complete!")
writer.close()
    
# # 학습 실행
# training_loop()