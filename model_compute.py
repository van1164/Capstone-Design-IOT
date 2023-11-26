from transformers import BertForMaskedLM
from transformers import BertTokenizer, BertForSequenceClassification
import torch
model = BertForMaskedLM.from_pretrained("bert_model")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert_model2', num_labels=2).to('cuda')
input_text = "iptables: denied incoming connection from 192.168.0.3 to port 22 Nov 20 12:46:01 server kernel: [1234567.891] iptables: blocked outgoing connection to 8.8.8.8 on port 53"

# 텍스트를 토큰화하고 모델 입력 형식으로 변환
encoding = tokenizer.encode_plus(
    input_text,
    add_special_tokens=True,
    max_length=128,
    return_token_type_ids=False,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt'
)

# 모델 예측 수행
model.eval()
with torch.no_grad():
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

# 로짓에서 예측값 얻기
predicted_class = torch.argmax(logits, dim=1).item()

# 예측 결과 출력
if predicted_class == 1:
    print(input_text)
    print("침투 흔적 발견")
else:
    print("문제 없음")
