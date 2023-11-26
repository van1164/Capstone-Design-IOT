from transformers import BertForMaskedLM

model = BertForMaskedLM.from_pretrained("bert_model")
model.eval()
print(model.eval())