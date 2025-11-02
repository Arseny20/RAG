
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

model.save_pretrained("/home/sofia/Desktop/Rag_statya/models/multilingual-e5-large")
tokenizer.save_pretrained("/home/sofia/Desktop/Rag_statya/models/tokenizer/multilingual-e5-large")