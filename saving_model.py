
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

model.save_pretrained("./model/multilingual-e5-large")

tokenizer.save_pretrained("./models/tokenizer/multilingual-e5-large")
