import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

path = r''
# sheet = ['Sheet1']
sheet = []
out_data = dict()
for i in range(len(sheet)):
    data = pd.read_excel(path, dtype=str, sheet_name=sheet[i])
    data['new'] = data['年龄'].map(str) + ',' + data['性别'].map(str)
    test = data[['序号', 'new']].set_index('序号').to_dict(orient='dict')['new']
    out_data.update(test)


bert_path = ""
tokenizer = BertTokenizer.from_pretrained(bert_path)
bert = BertModel.from_pretrained(bert_path, return_dict=True)
feature_dict = dict()
for key, value in out_data.items():
    inputs = tokenizer(value, return_tensors='pt')
    outputs = bert(**inputs)
    feature_dict[key] = outputs.last_hidden_state
    print(f'is finishing{key}')


np.save('./ageAddGender_external_hospitalData.npy', feature_dict)
