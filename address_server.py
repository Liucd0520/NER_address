from transformers import BertTokenizer, BertForTokenClassification, BertTokenizerFast
from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification, pipeline
import torch
import tqdm
import time
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import List

app = FastAPI()



class SendFilenamesItem(BaseModel):
    
    raw_info: str = "昨天我家门口出现一只猴子，我是住在秀沿西路123号滨江壹号小区的，希望尽快妥善处理"


    version: str = 'Address 1.0'


def model_infer(input_text:str = '', model_dir:str = None,  num_labels:int = 3):
    """
    针对一个输入的情况，后续再扩展为batch推理
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForTokenClassification.from_pretrained(model_dir,  num_labels=num_labels)  
    model = model.to(device)


    # 使用分词器对输入进行编码
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = inputs.to(device)

    # 设置模型为评估模式（不启用训练模式）
    model.eval()

    # 运行模型进行预测
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取预测结果
    predictions = torch.argmax(outputs.logits, dim=2).tolist()
    prediction = predictions[0]
    indices = [index for index, value in enumerate(prediction[1: -1]) if value == 1 or value == 2]
    entity = ''.join([input_text[idx] for idx in indices])

    return entity

@app.post('/parse_address/', name='地址实体抽取')
async def address_parser(request: SendFilenamesItem):

    input_text = request.raw_info
    
    try:
        address = model_infer(input_text, model_dir=model_dir)
        status_code = 200
    except:
        address = ''
        status_code = 477

    return {'status_code': status_code, 'address': address}



if __name__ == '__main__':
    model_dir = 'NER_model'

    uvicorn.run(app=app, host='0.0.0.0', port=80)
