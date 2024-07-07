from sentence_transformers import SentenceTransformer, util
import tqdm
import time
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import List
import pandas as pd
from utils import get_log

log_dir = './logs'
logger = get_log(log_dir)

app = FastAPI()


def embed_courpus(dataset_path, ):  # embedder

    corpus = []
    with open(dataset_path, encoding='utf8') as fIn:
        for line in tqdm.tqdm(fIn, desc='Read file'):
            line = line.strip()
            corpus.append(line)

    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    return corpus_embeddings, corpus


def records2dict(code_search_result, score, priority):
    """
    code_search_result: 符合raw_code or raw_name的查询结果
    """
    records_list = []

    for idx in range(len(code_search_result)):

        coding_code, coding_item, coding_part, coding_material, coding_specifications, coding_manufactor, coding_name = \
        code_search_result.iloc[idx]
        records_dict = {
            'code': coding_code, 'name': coding_name, 'use': coding_item, 'part': coding_part,
            'material': coding_material, 'specifications': coding_specifications, 'manufactor': coding_manufactor,
            'label': label, 'version': version, 'score': score, 'priority': priority}

        records_list.append(records_dict)

    return records_list


class SendFilenamesItem(BaseModel):
    query_list: List = [
        {
            "raw_code": "C1409020000000009241",
            "raw_name": "创可贴"
        },
        {
            "raw_code": "",
            "raw_name": "核酸提取或纯化试剂"
        },
        {
            "raw_code": "C170",
            "raw_name": "医用口罩",
        }
    ]

    top_k: int = 1
    version: str = '联仁1.0'


@app.post('/standard_coding/consumables/coding_consum/', name='耗材编码')
async def senmantic_search(request: SendFilenamesItem):
    """ 耗材编码[update to 2021-10]"""
    #
    """
    coding_name:  编码后的raw_name
    coding_code: 编码后的raw_code
    records_dict: 针对top_k个输出结果中的一条输出记录
    records_list: top_k个输出记录组成的List
    encode_dict: {'input_text':xx, 'entity_records': xx,  'entity_records': records_list}
    all_records: 所有查询编码的输出encode_dict组成的列表
    return {'status_code': xx, massege: xx, 'data': all_records}
    """


    top_k = request.top_k
    top_k = 1
    version = request.version
    query_list = request.query_list

    logger.info('START ...')
    print(query_list)
    raw_codes = [query['raw_code'] for query in query_list]
    raw_names = [query['raw_name'] for query in query_list]
    logger.info('raw_codes: {}'.format(raw_codes))
    logger.info('raw_name:  {} '.format(raw_names))

    all_records = []
    for raw_name, raw_code in zip(raw_names, raw_codes):

        encode_dict = {}
        encode_dict.update({'input_text': raw_name, 'entity_number': 1})

        records_list = []
        code_search_result = std_data[std_data['std_code'] == raw_code]
        if not code_search_result.empty:  # 如果满足条件的DataFrame不为空，则一切确定
            priority = 2
            score = 100

            code_search_result = std_data[std_data['std_code'] == raw_code]
            records_list = records2dict(code_search_result, score, priority)

        elif raw_name:  # 如果raw_code未匹配到std_code, 同时raw_name不为空，则计算raw_name相似度
            priority = 1

            query_embedding = embedder.encode(raw_name, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
            hits = hits[0]

            coding_name = corpus[hits[0]['corpus_id']]
            score = round(100 * hits[0]['score'])

            code_search_result = std_data[std_data['std_name'] == coding_name]
            records_list = records2dict(code_search_result, score, priority)

        else:  # raw_name为空--> 随便给一组数据

            coding_code = 'C0101010010100704937'
            score = 37
            priority = 4

            code_search_result = std_data[std_data['std_code'] == raw_code]
            records_list = records2dict(code_search_result, score, priority)

        logger.info(records_list)
        encode_dict.update({'entity_records': records_list})
        all_records.append(encode_dict)

    print(all_records)
    logger.info('END')

    return {'status_code': 200, 'message': 'success', 'data': all_records}


if __name__ == '__main__':
    label = '耗材'
    version = '联仁1.0'

    consum_path = './haocai.txt'
    model_path = r'D:\models_list\haocai_model'
    all_df = pd.read_csv('consum_full_2021_10.csv')
    std_data = all_df[['std_code', 'item', 'part', 'material', 'specifications', 'manufactor', 'std_name']]
    print(len(std_data))
    std_data = std_data.drop_duplicates()
    print(len(std_data))
    # std_data
    embedder = SentenceTransformer(model_path)
    start = time.time()
    corpus_embeddings, corpus = embed_courpus(consum_path)
    print(time.time() - start)

    uvicorn.run(app=app, host='0.0.0.0', port=80)

