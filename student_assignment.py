import datetime
import chromadb
import traceback

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

import pandas as pd

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

def generate_hw01():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    if collection.count() != 0:
        return collection
    # 讀取 CSV 檔案
    df = pd.read_csv("COA_OpenData.csv")

    # 將資料轉換為 ChromaDB 可以接受的格式
    documents = []
    metadatas = []
    ids = []

    # 處理每一條資料並儲存
    for idx, row in df.iterrows():
        # 擷取 Metadata
        metadata = {
            "file_name": "COA_OpenData.csv",
            "name": row["Name"],
            "type": row["Type"],
            "address": row["Address"],
            "tel": row["Tel"],
            "city": row["City"],
            "town": row["Town"],
            "date": int(datetime.strptime(row["CreateDate"], "%Y-%m-%d").timestamp())  # 將 CreateDate 轉為時間戳
        }

        # 擷取 HostWords 作為文件內容
        document = row["HostWords"]
        
        # 存入資料
        documents.append(document)
        metadatas.append(metadata)
        ids.append(str(idx))  # 每條資料的唯一 ID
    # 將資料寫入 ChromaDB
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    return collection

    
def generate_hw02(question, city, store_type, start_date, end_date):
    collection = generate_hw01()
    
    query_text = question
    query_city = city
    query_store_type = store_type
    query_start_date = start_date
    query_end_date = end_date

    start_timestamp = int(query_start_date.timestamp())
    end_timestamp = int(query_end_date.timestamp())

    where_conditions = []

    if query_city:
        where_conditions.append({"city": {"$in": query_city}})
    if query_store_type:
        where_conditions.append({"type": {"$in": query_store_type}})
    if query_start_date:
        start_timestamp = int(query_start_date.timestamp())
        where_conditions.append({"date": {"$gte": start_timestamp}})

    if query_end_date:
        end_timestamp = int(query_end_date.timestamp())
        where_conditions.append({"date": {"$lte": end_timestamp}})

    if len(where_conditions) > 1:
        query_where = {
            "$and": where_conditions
        }
    else:
        query_where = where_conditions[0]

    print(query_where)
    print(type(query_where))
    results = collection.query(
        query_texts=query_text,
        n_results=10,
        include=["metadatas", "distances"],
        where=query_where
    )

    distances = results['distances'][0]
    metadatas = results['metadatas'][0]

    filtered_results = [
        {
            'store_name': meta['name'],
            'similarity_score': distance
        }
        for distance, meta in zip(distances, metadatas)
        if 1 - distance > 0.80
    ]

    sorted_filtered_results = sorted(filtered_results, key=lambda x: x['similarity_score'], reverse=False)

    store_names = [result['store_name'] for result in sorted_filtered_results]

    return(store_names)
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    collection = generate_hw01()
    
    query_text = question
    query_store_name = store_name
    query_new_store_name = new_store_name
    query_city = city
    query_store_type = store_type

    where_conditions = []

    if query_city:
        where_conditions.append({"city": {"$in": query_city}})
    if query_store_type:
        where_conditions.append({"type": {"$in": query_store_type}})

    if len(where_conditions) > 1:
        query_where = {
            "$and": where_conditions
        }
    else:
        query_where = where_conditions[0]

    results = collection.query(
        query_texts=query_text,
        n_results=10,
        include=["metadatas", "distances"],
        where=query_where
    )

    distances = results['distances'][0]
    metadatas = results['metadatas'][0]

    filtered_results = [
        {
            'store_name': meta['name'],
            'similarity_score': distance,
            'new_store_name': f'{query_new_store_name}' if meta['name'] == query_store_name else ''
        }
        for distance, meta in zip(distances, metadatas)
        if 1 - distance > 0.80
    ]

    sorted_filtered_results = sorted(filtered_results, key=lambda x: x['similarity_score'], reverse=False)

    store_names = [result['store_name'] for result in sorted_filtered_results]

    print(store_names)

    for result in sorted_filtered_results:
        if result['new_store_name']:
            for meta in metadatas:
                if meta['name'] == result['store_name']:
                    meta.update({'new_store_name': result['new_store_name']})

    store_names = [result['new_store_name'] if result['new_store_name'] else result['store_name'] for result in sorted_filtered_results]
    return(store_names)
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection
