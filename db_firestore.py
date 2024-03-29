from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
import firebase_admin
from firebase_admin import credentials, storage
import json, os, dotenv
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
try:
    os.environ["FIREBASE_CREDENTIAL"] = dotenv.get_key(dotenv.find_dotenv(), "FIREBASE_CREDENTIAL")
    if os.environ.get["FIREBASE_CREDENTIAL"] == None:
        raise TypeError
except TypeError:
    import streamlit as st
    os.environ["FIREBASE_CREDENTIAL"] = st.secrets["FIREBASE_CREDENTIAL"]
cred = credentials.Certificate(json.loads(str(os.environ.get("FIREBASE_CREDENTIAL")), strict=False))

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {'storageBucket': 'healthhack-store.appspot.com'})

def get_csv(index_name, isDiseases = True):
    while index_name[-1]=="/":
        index_name = index_name[:-1]
    dir = index_name.split("/")
    
    ## Check if path exists locally
    for i in range(len(dir)):
        path = '/'.join(dir[:i+1])
        if not os.path.exists(path):
            os.mkdir(path)

    ## Check if file exists locally, get from blob
    if (not os.path.exists(index_name+"/diseases.csv") or
        not os.path.exists(index_name+"/network.csv")
        ):
        bucket = storage.bucket()
        blob = bucket.blob(f"{index_name}/diseases.csv")
        blob.download_to_filename(f"{index_name}/diseases.csv")
        bucket = storage.bucket()
        blob = bucket.blob(f"{index_name}/network.csv")
        blob.download_to_filename(f"{index_name}/network.csv")

    ## load store from local
    if isDiseases:
        store = pd.read_csv(f"{index_name}/diseases.csv")
        return store
    else:
        store = pd.read_csv(f"{index_name}/network.csv")
        return store

def get_store(index_name, embeddings = None):
    while index_name[-1]=="/":
        index_name = index_name[:-1]
    dir = index_name.split("/")
    
    ## Check if path exists locally
    for i in range(len(dir)):
        path = '/'.join(dir[:i+1])
        if not os.path.exists(path):
            os.mkdir(path)

    ## Check if file exists locally, get from blob
    if (not os.path.exists(index_name+"/index.faiss") or
        not os.path.exists(index_name+"/index.pkl")
        ):
        bucket = storage.bucket()
        blob = bucket.blob(f"{index_name}/index.pkl")
        blob.download_to_filename(f"{index_name}/index.pkl")
        bucket = storage.bucket()
        blob = bucket.blob(f"{index_name}/index.faiss")
        blob.download_to_filename(f"{index_name}/index.faiss")
    
    ## check embeddings, default to BGE
    if embeddings is None:
        model_name = "bge-large-en-v1.5"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceBgeEmbeddings(
            # model_name=model_name, 
            model_kwargs = model_kwargs,
            encode_kwargs = encode_kwargs)

    ## load store from local
    store = FAISS.load_local(index_name, embeddings)
    return store

def update_csv_from_local(index_name):
    while index_name[-1]=="/":
        index_name = index_name[:-1]
    pathdir = index_name.split("/")
    
    ## Check if path exists locally
    for i in range(len(pathdir)):
        path = '/'.join(pathdir[:i+1])
        if not os.path.exists(path):
            raise Exception("Index name does not exist locally")

    if (not os.path.exists(index_name+"/diseases.csv") or
        not os.path.exists(index_name+"/network.csv")
        ):
        raise("Index is missing some files (network.csv, diseases.csv)")
    
    ## Update store
    bucket = storage.bucket()
    blob = bucket.blob(index_name+"/diseases.csv")
    blob.upload_from_filename(index_name+"/diseases.csv")
    blob = bucket.blob(index_name+"/network.csv")
    blob.upload_from_filename(index_name+"/network.csv")
    return True

def update_store_from_local(index_name):
    while index_name[-1]=="/":
        index_name = index_name[:-1]
    pathdir = index_name.split("/")
    
    ## Check if path exists locally
    for i in range(len(pathdir)):
        path = '/'.join(pathdir[:i+1])
        if not os.path.exists(path):
            raise Exception("Index name does not exist locally")

    ## Check if file exists locally, get from blob
    if (not os.path.exists(index_name+"/index.faiss") or
        not os.path.exists(index_name+"/index.pkl")
        ):
        raise("Index is missing some files (index.faiss, index.pkl)")
    
    ## Update store
    bucket = storage.bucket()
    blob = bucket.blob(index_name+"/index.faiss")
    blob.upload_from_filename(index_name+"/index.faiss")
    blob = bucket.blob(index_name+"/index.pkl")
    blob.upload_from_filename(index_name+"/index.pkl")
    return True
    
    

if __name__ == "__main__":
    print("y r u running dis")