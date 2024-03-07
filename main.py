import time
from typing import Callable, Union
from fastapi import APIRouter, FastAPI, Request, Response, UploadFile, File, HTTPException, Header, Form
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydub import AudioSegment
from fastapi.routing import APIRoute
from pydantic import BaseModel
import os
import librosa
import numpy as np
import pandas as pd
import pickle
import soundfile as sf
import warnings
import uuid
from dotenv import load_dotenv
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from fastapi.security import APIKeyHeader
from typing import Annotated, AsyncIterator
from starlette.middleware.cors import CORSMiddleware
from datetime import datetime
from fastapi.logger import logger
from typing import List
from starlette.concurrency import iterate_in_threadpool
import json


app = FastAPI()
load_dotenv()

origins = ["https://chordsdotzip.netlify.app"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(os.getenv('MONGO_DB'))
client = MongoClient(os.getenv('MONGO_DB'))
database = client["ChordsdotzipDB"]
collection = database["logging"]

API_KEY:str = os.getenv('API_KEY')

class AuditLog(BaseModel):
    timestamp: str
    total_time: float
    path: str
    method: str
    status_code: int
    client: str
    url:str
    chords: List[List[str]]


async def api_key_middleware(request: Request, API_KEY: str):
    print(request.headers)
    print(API_KEY)
    api_key = request.headers.get('authorization')
    if api_key:
        api_key = api_key.split(" ")[1]
    print(api_key)
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return
  

@app.middleware('http')
async def audit_middleware(request: Request, call_next):
    # x = await api_key_middleware(request,os.getenv('API_KEY'))
    # print('x: ',x)
    start_time = datetime.utcnow()
    response = await call_next(request)
    response_body = [section async for section in response.body_iterator]
    response.body_iterator = iterate_in_threadpool(iter(response_body))
    res_body = response_body[0].decode()
    if res_body[0] == '{':
        res_body = json.loads(res_body)
    
    end_time = datetime.utcnow()
    elapsed_time = (end_time - start_time).total_seconds() * 1000  # Convert to milliseconds
    print('res: ',res_body)
    chords = [[""]]
    url = ""
    if "detail" in res_body:
        if "chords" in res_body['detail']:
            chords = res_body['detail']['chords']
        if "url" in res_body['detail'] :  
            url =  res_body['detail']['url'] 
    else:
        return response
    audit_log = AuditLog(
        timestamp=start_time.isoformat(),
        total_time=elapsed_time,
        path=request.url.path,
        method=request.method,
        status_code=response.status_code,
        client=request.client.host,
        url=url,
        chords=chords,
    )
    collection.insert_one(audit_log.dict())
    return response



def split_audio(file_path: str, segment_length_ms: int = 500):
    x, sr = librosa.load(file_path)
    segments = [x[i:i + int(sr * segment_length_ms / 1000)] for i in range(0, len(x), int(sr * segment_length_ms / 1000))]
    return segments, sr

def create_chroma_features(segments,sr):
    chroma_features = []
    for i, segment in enumerate(segments):
        chroma = librosa.feature.chroma_cqt(y=segment, sr=sr)
        arr = []
        for i in range(len(chroma)):
            chroma[i] = sum(chroma[i])/len(chroma[i])
            arr.append(chroma[i][0])
        chroma_features.append(arr)
    return chroma_features

def count_contiguous_chords(chords):
    chord_counts = []
    current_chord = None
    count = 0

    for chord in chords:
        if chord == current_chord:
            count += 1
        else:
            if current_chord is not None:
                chord_counts.append([current_chord, count])
            current_chord = chord
            count = 1

    # Add the last chord and count
    if current_chord is not None:
        chord_counts.append([current_chord, count])

    return chord_counts






# class TimedRoute(APIRoute):
#     def get_route_handler(self) -> Callable:
#         original_route_handler = super().get_route_handler()

#         async def custom_route_handler(request: Request) -> Response:
#             before = time.time()
#             response: Response = await original_route_handler(request)
#             duration = time.time() - before
#             response.headers["X-Response-Time"] = str(duration)
#             print(f"route duration: {duration}")
#             print(f"route response: {response}")
#             print(f"route response headers: {response.headers}")
#             print('ssssssss')
#             return response

#         return custom_route_handler




# router = APIRouter(route_class=TimedRoute)





# api_key_header = APIKeyHeader(name="authorization", auto_error=False)







@app.get("/")
def read_root():
    return {"test_connection":"Hello, World!"}

@app.get("/test-cors")
def test_cors():
    return {"message": "CORS test successful"}

@app.get("/items")
async def read_items(user_agent: Annotated[str | None, Header()] = None):
    print(user_agent)
    return {"User-Agent": user_agent}




@app.post("/files")
async def create_file(file: UploadFile = File(),url:str = Form()):
    warnings.filterwarnings('ignore')
    if file.content_type not in {"audio/mpeg", "audio/mp3", "audio/wav", "video/mp4",'audio/x-wav'}:
        raise HTTPException(status_code=400, detail="Only MP3, MP4, WAV files are supported.")
    name_uuid = uuid.uuid4().hex + '.wav'
    file_content = await file.read()
    file_path = os.path.join(os.getcwd(), name_uuid)
    with open(file_path, "wb") as f:
        f.write(file_content)
    segments,sr = split_audio(file_path)
    chroma_features = create_chroma_features(segments,sr)
    myModel = pickle.load(open("./Model.sav", 'rb'))
    chords = []
    for i,chroma in enumerate(chroma_features):
        chroma = [chroma]
        chord = myModel.predict(chroma)
        chords.append(chord[0])
    try:
        os.remove(file_path)
        print(f"The file '{file_path}' has been deleted successfully.")
    except OSError as e:
        print(f"Error deleting the file '{file_path}': {e}")
    return HTTPException(status_code=200, detail={'chords':[chords],'url':url})