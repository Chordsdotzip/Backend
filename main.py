import time
from typing import Callable, Union
from fastapi import APIRouter, FastAPI, Request, Response, UploadFile, File, HTTPException, Header
from pydub import AudioSegment
from fastapi.routing import APIRoute
# from fastapi.middleware.cors import CORSMiddleware
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
from typing import Annotated
# from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.cors import CORSMiddleware as CORSMiddleware


# def create_app() -> CORSMiddleware:
#     """Create app wrapper to overcome middleware issues."""
#     fastapi_app = FastAPI()
#     fastapi_app.include_router(APIRouter)
#     return CORSMiddleware(
#         fastapi_app,
#         allow_origins=["*"],
#         allow_credentials=True,
#         allow_methods=["*"],
#         allow_headers=["*"],
#     )


# app = create_app()
app = FastAPI()
load_dotenv()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # expose_headers=["*"]
)

# Set up logging
# logging.basicConfig(level=logging.INFO)




# Set up MongoDB connection
print(os.getenv('MONGO_DB'))
# mongo_client = MongoClient(os.getenv('MONGO_DB'))
# mongo_db = mongo_client["ChordsdotzipDB"]
# mongo_collection = mongo_db["logging"]
client = AsyncIOMotorClient(os.getenv('MONGO_DB'))
database = client["ChordsdotzipDB"]
collection = database["logging"]

API_KEY:str = os.getenv('API_KEY')




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


# async def api_key_middleware(request: Request, call_next):
#     print(request.headers)
#     print(request)
#     api_key = await api_key_header(request).split[1]
#     print(api_key)
#     if not api_key or api_key != API_KEY:
#         raise HTTPException(
#             status_code=401,
#             detail="Invalid API key",
#             headers={"WWW-Authenticate": "Bearer"},
#         )

#     response = await call_next(request)
#     return response

# app.middleware('http')(api_key_middleware)



@app.get("/")
def read_root():
    return {"Hello": "World!"}

@app.get("/test-cors")
def test_cors():
    return {"message": "CORS test successful"}

@app.get("/items/")
async def read_items(user_agent: Annotated[str | None, Header()] = None):
    print(user_agent)
    return {"User-Agent": user_agent}



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


@app.post("/files/")
async def create_file(file: UploadFile = File()):
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
    # chords = count_contiguous_chords(chords)
    try:
        os.remove(file_path)
        print(f"The file '{file_path}' has been deleted successfully.")
    except OSError as e:
        print(f"Error deleting the file '{file_path}': {e}")
    # log_entry = {
    #     "route": request.url.path,
    #     "duration": duration,
    #     "response": response.body.decode("utf-8"),
    #     "headers": dict(response.headers),
    #     "timestamp": time.time()
    # }

    # result = await collection.insert_one(log_entry)
    return {'chords':[chords]}