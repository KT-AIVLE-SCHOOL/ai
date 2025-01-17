from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from typing import Annotated
from pydantic import BaseModel
import requests
import os

app = FastAPI()
path = os.getcwd()

class ChatRequest(BaseModel):
    question: str

@app.get('/api/getAnswer')
async def getAnswer(req: Annotated[ChatRequest, Query()]):
   response = {'answer': req.question}
   return JSONResponse(content=response)