from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from backend.llm import graph
import requests

app = FastAPI()

class chatrequest(BaseModel):
    pdf:str
    question:str

@app.post('/chat')
def api_call(pdf : UploadFile = File(...),question :str = Form(...)):
    with open(pdf.filename ,"wb") as f :
        f.write(pdf.file.read())
    result = graph.invoke({
        "pdf":pdf.filename,
        "user_query":question
    })
    return {"answer": result["answer"]}
    