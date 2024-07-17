from fastapi import FastAPI
from replace_bert import get_importance_values

app = FastAPI()

stopwords = []
with open("stopwords.txt", "r") as f:
    for line in f.readlines():
        stopwords.append(line.strip())

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/explain")
async def explain(prompt):
    response = get_importance_values(prompt, stopwords)

    return {"message": response}