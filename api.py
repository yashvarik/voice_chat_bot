from fastapi import FastAPI
from pydantic import BaseModel
from another_bot import graph
from typing_extensions import Literal,Annotated,Any

class Userinput(BaseModel):
    input:str


app=FastAPI()

@app.post("/ describe")
def Input_query(data:Userinput):
    response=graph.invoke({'messages':data.input})
    return ("response:",response)