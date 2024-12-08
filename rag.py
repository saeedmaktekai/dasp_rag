from dotenv import load_dotenv
import os
from fastapi import FastAPI
import dspy
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key)
dspy.configure(lm=lm)
from pydantic import BaseModel

class Question(BaseModel):
    question: str

app = FastAPI()
@app.post("/simpleqa")
async def simpleqa(question: Question):
   math = dspy.ChainOfThought("question -> answer: float")
   answer = math(question=question)
   return answer


   


