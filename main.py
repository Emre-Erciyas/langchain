import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities.sql_database import SQLDatabase
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

db = SQLDatabase.from_uri(os.environ.get("POSTGRES_URL"))
llm = ChatOpenAI(model="gpt-4", temperature=0)
generate_query = create_sql_query_chain(llm, db)
execute_query = QuerySQLDataBaseTool(db=db)

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def run_query(request: QueryRequest):
    try:
        query = generate_query.invoke({"question": request.question})
        response = execute_query.invoke(query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
