from fastapi import FastAPI, HTTPException, Query
import pandas as pd

questions_data = pd.read_csv("interview_questions.csv")

app = FastAPI(title="Interview Questions API")

@app.get("/questions/{company_name}")
def get_questions(company_name: str, limit: int = Query(default=100, description="Max number of questions to return")):
    filtered = questions_data[questions_data['Company'].str.lower() == company_name.lower()]

    if filtered.empty:
        raise HTTPException(status_code=404, detail="Company not found or no questions available")

    return filtered['Question'].tolist()[:limit]
