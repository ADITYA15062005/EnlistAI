from fastapi import FastAPI, HTTPException
import pandas as pd

app = FastAPI(title="Company Info & Interview Questions API")

company_data = pd.read_csv("top_companies.csv")
questions_data = pd.read_csv("interview_questions.csv")


@app.get("/companies")
def list_all_companies():
    return company_data['Company'].unique().tolist()


@app.get("/company-info/{company_name}")
def get_company_info_and_questions(company_name: str):
    company = company_data[company_data['Company'].str.lower() == company_name.lower()]
    if company.empty:
        raise HTTPException(status_code=404, detail="Company not found in company data")
    row = company.iloc[0]
    company_details = {
        "Company": row["Company"],
        "Rating": row.get("Rating", "N/A"),
        "Reviews": row.get("Reviews", "N/A"),
        "Industry & Location": row.get("Industry & Location", "N/A")
    }

    company_questions = questions_data[questions_data['Company'].str.lower() == company_name.lower()]
    if company_questions.empty:
        questions = []
    else:
        questions = company_questions['Question'].tolist()[:limit]

    return {
        "Company Details": company_details,
        "Total Questions Found": len(questions),
        "Questions": questions
    }
