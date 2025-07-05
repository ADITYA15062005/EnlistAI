from fastapi import FastAPI, HTTPException
import pandas as pd
company_data = pd.read_csv("top_companies.csv")

app = FastAPI(title="Company Info API")

@app.get("/companies")
def get_companies():
    return company_data['Company'].unique().tolist()

@app.get("/company/{company_name}")
def get_company_details(company_name: str):
    company = company_data[company_data['Company'].str.lower() == company_name.lower()]

    if company.empty:
        raise HTTPException(status_code=404, detail="Company not found")

    row = company.iloc[0]

    return {
        "Company": row["Company"],
        "Rating": row.get("Rating", "N/A"),
        "Reviews": row.get("Reviews", "N/A"),
        "Industry & Location": row.get("Industry & Location", "N/A")
    }
