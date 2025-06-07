from fastapi import FastAPI

app = FastAPI(title="Restaurant Reviews API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the RR API!"}

