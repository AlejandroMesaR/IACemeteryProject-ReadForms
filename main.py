from fastapi import FastAPI
from Routes.form_routes import router as form_router

app = FastAPI(title="Form Processing API", description="API for digitizing handwritten form fields")

# Include the form processing router
app.include_router(form_router, prefix="/api/v1", tags=["Form Processing"])

@app.get("/", tags=["Main"])
def main():
    return {"message": "Hello World"}