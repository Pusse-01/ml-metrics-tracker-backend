import os
from fastapi import FastAPI
from app.routers import metrics

app = FastAPI(
    title="Metrics Management Service",
    root_path=os.getenv("ROOT_PATH", ""),
    root_path_in_servers=True,
)

app.include_router(metrics.router)


@app.get("/")
async def read_root():
    return {"message": "Welcome to fasztAPI_project_template API message "}
