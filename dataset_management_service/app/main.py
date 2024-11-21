import os
from fastapi import FastAPI
from app.routers import dataset, preprocess
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Dataset Management Service",
    root_path=os.getenv("ROOT_PATH", ""),
    root_path_in_servers=True,
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    # Adjust to your frontend's origin for better security
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(dataset.router, prefix="/datasets", tags=["dataset"])
app.include_router(preprocess.router, prefix="/preprocess", tags=["preprocess"])


@app.get("/")
async def read_root():
    return {"message": "Welcome to fasztAPI_project_template API message "}


@app.get("/health")
def read_health():
    return {"status": "healthy"}
