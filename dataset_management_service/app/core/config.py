from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str
    MONGODB_URI: str
    S3_BUCKET_NAME: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "us-east-1"

    class Config:
        env_file = ".env"


settings = Settings()
