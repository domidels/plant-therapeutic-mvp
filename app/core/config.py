from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    ncbi_tool: str = os.getenv("NCBI_TOOL")
    ncbi_email: str = os.getenv("NCBI_EMAIL")
    user_agent: str = "plant-therapeutics-mvp/0.1 (+contact: {email})"
    HF_TOKEN: str = os.getenv("HF_TOKEN")
    NUM_PARALLEL: int = int(os.getenv("NUM_PARALLEL", 1))
    TURSO_DATABASE_URL: str = os.getenv("TURSO_DATABASE_URL")
    TURSO_AUTH_TOKEN: str = os.getenv("TURSO_AUTH_TOKEN")

settings = Settings()