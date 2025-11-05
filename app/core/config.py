from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    ncbi_tool: str = os.getenv("NCBI_TOOL")
    ncbi_email: str = os.getenv("NCBI_EMAIL")
    user_agent: str = "plant-therapeutics-mvp/0.1 (+contact: {email})"

settings = Settings()