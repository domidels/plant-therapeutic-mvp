from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from app.api.routes import router
from app.services.render import render_index
import os, pathlib
from pathlib import Path
from fastapi.templating import Jinja2Templates
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Natural Therapeutics MVP", version="0.1")

# Static
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

def render_index(request: Request):
    # index.html doit être dans templates/
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return render_index(request)


app.include_router(router, prefix="/api", tags=["recommendations"])
