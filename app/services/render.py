from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

def render_index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})
