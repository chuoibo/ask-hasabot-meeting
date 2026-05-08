from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn

from main import router


app = FastAPI()
app.include_router(router)


@app.get("/")
async def demo_root():
    return FileResponse(Path(__file__).with_name("demo.html"))


@app.get("/demo")
async def demo_page():
    return FileResponse(Path(__file__).with_name("demo.html"))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
