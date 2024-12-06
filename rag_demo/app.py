from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from rag_demo.pipeline import process_pdf
import nest_asyncio
from rag.retriever import RAGPipeline
from loguru import logger

app = FastAPI()

# Apply nest_asyncio at the start of the application
nest_asyncio.apply()

# Create templates directory if it doesn't exist
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs("data", exist_ok=True)

        file_path = f"data/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process the PDF file with proper await statements
        await process_pdf(file_path)

        # Return template response with success message
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "message": f"Successfully processed {file.filename}",
                "processing": False,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "upload.html", {"request": request, "error": str(e), "processing": False}
        )


@app.post("/chat")
async def chat(chat_request: ChatRequest):
    rag_pipeline = RAGPipeline()
    try:
        answer = rag_pipeline.rag(chat_request.question)
        print(answer)
        logger.info(answer)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
