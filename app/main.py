from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates

from starlette.background import BackgroundTask

from typing import List

import os
import uuid
import shutil
import zipfile

from app.predict import classify_and_organize

app = FastAPI()

templates = Jinja2Templates(
    directory="app/templates"
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


def cleanup(path: str):

    shutil.rmtree(
        path,
        ignore_errors=True
    )


@app.post("/upload")
async def upload_images(
    files: List[UploadFile] = File(...)
):

    session_id = str(uuid.uuid4())

    base_dir = f"/tmp/{session_id}"

    upload_dir = os.path.join(
        base_dir,
        "uploads"
    )

    output_dir = os.path.join(
        base_dir,
        "classified"
    )

    os.makedirs(
        upload_dir,
        exist_ok=True
    )

    saved_files = []

    for file in files:

        file_path = os.path.join(
            upload_dir,
            file.filename
        )

        with open(file_path, "wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        saved_files.append(file_path)

    classify_and_organize(
        saved_files,
        output_dir
    )

    zip_path = f"{output_dir}.zip"

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_STORED
    ) as zipf:

        for root, dirs, files in os.walk(output_dir):

            for file in files:

                file_path = os.path.join(
                    root,
                    file
                )

                arcname = os.path.relpath(
                    file_path,
                    output_dir
                )

                zipf.write(
                    file_path,
                    arcname
                )

    exterior_files = sorted(os.listdir(os.path.join(output_dir, "exterior")))
    interior_files = sorted(os.listdir(os.path.join(output_dir, "interior")))

    return {
        "session_id": session_id,
        "exterior": exterior_files,
        "interior": interior_files,
        "download_url": f"/download/{session_id}"
    }


@app.get("/preview/{session_id}/{category}/{filename}")
async def preview_image(
    session_id: str,
    category: str,
    filename: str
):
    if category not in {"interior", "exterior"}:
        raise HTTPException(status_code=404, detail="Category not found")

    base_dir = os.path.join("/tmp", session_id)
    file_path = os.path.join(
        base_dir,
        "classified",
        category,
        os.path.basename(filename)
    )

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path)


@app.get("/download/{session_id}")
async def download_zip(session_id: str):
    base_dir = os.path.join("/tmp", session_id)
    output_dir = os.path.join(base_dir, "classified")
    zip_path = f"{output_dir}.zip"

    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="ZIP not found")

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename="classified_images.zip",
        background=BackgroundTask(cleanup, base_dir)
    )