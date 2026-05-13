from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.background import BackgroundTask

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

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename="classified_images.zip",
        background=BackgroundTask(
            cleanup,
            base_dir
        )
    )