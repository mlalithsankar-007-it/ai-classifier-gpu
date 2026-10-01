from fastapi import FastAPI, UploadFile, File, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from typing import List

from functools import lru_cache
import mimetypes
import json
import os
import tempfile
import traceback
import uuid
import shutil
import zipfile

from botocore.exceptions import ClientError

from app.predict import classify_and_organize, create_minio_client

app = FastAPI()

templates = Jinja2Templates(
    directory="app/templates"
)

SESSION_BUCKET = os.getenv("SESSION_BUCKET", "images")
SESSION_PREFIX = os.getenv("SESSION_PREFIX", "sessions").strip("/") or "sessions"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@lru_cache(maxsize=1)
def get_minio_client():
    try:
        return create_minio_client()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


def get_session_id(session_id: str) -> str:
    try:
        return str(uuid.UUID(session_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


def session_object_key(session_id: str, *parts: str) -> str:
    return "/".join((SESSION_PREFIX, session_id, *parts))


def delete_session_objects(s3, session_id: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=SESSION_BUCKET,
        Prefix=f"{session_object_key(session_id)}/",
    ):
        objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        if objects:
            result = s3.delete_objects(
                Bucket=SESSION_BUCKET,
                Delete={"Objects": objects, "Quiet": True},
            )
            if result.get("Errors"):
                raise RuntimeError("MinIO did not delete every session object")


def stream_object(body):
    try:
        while chunk := body.read(1024 * 1024):
            yield chunk
    finally:
        body.close()


def save_session_status(s3, session_id: str, status: dict):
    s3.put_object(
        Bucket=SESSION_BUCKET,
        Key=session_object_key(session_id, "status.json"),
        Body=json.dumps(status).encode("utf-8"),
        ContentType="application/json",
    )


def process_session(session_id: str, uploaded_files: list):
    base_dir = tempfile.mkdtemp(prefix=f"{session_id}-")
    upload_dir = os.path.join(base_dir, "uploads")
    output_dir = os.path.join(base_dir, "classified")
    zip_path = f"{output_dir}.zip"

    try:
        s3 = create_minio_client()
        os.makedirs(upload_dir, exist_ok=True)
        saved_files = []

        for item in uploaded_files:
            file_path = os.path.join(upload_dir, item["filename"])
            s3.download_file(SESSION_BUCKET, item["key"], file_path)
            saved_files.append(file_path)

        classify_and_organize(saved_files, output_dir)

        exterior_dir = os.path.join(output_dir, "exterior")
        interior_dir = os.path.join(output_dir, "interior")
        exterior_files = sorted(os.listdir(exterior_dir))
        interior_files = sorted(os.listdir(interior_dir))

        with zipfile.ZipFile(
            zip_path,
            "w",
            compression=zipfile.ZIP_STORED,
        ) as zipf:
            for root, _, local_files in os.walk(output_dir):
                for filename in local_files:
                    file_path = os.path.join(root, filename)
                    zipf.write(file_path, os.path.relpath(file_path, output_dir))

        for category, filenames in (
            ("exterior", exterior_files),
            ("interior", interior_files),
        ):
            for filename in filenames:
                s3.upload_file(
                    os.path.join(output_dir, category, filename),
                    SESSION_BUCKET,
                    session_object_key(session_id, category, filename),
                )

        s3.upload_file(
            zip_path,
            SESSION_BUCKET,
            session_object_key(session_id, "classified_images.zip"),
        )
        save_session_status(
            s3,
            session_id,
            {
                "state": "complete",
                "exterior": exterior_files,
                "interior": interior_files,
            },
        )
    except Exception:
        print(f"Error processing session {session_id}:")
        traceback.print_exc()
        try:
            save_session_status(
                create_minio_client(),
                session_id,
                {"state": "failed", "detail": "Image processing failed. Please upload again."},
            )
        except Exception:
            traceback.print_exc()
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


@app.post("/upload")
async def upload_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):

    session_id = str(uuid.uuid4())
    s3 = None

    try:
        s3 = get_minio_client()
        uploaded_files = []
        used_names = set()

        for index, file in enumerate(files):
            filename = os.path.basename((file.filename or "").replace("\\", "/"))
            if not filename:
                raise HTTPException(status_code=400, detail="Every uploaded file must have a filename")

            stem, extension = os.path.splitext(filename)
            unique_name = filename
            suffix = 1
            while unique_name in used_names:
                unique_name = f"{stem}_{suffix}{extension}"
                suffix += 1
            used_names.add(unique_name)

            object_key = session_object_key(
                session_id,
                "uploads",
                f"{index:06d}_{unique_name}",
            )
            s3.upload_fileobj(file.file, SESSION_BUCKET, object_key)
            uploaded_files.append({"key": object_key, "filename": unique_name})

        save_session_status(s3, session_id, {"state": "processing"})
        background_tasks.add_task(process_session, session_id, uploaded_files)

    except HTTPException:
        if s3 is not None:
            try:
                delete_session_objects(s3, session_id)
            except Exception:
                traceback.print_exc()
        raise
    except Exception as exc:
        print("Error in upload_images endpoint:")
        traceback.print_exc()
        if s3 is not None:
            try:
                delete_session_objects(s3, session_id)
            except Exception:
                traceback.print_exc()
        raise HTTPException(status_code=500, detail="Image processing failed") from exc
    return JSONResponse(
        status_code=202,
        content={
            "session_id": session_id,
            "status_url": f"/status/{session_id}",
            "download_url": f"/download/{session_id}",
        },
    )


@app.get("/status/{session_id}")
async def session_status(session_id: str):
    session_id = get_session_id(session_id)
    s3 = get_minio_client()
    try:
        result = s3.get_object(
            Bucket=SESSION_BUCKET,
            Key=session_object_key(session_id, "status.json"),
        )
        body = result["Body"]
        try:
            return json.loads(body.read().decode("utf-8"))
        finally:
            body.close()
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        raise


@app.get("/preview/{session_id}/{category}/{filename}")
async def preview_image(
    session_id: str,
    category: str,
    filename: str
):
    if category not in {"interior", "exterior"}:
        raise HTTPException(status_code=404, detail="Category not found")

    session_id = get_session_id(session_id)
    safe_filename = os.path.basename(filename)
    s3 = get_minio_client()
    try:
        result = s3.get_object(
            Bucket=SESSION_BUCKET,
            Key=session_object_key(session_id, category, safe_filename),
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
            raise HTTPException(status_code=404, detail="File not found") from exc
        raise

    media_type = mimetypes.guess_type(safe_filename)[0] or "application/octet-stream"
    return StreamingResponse(stream_object(result["Body"]), media_type=media_type)


@app.get("/download/{session_id}")
async def download_zip(session_id: str):
    session_id = get_session_id(session_id)
    s3 = get_minio_client()
    try:
        result = s3.get_object(
            Bucket=SESSION_BUCKET,
            Key=session_object_key(session_id, "classified_images.zip"),
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
            raise HTTPException(status_code=404, detail="ZIP not found") from exc
        raise

    return StreamingResponse(
        stream_object(result["Body"]),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="classified_images.zip"'},
    )


@app.post("/download/{session_id}/complete")
async def complete_download(session_id: str):
    session_id = get_session_id(session_id)
    s3 = get_minio_client()
    try:
        s3.head_object(
            Bucket=SESSION_BUCKET,
            Key=session_object_key(session_id, "classified_images.zip"),
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        raise

    delete_session_objects(s3, session_id)
    return {"deleted": True}