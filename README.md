# AI Classifier Web

## MinIO configuration

The model continues to use `MODEL_BUCKET` and `MODEL_OBJECT` (by default, `models/classifier/v1/interior_exterior_classifier.h5`). Uploaded images and classified results use the existing MinIO `images` bucket under a session-specific prefix.

Set these environment variables in the OpenShift Deployment:

- `MINIO_ENDPOINT`: MinIO S3-compatible endpoint URL.
- `MINIO_ACCESS_KEY` and `MINIO_SECRET_KEY`: credentials used by the app.
- `SESSION_BUCKET`: bucket for upload sessions; defaults to `images`.
- `SESSION_PREFIX`: object-key prefix for upload sessions; defaults to `sessions`.

Each upload uses `images/sessions/<session-id>/`. Original files go in `uploads/`, classified files go directly in `interior/` and `exterior/`, and the downloadable ZIP is stored at the session root. The app credentials need `PutObject`, `GetObject`, `ListBucket`, and `DeleteObject` access to the session prefix, plus multipart upload/list/abort permissions supported by your MinIO policy for larger files. The model bucket must remain readable by the app as before.

The app removes a session prefix only after the browser has received the complete ZIP and sends the completion request. If the transfer fails, the browser does not send that request and the files remain available for retry. Because users may close or refresh the page without downloading, configure a MinIO lifecycle rule filtered to the configured session prefix (default `sessions/`) in the `images` bucket to expire objects after a suitable retention period (for example, one day). This leaves other objects in `images` and the model bucket untouched.

The browser can confirm that the full ZIP response arrived, but cannot reliably confirm that the browser successfully wrote it to the user's disk. The lifecycle rule is therefore also the recovery path for abandoned sessions or local save failures.
