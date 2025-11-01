import os
import uuid
import shutil
import asyncio
import mimetypes
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Iterator
import json

# FIX: Import Cookie
from fastapi import FastAPI, Request, Form, Depends, UploadFile, File, BackgroundTasks, HTTPException, status, Cookie
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
# FIX: OAuth2PasswordRequestForm is only for the /token route, not needed globally
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import jwt, JWTError

from sqlalchemy import create_engine, Column, String, Boolean, DateTime, Text, Integer, ForeignKey, UniqueConstraint, \
    func
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base

# ----- GCS Imports -----
from google.cloud import storage
from google.oauth2 import service_account
from io import BytesIO

# Optional ffmpeg-python wrapper
try:
    import ffmpeg
except Exception:
    ffmpeg = None

# Load .env if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# ----- Config -----
SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
BASE_DIR = Path(__file__).parent.resolve()
# NOTE: Removed UPLOAD_DIR, AVATAR_DIR, THUMB_DIR. Using a single 'tmp' directory for processing only.
# Keep the static mount for serving front-end assets.

# GCS Config
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "vidserv")
# Optional: Path to a service account JSON file for explicit credentials
GCS_CREDENTIALS_JSON = os.environ.get("GCS_CREDENTIALS_JSON")

# Database: prefer MySQL if env set, else sqlite
DB_USER = os.environ.get("MYSQL_USER")
DB_PW = os.environ.get("MYSQL_PASSWORD")
DB_HOST = os.environ.get("MYSQL_HOST")
DB_NAME = os.environ.get("MYSQL_DATABASE")
DB_PORT = os.environ.get("MYSQL_PORT", "3306")

if DB_USER and DB_PW and DB_HOST and DB_NAME:
    DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PW}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    DATABASE_URL = f"sqlite:///{BASE_DIR / 'app.db'}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
                       pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# ----- GCS Client Initialization -----
try:
    credentials = None

    if GCS_CREDENTIALS_JSON:
        # Priority 1: Load from raw JSON string environment variable (DigitalOcean friendly)
        print("Attempting to load GCS credentials from environment JSON string...")
        credentials_info = json.loads(GCS_CREDENTIALS_JSON)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)

    if credentials:
        gcs_client = storage.Client(credentials=credentials)
    else:
        # Fallback to default credentials (e.g., from metadata server if running on GCP)
        print("Attempting to load GCS credentials using default discovery.")
        gcs_client = storage.Client()

    gcs_bucket = gcs_client.bucket(GCS_BUCKET_NAME)
    print(f"Connected to GCS bucket: {GCS_BUCKET_NAME}")

except Exception as e:
    print(f"WARNING: Could not initialize GCS client. {e}")
    exit()


# ----- Models -----
class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(80), unique=True, nullable=False, index=True)
    email = Column(String(150), unique=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    role = Column(String(30), default="User", nullable=False)
    bio = Column(Text, default="")
    # Stores the GCS object name (e.g., 'avatars/user-id.jpg')
    avatar_path = Column(String(255), nullable=True)
    watching_json = Column(Text, default="[]", nullable=False)
    banned = Column(Text, default=False, nullable=False)

    videos = relationship("Video", back_populates="owner")
    comments = relationship("Comment", back_populates="user")
    likes = relationship("VideoLike", back_populates="user")

    @property
    def watching(self):
        import json
        try:
            return json.loads(self.watching_json or "[]")
        except Exception:
            return []

    @watching.setter
    def watching(self, val):
        import json
        self.watching_json = json.dumps(val)


class Video(Base):
    __tablename__ = "videos"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=False)
    description = Column(Text, default="")
    owner_id = Column(String(36), ForeignKey("users.id"))
    owner_username = Column(String(80), nullable=False)
    # Stores the GCS object name (e.g., 'videos/processed_uuid.mp4')
    filepath = Column(String(400), nullable=False)
    # Stores the GCS object name (e.g., 'thumbs/uuid.jpg')
    thumbnail = Column(String(400), nullable=True)
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    is_public = Column(Boolean, default=False)
    is_processed = Column(Boolean, default=False)
    likes_count = Column(Integer, default=0)

    owner = relationship("User", back_populates="videos")
    comments = relationship("Comment", back_populates="video")
    liked_by = relationship("VideoLike", back_populates="video")


class Comment(Base):
    __tablename__ = "comments"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    video_id = Column(String(36), ForeignKey("videos.id"))
    user_id = Column(String(36), ForeignKey("users.id"))
    username = Column(String(80))
    text = Column(Text)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    is_pinned = Column(Boolean, default=False)

    video = relationship("Video", back_populates="comments")
    user = relationship("User", back_populates="comments")


class VideoLike(Base):
    __tablename__ = "video_likes"
    user_id = Column(String(36), ForeignKey("users.id"), primary_key=True)
    video_id = Column(String(36), ForeignKey("videos.id"), primary_key=True)
    user = relationship("User", back_populates="likes")
    video = relationship("Video", back_populates="liked_by")
    __table_args__ = (UniqueConstraint('user_id', 'video_id', name='_user_video_uc'),)


Base.metadata.create_all(bind=engine)


# ----- GCS Helpers -----
def upload_file_to_gcs_from_file(file: UploadFile, destination_blob_name: str, content_type: Optional[str] = None):
    """Uploads an UploadFile content directly to the bucket using upload_from_file()."""
    if not gcs_bucket:
        print("GCS client not initialized. Skipping upload.")
        return False
    try:
        blob = gcs_bucket.blob(destination_blob_name)
        # Rewind the file pointer to the beginning before uploading
        file.file.seek(0)
        blob.upload_from_file(file.file, content_type=content_type)
        return True
    except Exception as e:
        print(f"GCS upload failed for {destination_blob_name}: {e}")
        return False


def upload_file_to_gcs_from_path(source_file_path: Path, destination_blob_name: str,
                                 content_type: Optional[str] = None):
    """Uploads a file from a local path to the bucket. Used in background processing."""
    if not gcs_bucket:
        print("GCS client not initialized. Skipping upload.")
        return False
    try:
        blob = gcs_bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_path, content_type=content_type)
        return True
    except Exception as e:
        # NOTE: This is the error from the user's initial prompt!
        print(f"GCS upload failed for {destination_blob_name}: {e}")
        return False


def delete_file_from_gcs(blob_name: str):
    """Deletes a file from the bucket."""
    if not gcs_bucket:
        print("GCS client not initialized. Skipping delete.")
        return True  # Assume successful for safety
    try:
        blob = gcs_bucket.blob(blob_name)
        if blob.exists():
            blob.delete()
        return True
    except Exception as e:
        print(f"GCS delete failed for {blob_name}: {e}")
        return False


def get_gcs_signed_url(blob_name: str, expiration_seconds: int = 3600) -> Optional[str]:
    """Generates a v4 signed URL for a GCS blob."""
    if not gcs_bucket:
        return None
    try:
        blob = gcs_bucket.blob(blob_name)
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(seconds=expiration_seconds),
            method="GET",
        )
        return url
    except Exception as e:
        print(f"GCS signed URL generation failed for {blob_name}: {e}")
        return None


# ----- FastAPI app -----
app = FastAPI(title="VidServ")
# NOTE: The static mount remains for front-end assets (CSS, JS)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
# FIX: Set auto_error=False so the dependency can check for header OR cookie
# without erroring immediately if the header is missing.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)


# ----- Helpers -----
def get_db() -> Iterator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_password_hash(pw: str) -> str:
    return pwd_ctx.hash(pw)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


def create_access_token(subject: str, expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode = {"sub": subject, "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


# FIX: Updated dependency to check for EITHER header or cookie
async def get_current_user(
        token_header: Optional[str] = Depends(oauth2_scheme),
        token_cookie: Optional[str] = Cookie(None, alias="access_token"),
        db=Depends(get_db)
):
    """
    Gets the current user from EITHER the Authorization header or the access_token cookie.
    """
    token = token_header or token_cookie

    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    user_id = decode_token(token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


# Admin dependency
async def require_admin(current_user: User = Depends(get_current_user)):
    if current_user.role != "Administrator":
        raise HTTPException(status_code=403, detail="Administrator access required")
    return current_user


# FIX: Updated dependency to check for EITHER header or cookie
async def get_optional_current_user(
        token_header: Optional[str] = Depends(oauth2_scheme_optional),
        token_cookie: Optional[str] = Cookie(None, alias="access_token"),
        db=Depends(get_db)
):
    """
    Returns the user if token is valid (from header or cookie), otherwise returns None.
    """
    token = token_header or token_cookie
    if not token:
        return None
    user_id = decode_token(token)
    if not user_id:
        return None  # Invalid token, but don't raise 401
    user = db.query(User).filter(User.id == user_id).first()
    return user  # Returns user or None if not found


# ----- Video background processing (FFmpeg) -----
def generate_thumbnail(video_path: Path, thumb_path: Path):
    """Generates a thumbnail from a local video file at 1 second mark."""
    try:
        if ffmpeg:
            (
                ffmpeg
                .input(str(video_path), ss=1)  # Use ss=1 (1 second mark) for a non-random frame
                .filter('scale', 320, -1)
                .output(str(thumb_path), vframes=1)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        else:
            # no ffmpeg-python installed; skip
            pass
    except Exception as e:
        print(f"Thumbnail generation failed: {e}")
        pass


def compress_and_clean(original_gcs_path: str, video_id: str, original_ext: str, db_session):
    """
    Use ffmpeg to remove duplicate frames and reduce silent audio on local files.
    This runs in a background task and manages GCS I/O.
    """
    processed_gcs_path = f"videos/processed_{video_id}.mp4"

    if not gcs_bucket:
        print("GCS not initialized, cannot process video.")
        # We need a fallback path, but since we are removing local storage,
        # we will stop processing if GCS fails.
        return

    # Create temporary paths for local processing (will be deleted on exit)
    (BASE_DIR / "tmp").mkdir(exist_ok=True)
    original_path = BASE_DIR / "tmp" / f"orig_{video_id}{original_ext}"
    processed_path = BASE_DIR / "tmp" / f"processed_{video_id}.mp4"
    thumb_path = BASE_DIR / "tmp" / f"{video_id}.jpg"

    # 1. Download original from GCS
    try:
        original_blob = gcs_bucket.blob(original_gcs_path)
        if not original_blob.exists():
            print(f"Original file {original_gcs_path} not found in GCS.")
            return

        print(f"Downloading {original_gcs_path} to {original_path}")
        original_blob.download_to_filename(original_path)
    except Exception as e:
        print(f"GCS download failed: {e}")
        return

    # 2. Process locally
    try:
        if ffmpeg:
            # Filters: mpdecimate to drop duplicates; silenceremove to drop long silence
            stream = ffmpeg.input(str(original_path))
            v = stream.video.filter('mpdecimate')
            # re-time frames after mpdecimate
            v = v.filter('setpts', 'N/FRAME_RATE/TB')
            a = stream.audio.filter('silenceremove', begin="0.5", stop_periods=1, stop_threshold='-35dB')
            (
                ffmpeg
                .output(v, a, str(processed_path), vcodec='libx264', acodec='aac', strict='experimental',
                        preset='veryfast',
                        crf=28)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        else:
            # If no ffmpeg, copy original to processed path for subsequent thumbnail and GCS upload
            shutil.copy2(original_path, processed_path)
    except Exception as e:
        print(f"FFmpeg processing failed: {e}. Falling back to copying original.")
        # fallback: copy original
        shutil.copy2(original_path, processed_path)

    # 3. Generate thumbnail from processed video
    generate_thumbnail(processed_path, thumb_path)

    # 4. Upload processed video and thumbnail to GCS
    video_upload_success = upload_file_to_gcs_from_path(processed_path, processed_gcs_path, content_type="video/mp4")

    thumb_gcs_path = f"thumbs/{video_id}.jpg"
    thumb_upload_success = upload_file_to_gcs_from_path(thumb_path, thumb_gcs_path, content_type="image/jpeg")

    # 5. Clean up local files and update DB
    try:
        if video_upload_success:
            video = db_session.query(Video).filter(Video.id == video_id).first()
            if video:
                # Store the GCS object name for processed video and thumbnail
                video.filepath = processed_gcs_path
                video.is_processed = True
                if thumb_upload_success:
                    video.thumbnail = thumb_gcs_path
                db_session.add(video)
                db_session.commit()

            # Delete original file from GCS (it's no longer needed)
            delete_file_from_gcs(original_gcs_path)

    except Exception as e:
        print(f"DB update/cleanup failed: {e}")
        db_session.rollback()
    finally:
        # Final cleanup for all local temp files
        if original_path.exists(): original_path.unlink()
        if processed_path.exists(): processed_path.unlink()
        if thumb_path.exists(): thumb_path.unlink()


async def background_process_video(video_id: str, original_gcs_path: str, original_ext: str):
    db = SessionLocal()
    try:
        compress_and_clean(original_gcs_path, video_id, original_ext, db)
    except Exception as e:
        print(f"Background processing failed: {e}")
    finally:
        db.close()


# ----- Routes: pages (Unchanged, templates will use GCS signed URLs) -----
@app.get("/", response_class=HTMLResponse)
def page_home(request: Request, db=Depends(get_db), token: Optional[str] = None,
              current_user: Optional[User] = Depends(get_optional_current_user)):
    # token is optional query param for public access fallback
    # FIX: Use the dependency-injected user if available
    user = current_user
    if not user and token:  # Fallback for query param token
        uid = decode_token(token)
        if uid:
            user = db.query(User).filter(User.id == uid).first()

    # Recent uploads (public + processed)
    recent = db.query(Video).filter(Video.is_public == True, Video.is_processed == True).order_by(
        Video.upload_date.desc()).limit(12).all()
    watching = []
    if user:
        watch_ids = user.watching
        if watch_ids:
            watching = db.query(Video).filter(Video.owner_id.in_(watch_ids), Video.is_public == True,
                                              Video.is_processed == True).order_by(Video.upload_date.desc()).limit(
                12).all()
    # Add signed URLs for thumbnails
    for video in recent + watching:
        video.thumbnail_url = get_gcs_signed_url(video.thumbnail) if video.thumbnail else None

    # Add signed URL for user's avatar
    # FIX: Added a check for 'user' to prevent AttributeError on unauthenticated requests
    if user:
        if user.avatar_path:
            user.avatar_path = get_gcs_signed_url(user.avatar_path)
        else:
            user.avatar_path = None

    return templates.TemplateResponse("home.html",
                                      {"request": request, "recent": recent, "watching": watching, "user": user})


@app.get("/login", response_class=HTMLResponse)
def page_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/register", response_class=HTMLResponse)
def page_register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.get("/upload", response_class=HTMLResponse)
def page_upload(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("upload.html", {"request": request, "user": current_user})


@app.get("/watch/{video_id}", response_class=HTMLResponse)
def page_watch(
        request: Request,
        video_id: str,
        db=Depends(get_db),
        token: Optional[str] = None,
        current_user: Optional[User] = Depends(get_optional_current_user)
):
    # Get current user
    user = current_user
    if not user and token:
        uid = decode_token(token)
        if uid:
            user = db.query(User).filter(User.id == uid).first()

    # Get video
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404)

    # Permission check
    if not video.is_public:
        if not user or (user.id != video.owner_id and user.role != "Administrator"):
            raise HTTPException(status_code=403, detail="Video not public")

    # Generate GCS signed URL for video streaming
    video.stream_url = get_gcs_signed_url(video.filepath, expiration_seconds=86400)
    # Generate GCS signed URL for thumbnail
    video.thumbnail_url = get_gcs_signed_url(video.thumbnail) if video.thumbnail else None

    # Get comments
    comments = db.query(Comment).filter(Comment.video_id == video_id) \
        .order_by(Comment.is_pinned.desc(), Comment.timestamp.asc()).all()

    # Get video owner
    owner = db.query(User).filter(User.id == video.owner_id).first()
    owner_data = None
    if owner:
        # Remove sensitive fields
        owner_data = {k: v for k, v in owner.__dict__.items() if
                      k not in ("hashed_password", "email", "_sa_instance_state")}
        # Add avatar URL
        owner_data["avatar_path"] = get_gcs_signed_url(owner.avatar_path) if owner.avatar_path else None

    # Add current user's avatar URL
    # FIX: Added a check for 'user'
    if user:
        if user.avatar_path:
            user.avatar_path = get_gcs_signed_url(user.avatar_path)
        else:
            user.avatar_path = None

    return templates.TemplateResponse(
        "watch.html",
        {"request": request, "video": video, "comments": comments, "user": user, "owner": owner_data, "ispublic": video.is_public}
    )


# FIX: Updated route signature to use the new optional dependency
@app.get("/profile/{user_id}", response_class=HTMLResponse)
def page_profile(request: Request, user_id: str, db=Depends(get_db),
                 current_user: Optional[User] = Depends(get_optional_current_user)):
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(status_code=404)
    # show public videos
    vids = db.query(Video).filter(Video.owner_id == user_id, Video.is_public == True,
                                  Video.is_processed == True).order_by(Video.upload_date.desc()).all()

    # Add GCS signed URLs for video thumbnails
    for video in vids:
        video.thumbnail_url = get_gcs_signed_url(video.thumbnail) if video.thumbnail else None

    # FIX: Pass the current_user (or None) to the template
    target_data = None
    target_data = {k: v for k, v in target.__dict__.items() if
                   k not in ("hashed_password", "email", "_sa_instance_state")}

    # Add GCS signed URL for target's avatar
    target_data["avatar_path"] = get_gcs_signed_url(target.avatar_path) if target.avatar_path else None

    # Add GCS signed URL for current user's avatar
    # FIX: Added a check for 'current_user' to prevent AttributeError
    if current_user:
        if current_user.avatar_path:
            current_user.avatar_path = get_gcs_signed_url(current_user.avatar_path)
        else:
            current_user.avatar_path = None

    return templates.TemplateResponse("profile.html",
                                      {"request": request, "profile": target_data, "videos": vids,
                                       "user": current_user})


@app.get("/admin", response_class=HTMLResponse)
def page_admin(request: Request, db=Depends(get_db), current_user: User = Depends(require_admin)):
    pending = db.query(Video).filter(Video.is_public == False).order_by(Video.upload_date.asc()).all()

    # Add GCS signed URLs for video thumbnails
    for video in pending:
        video.thumbnail_url = get_gcs_signed_url(video.thumbnail) if video.thumbnail else None

    return templates.TemplateResponse("admin.html", {"request": request, "pending": pending, "user": current_user})


# ----- API endpoints (auth, CRUD, interactions) -----
@app.post("/api/register")
def api_register(username: str = Form(...), email: str = Form(...), password: str = Form(...), db=Depends(get_db)):
    if db.query(User).filter((User.username == username) | (User.email == email)).first():
        raise HTTPException(status_code=400, detail="Username or email already in use")
    user = User(username=username, email=email, hashed_password=get_password_hash(password), role="User", banned=False)
    db.add(user);
    db.commit();
    db.refresh(user)
    # make first user admin
    if db.query(User).count() == 1:
        user.role = "Administrator";
        db.add(user);
        db.commit()
    return RedirectResponse("/login")


@app.post("/token")
def api_token(form_data: OAuth2PasswordRequestForm = Depends(), db=Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect credentials")
    token = create_access_token(user.id)
    return {"access_token": token, "token_type": "bearer", "user_id": user.id, "role": user.role}


@app.post("/api/profile/edit")
def api_profile_edit(bio: Optional[str] = Form(None), avatar: Optional[UploadFile] = File(None),
                     current_user: User = Depends(get_current_user), db=Depends(get_db)):
    changed = False
    if bio is not None:
        current_user.bio = bio
        changed = True

    if avatar and avatar.filename:
        # 1. Upload to GCS directly from the file stream
        ext = Path(avatar.filename).suffix or ".jpg"
        fname = f"{current_user.id}{ext}"
        gcs_path = f"avatars/{fname}"
        content_type = avatar.content_type or mimetypes.guess_type(fname)[0] or "image/jpeg"

        # Use the upload_file_to_gcs_from_file helper
        upload_success = upload_file_to_gcs_from_file(avatar, gcs_path, content_type)

        if upload_success:
            # 2. Update DB with GCS path
            current_user.avatar_path = gcs_path
            changed = True
        else:
            raise HTTPException(status_code=500, detail="Avatar upload failed to GCS")

    if changed:
        db.add(current_user);
        db.commit()

    return RedirectResponse(f"/profile/{current_user.id}", status_code=status.HTTP_302_FOUND)


@app.post("/api/profile/delete")
def api_profile_delete(current_user: User = Depends(get_current_user), db=Depends(get_db)):
    # delete related content
    db.query(Comment).filter(Comment.user_id == current_user.id).delete(synchronize_session=False)
    db.query(VideoLike).filter(VideoLike.user_id == current_user.id).delete(synchronize_session=False)

    # delete videos and associated files from GCS
    vids = db.query(Video).filter(Video.owner_id == current_user.id).all()
    for v in vids:
        # Delete video file and thumbnail from GCS
        if v.filepath: delete_file_from_gcs(v.filepath)
        if v.thumbnail: delete_file_from_gcs(v.thumbnail)
        db.delete(v)

    # delete avatar from GCS
    if current_user.avatar_path:
        delete_file_from_gcs(current_user.avatar_path)

    db.delete(current_user)
    db.commit()
    return {"message": "account deleted"}


@app.post("/api/videos/upload")
def api_video_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...), title: str = Form(...),
                     description: str = Form(""), current_user: User = Depends(get_current_user), db=Depends(get_db)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file")

    ext = Path(file.filename).suffix or ".mp4"
    vid = str(uuid.uuid4())
    original_gcs_path = f"videos/orig_{vid}{ext}"

    # 1. Upload to GCS immediately (unprocessed) directly from the file stream
    content_type = file.content_type or mimetypes.guess_type(original_gcs_path)[0] or "video/mp4"

    # Save file content temporarily to read multiple times (for direct upload and background task)
    file_content = file.file.read()
    file_like_object = BytesIO(file_content)

    # Rewind and upload to GCS (original file)
    file_like_object.seek(0)

    if not gcs_bucket:
        raise HTTPException(status_code=500, detail="GCS not initialized.")

    try:
        blob = gcs_bucket.blob(original_gcs_path)
        blob.upload_from_file(file_like_object, content_type=content_type)
        upload_success = True
    except Exception as e:
        print(f"Initial GCS upload failed: {e}")
        upload_success = False

    if not upload_success:
        raise HTTPException(status_code=500, detail="Initial video upload failed to GCS")

    # 2. Create DB record (filepath stores GCS object name of the original)
    video = Video(id=vid, title=title, description=description, owner_id=current_user.id,
                  owner_username=current_user.username, filepath=original_gcs_path, is_public=False,
                  is_processed=False)
    db.add(video);
    db.commit();
    db.refresh(video)

    # 3. Background processing task: it will download the original, process it, and upload the result
    background_tasks.add_task(background_process_video, vid, original_gcs_path, ext)

    return RedirectResponse(url=f"/watch/{vid}", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/api/videos/public")
def api_public_videos(db=Depends(get_db)):
    vids = db.query(Video).filter(Video.is_public == True, Video.is_processed == True).order_by(
        Video.upload_date.desc()).all()
    # minimal serialization
    out = []
    for v in vids:
        out.append({
            "id": v.id, "title": v.title, "description": v.description, "owner_id": v.owner_id,
            "owner_username": v.owner_username, "likes": v.likes_count,
            "upload_date": v.upload_date.isoformat(),
            # Provide GCS signed URL for thumbnail
            "thumbnail_url": get_gcs_signed_url(v.thumbnail) if v.thumbnail else None
        })
    return out


@app.get("/api/videos/{video_id}/likes")
def api_video_likes(video_id: str, db=Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return {"video_id": video.id, "likes": video.likes_count}


@app.get("/api/videos/{video_id}")
def api_video_detail(video_id: str, token: Optional[str] = None, db=Depends(get_db)):
    user = None
    if token:
        uid = decode_token(token)
        if uid:
            user = db.query(User).filter(User.id == uid).first()
    v = db.query(Video).filter(Video.id == video_id).first()
    if not v:
        raise HTTPException(status_code=404)
    if not v.is_public and (not user or (user.id != v.owner_id and user.role != "Administrator")):
        raise HTTPException(status_code=403)

    # Generate GCS signed URL for the video stream
    stream_url = get_gcs_signed_url(v.filepath) if v.is_processed else None

    return {
        "id": v.id, "title": v.title, "description": v.description, "owner_id": v.owner_id,
        "owner_username": v.owner_username, "likes": v.likes_count,
        "upload_date": v.upload_date.isoformat(),
        "is_public": v.is_public, "is_processed": v.is_processed,
        "thumbnail_url": get_gcs_signed_url(v.thumbnail) if v.thumbnail else None,
        "stream_url": stream_url  # The front-end should use this URL to stream
    }


# FIX: Changed this function signature to use the optional user dependency
@app.get("/api/videos/{video_id}/stream")
def api_video_stream(video_id: str, user: Optional[User] = Depends(get_optional_current_user), db=Depends(get_db)):
    """
    Redirects to the GCS signed URL for the video, handling permissions.
    """
    v = db.query(Video).filter(Video.id == video_id).first()
    if not v:
        raise HTTPException(status_code=404)

    # Permission check now correctly uses the 'user' object (which can be Admin)
    if not v.is_public and (not user or (user.id != v.owner_id and user.role != "Administrator")):
        raise HTTPException(status_code=403, detail="Video not public or user not authorized")

    if not v.filepath:
        raise HTTPException(status_code=500, detail="Video path missing (GCS)")

    # Generate and return a temporary redirect to the signed URL
    stream_url = get_gcs_signed_url(v.filepath)
    if not stream_url:
        raise HTTPException(status_code=500, detail="Could not generate stream URL")

    # Use RedirectResponse to send the user to the GCS signed URL
    return RedirectResponse(stream_url, status_code=status.HTTP_302_FOUND)


# New endpoint to serve avatars from GCS with a redirect to the signed URL
@app.get("/avatars/{filename}")
def api_avatar_stream(filename: str):
    """Generates a signed URL for an avatar and redirects."""
    # The database stores the full GCS path, e.g., 'avatars/user-id.jpg'.
    # We must construct the path from the filename.
    gcs_path = f"avatars/{filename}"
    avatar_path = get_gcs_signed_url(gcs_path)
    if not avatar_path:
        # Fallback for missing avatar (or if GCS fails)
        raise HTTPException(status_code=404)
    return RedirectResponse(avatar_path, status_code=status.HTTP_302_FOUND)


# New endpoint to serve thumbnails from GCS with a redirect to the signed URL
@app.get("/thumbs/{filename}")
def api_thumbnail_stream(filename: str):
    """Generates a signed URL for a thumbnail and redirects."""
    # The database stores the full GCS path, e.g., 'thumbs/video-id.jpg'.
    # We must construct the path from the filename.
    gcs_path = f"thumbs/{filename}"
    thumb_url = get_gcs_signed_url(gcs_path)
    if not thumb_url:
        # Fallback for missing thumbnail (or if GCS fails)
        raise HTTPException(status_code=404)
    return RedirectResponse(thumb_url, status_code=status.HTTP_302_FOUND)


# Note: Comments, Likes, and Admin routes remain mostly the same,
# but Admin rejection logic is updated for GCS.


@app.get("/api/videos/{video_id}/comments")
def api_get_comments(video_id: str, db=Depends(get_db)):
    v = db.query(Video).filter(Video.id == video_id).first()
    if not v or not v.is_public:
        raise HTTPException(status_code=404)
    cs = db.query(Comment).filter(Comment.video_id == video_id).order_by(Comment.is_pinned.desc(),
                                                                         Comment.timestamp.asc()).all()
    out = []
    for c in cs:
        out.append({"id": c.id, "username": c.username, "text": c.text, "timestamp": c.timestamp.isoformat(),
                    "is_pinned": c.is_pinned, "user_id": c.user_id})
    return out


@app.post("/api/videos/{video_id}/comment")
def api_post_comment(video_id: str, text: str = Form(...), current_user: User = Depends(get_current_user),
                     db=Depends(get_db)):
    v = db.query(Video).filter(Video.id == video_id).first()
    if not v or not v.is_public:
        raise HTTPException(status_code=404)
    c = Comment(video_id=video_id, user_id=current_user.id, username=current_user.username, text=text)
    db.add(c);
    db.commit();
    db.refresh(c)
    return {"message": "commented", "comment_id": c.id}


@app.get("/api/videos/{video_id}/like_status")
def api_like_status(video_id: str, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    v = db.query(Video).filter(Video.id == video_id).first()
    if not v or not v.is_public:
        raise HTTPException(status_code=404)
    existing = db.query(VideoLike).filter(VideoLike.video_id == video_id, VideoLike.user_id == current_user.id).first()
    return {"is_liked": bool(existing), "total": v.likes_count}


@app.post("/api/videos/{video_id}/like")
def api_toggle_like(video_id: str, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    v = db.query(Video).filter(Video.id == video_id).first()
    if not v or not v.is_public:
        raise HTTPException(status_code=404)
    existing = db.query(VideoLike).filter(VideoLike.video_id == video_id, VideoLike.user_id == current_user.id).first()
    if existing:
        db.delete(existing)
        v.likes_count = max(0, v.likes_count - 1)
        db.commit()
        return {"message": "unliked", "likes": v.likes_count, "is_liked": False}
    else:
        like = VideoLike(user_id=current_user.id, video_id=video_id)
        db.add(like)
        v.likes_count += 1
        db.commit()
        return {"message": "liked", "likes": v.likes_count, "is_liked": True}


@app.post("/api/users/{user_id}/watch")
def api_toggle_watch(user_id: str, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(status_code=404)
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot watch yourself")
    lst = current_user.watching
    if user_id in lst:
        lst.remove(user_id)
        action = "unwatched"
    else:
        lst.append(user_id)
        action = "watched"
    current_user.watching = lst
    db.add(current_user);
    db.commit()
    return {"message": action}


# Admin actions
@app.post("/api/admin/review/{video_id}")
def api_admin_review(video_id: str, action: str = Form(...), admin: User = Depends(require_admin), db=Depends(get_db)):
    v = db.query(Video).filter(Video.id == video_id).first()
    if not v:
        raise HTTPException(status_code=404)
    if action == "approve":
        v.is_public = True
        db.add(v);
        db.commit()
        return {"message": "approved"}
    elif action == "reject":
        # Delete video and thumbnail from GCS
        if v.filepath: delete_file_from_gcs(v.filepath)
        if v.thumbnail: delete_file_from_gcs(v.thumbnail)

        db.delete(v);
        db.commit()
        return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)
    raise HTTPException(status_code=400, detail="invalid action")

uvicorn.run(app,host=0.0.0.0,port=8080)
