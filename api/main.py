from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import chat as chat_router

app = FastAPI(title="Video Highlights Chat API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router.router)
