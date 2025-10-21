from typing import List
from tqdm import tqdm

from app.config import Config
from app.db.repository import Repository
from app.processors.video_downloader import VideoDownloader
from app.processors.audio_transcriber import AudioTranscriber
from app.processors.scene_detector import SceneDetector
from app.processors.frame_sampler import FrameSampler
from app.processors.object_detector import ObjectDetector
from app.llm.gemini_client import GeminiClient
from app.llm.highlight_selector import HighlightSelector
from app.types import HighlightModel, VideoRecord


class VideoProcessor:
    def __init__(self):
        self.repo = Repository()
        self.repo.create_schema()

        # processors (DI-friendly)
        self.downloader = VideoDownloader()
        self.transcriber = AudioTranscriber(Config.whisper_model)
        self.scenes = SceneDetector()
        self.sampler = FrameSampler(Config.frame_sample_every_sec)
        self.objects = ObjectDetector(Config.yolo_model)
        self.llm_client = GeminiClient()
        self.selector = HighlightSelector(self.llm_client)

    def process(self, source: str) -> tuple[VideoRecord, List[HighlightModel]]:
        # 1) get video (path, uid)
        vpath, uid = self.downloader.fetch(source)

        # 2) # Pass the *source* (URL or local path). The transcriber accepts either.
        transcript, duration = self.transcriber.transcribe(source)

        # 3) register video
        video = self.repo.upsert_video(source=source, video_uid=uid, duration_sec=int(duration) if duration else None)

        # 4) scene boundaries
        segs = self.scenes.detect_scenes(vpath)
        if not segs:
            segs = [(0, int(duration) if duration else 60)]

        # 5) per-scene: frames → objects → LLM
        highlights: List[HighlightModel] = []
        for (start, end) in tqdm(segs, desc="Analyzing scenes"):
            frames = self.sampler.sample(vpath, start, end)
            objs = self.objects.detect_in_frames(frames)
            hl = self.selector.analyze_segment((start, end), transcript, objs)
            if hl and hl.description:
                # 6) embed description
                emb = self.selector.embed_desc(hl.description)
                hl.embedding = emb
                highlights.append(hl)

        if highlights:
            self.repo.add_highlights(video.id, highlights)

        return video, highlights
