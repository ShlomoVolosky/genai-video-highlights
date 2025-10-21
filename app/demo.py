import argparse
from app.main import VideoProcessor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Either a YouTube URL, a local video path, or a text file with one URL/path per line.")
    args = ap.parse_args()

    vp = VideoProcessor()

    sources: list[str] = []
    if args.input.lower().endswith(".txt"):
        with open(args.input, "r", encoding="utf-8") as f:
            sources = [ln.strip() for ln in f if ln.strip()]
    else:
        sources = [args.input]

    for src in sources:
        print(f"\n=== Processing: {src} ===")
        video, highlights = vp.process(src)
        print(f"Saved {len(highlights)} highlights for video_id={video.id}")

if __name__ == "__main__":
    main()
