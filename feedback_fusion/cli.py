"""Command-line interface for Feedback-Fusion.

Analyse a feedback file or a YouTube video without launching Streamlit:

    python -m feedback_fusion data/sample_feedback.csv
    python -m feedback_fusion --youtube https://youtu.be/VIDEO_ID
"""

import argparse
import json
import sys

from .config import ConfigError
from .sentiment import NEGATIVE, NEUTRAL, POSITIVE
from .sources.files import FeedbackFileError, analyze_frame, load_feedback
from .sources.youtube import YouTubeAnalyzer, YouTubeError, extract_video_id
from .summarization import summarize


def _print_stats(stats, title):
    counts = stats["counts"]
    percentages = stats["percentages"]
    print(f"\n{title}")
    print("-" * len(title))
    print(f"  Total analysed : {stats['total']}")
    for label in (POSITIVE, NEGATIVE, NEUTRAL):
        print(f"  {label:<14} : {counts[label]:>5}  ({percentages[label]}%)")
    print(f"  Mean compound  : {stats['average_compound']}")
    print(f"  Overall        : {stats['overall']}")


def _print_keywords(keywords):
    if not keywords:
        return
    print("\nTop keywords")
    print("------------")
    for name, score in keywords:
        print(f"  {name:<20} {score}")


def _emit(payload, args, texts, title):
    if args.json:
        print(
            json.dumps(
                {
                    "stats": payload["stats"],
                    "keywords": payload["keywords"],
                    "summary": summarize(texts) if args.summary else None,
                },
                indent=2,
            )
        )
        return

    _print_stats(payload["stats"], title)
    _print_keywords(payload["keywords"])

    for category, stats in payload.get("by_category", {}).items():
        _print_stats(stats, f"Category: {category}")

    if args.summary:
        print("\nSummary")
        print("-------")
        print(summarize(texts) or "(not enough text to summarise)")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="feedback_fusion",
        description="Analyse feedback sentiment from a file or a YouTube video.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("path", nargs="?", help="Path to a CSV or JSON feedback file")
    source.add_argument("--youtube", metavar="URL", help="YouTube video URL or ID")
    parser.add_argument("--summary", action="store_true", help="Also generate a summary")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    parser.add_argument(
        "--no-store", action="store_true", help="Skip writing results to MongoDB"
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.youtube:
        video_id = extract_video_id(args.youtube)
        if not video_id:
            print(f"error: not a valid YouTube URL or ID: {args.youtube}", file=sys.stderr)
            return 2
        try:
            payload = YouTubeAnalyzer().analyze(video_id, persist=not args.no_store)
        except (ConfigError, YouTubeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _emit(payload, args, payload["comments"], f"YouTube video {video_id}")
        return 0

    try:
        frame, feedback_col, category_col = load_feedback(args.path, filename=args.path)
    except FileNotFoundError:
        print(f"error: no such file: {args.path}", file=sys.stderr)
        return 2
    except FeedbackFileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = analyze_frame(frame, feedback_col, category_col)
    _emit(payload, args, payload["texts"], f"Feedback file {args.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
