import argparse

from rem.config import DISMISS_MEDIA_KEY, GESTURE_MEDIA, MEME_DESCRIPTIONS, TOGGLE_DETECTION_KEY
from rem.runtime import main as run_camera


def main():
    parser = argparse.ArgumentParser(
        description="Show memes for detected gestures.",
        epilog=(
            f"Camera controls: {chr(TOGGLE_DETECTION_KEY)} toggles detection; "
            f"{chr(DISMISS_MEDIA_KEY)} dismisses the active meme; q or Escape quits."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show face and hand landmarks plus live detection metrics",
    )
    discovery = parser.add_mutually_exclusive_group()
    discovery.add_argument(
        "--list-memes",
        action="store_true",
        help="list all configured gesture memes, files, and priorities, then exit",
    )
    discovery.add_argument(
        "--describe-meme",
        metavar="NAME",
        help="show how to activate one meme; use --list-memes to find names",
    )
    args = parser.parse_args()
    if args.list_memes:
        for name, media in GESTURE_MEDIA.items():
            print(f"{name:24} {media.file:52} priority={media.priority}")
        return
    if args.describe_meme:
        media = GESTURE_MEDIA.get(args.describe_meme)
        if media is None:
            parser.error(f"unknown meme {args.describe_meme!r}; use --list-memes to see valid names")
        print(f"{args.describe_meme}: {MEME_DESCRIPTIONS[args.describe_meme]}")
        print(f"Media: {media.file} | priority: {media.priority}")
        return
    run_camera(debug=args.debug)


if __name__ == "__main__":
    main()
