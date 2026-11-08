import argparse

from utils.gestures import main as run_camera


def main():
    parser = argparse.ArgumentParser(description="Show memes for detected gestures.")
    parser.add_argument("--debug", action="store_true", help="show landmarks and live detection metrics")
    args = parser.parse_args()
    run_camera(debug=args.debug)


if __name__ == "__main__":
    main()
