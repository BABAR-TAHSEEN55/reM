import os

import cv2

MEMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memes")
IMAGE_PATH = os.path.join(MEMES_DIR, "jjk.webp")


def show_image(image_path=IMAGE_PATH):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {image_path}")

    cv2.imshow("Logo", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_image()
