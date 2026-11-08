import os

import cv2

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


def record_camera(output_path="output.mp4", camera_index=1, fps=20.0):
    """Record video from a camera until the user presses 'q'."""
    # Open the camera
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        cam.release()
        raise RuntimeError(f"Could not open video writer for {output_path!r}")

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                break

            # Write the frame to the output file
            out.write(frame)

            # Display the captured frame
            cv2.imshow("Camera", frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Release the capture and writer objects
        cam.release()
        out.release()
        cv2.destroyAllWindows()


# if __name__ == "__main__":
#     record_camera()
