import cv2 as cv
import numpy as np


def playVideoFrameFile():
    # 1. set the video path
    video_path = "data/raw/trimmedJumper.mp4"

    # 2. Ask OpenCV to open the video
    cap = cv.VideoCapture(video_path)

    # print the video's width, height, fps, frame count, duration
    video_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    print(f"Video Width: {int(video_width)}")

    video_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    print(f"Video Height: {int(video_height)}")

    fps = cap.get(cv.CAP_PROP_FPS)
    print(f"FPS: {int(fps)}")

    frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    print(f"Frame Count: {int(frame_count)}")

    # duration = frame_count / fps
    print(f"Duration: {int(frame_count) / int(fps)}")

    # 3. Check whether the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 4. Repeatedly read the next frame
    while True:
        ret, frame = cap.read()

        # if frame is read correctly ret is true
        # 5. If no frame is returned, stop
        if not ret:
            print("Can't recieve frame (stream end?). Exiting...")
            break

        # Resize the frame
        frame = cv.resize(frame, (1280, 720))

        # Convert from BGR to HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Define lower orange range
        lower_orange = np.array([10, 100, 100])

        # Define Upper orange range
        upper_orange = np.array([25, 255, 255])

        # Create an orange mask from HSV image
        mask = cv.inRange(hsv, lower_orange, upper_orange)

        # Bitwise-AND mask and original image
        res = cv.bitwise_and(frame, frame, mask=mask)

        # adds a gray filter
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # 6. Show the frame
        cv.imshow('frame', frame)

        # show the mask
        cv.imshow('mask', mask)

        cv.imshow('res', res)

        key = cv.waitKey(5) & 0xFF
        # 7. If user presses q stop
        if key == ord('q'):
            break

        # Show mask
        elif cv.waitKey(1) == ord('m'):
            pass

    # 8. Release the video object
    cap.release()

    # 9. Destroy the display window
    cv.destroyAllWindows()


def main():
    playVideoFrameFile()


if __name__ == "__main__":
    main()
