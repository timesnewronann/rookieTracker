import cv2 as cv
import numpy as np


def playVideoFrameFile():
    # 1. set the video path
    video_path = "data/raw/trimmedJumper.mp4"

    # 2. Ask OpenCV to open the video
    cap = cv.VideoCapture(video_path)

    # 3. Check whether the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # ---METADATA---
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
    # ---METADATA---

    display_mode = "frame"

    # Region of Interest to help grab the ball 

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

        # debug
        debug_frame = frame.copy()

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

        # Find contours from the mask
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # loop through the contours
        for contour in contours:
            # get the area
            area = cv.contourArea(contour)

            # draw contour with a smaller test threshold
            if area < 100:
                continue

            x, y, w, h = cv.boundingRect(contour)

            # Compute the aspect ratio
            aspect_ratio = w / h

            # add a filter
            # if aspect_ratio < 0.7 or aspect_ratio > 1.3:
            #     continue

            print(f"area={area:.1f}, w={w}, h={h}, aspect_ratio={aspect_ratio:.2f}")

            # Draw everything on the debug frame
            cv.drawContours(debug_frame, [contour], -1, (0, 255, 0), 2)

            # blue bounding box
            cv.rectangle(debug_frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv.putText(
                debug_frame,
                f"{aspect_ratio:.2f}",
                (x, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        cv.imwrite("debug_frame.jpg", debug_frame)

        if display_mode == "frame":
            cv.imshow("ShotTracker", debug_frame)

        elif display_mode == "mask":
            cv.imshow("ShotTracker", mask)

        elif display_mode == "res":
            cv.imshow("ShotTracker", res)

        key = cv.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("1"):
            display_mode = "frame"
        elif key == ord("2"):
            display_mode = "mask"
        elif key == ord("3"):
            display_mode = "res"
        elif key == ord("s"):
            cv.imwrite("debug_frame.jpg", debug_frame)
            print("Saved debug_frame.jpg")

    # 8. Release the video object
    cap.release()

    # 9. Destroy the display window
    cv.destroyAllWindows()


def main():
    playVideoFrameFile()


if __name__ == "__main__":
    main()
