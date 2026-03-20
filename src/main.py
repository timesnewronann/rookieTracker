import cv2 as cv
import numpy as np
import math


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

        # copy for drawing
        debug_frame = frame.copy()

        # define ROI bounds
        roi_x1 = 250
        roi_y1 = 300
        roi_x2 = 1050
        roi_y2 = 720

        # Draw the ROI rectangle on debug_frame
        cv.rectangle(debug_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)

        # Crop ROI
        roi = frame[roi_y1: roi_y2, roi_x1:roi_x2]

        # Define lower orange range
        lower_orange = np.array([10, 100, 100])

        # Define Upper orange range
        upper_orange = np.array([25, 255, 255])

        # # convert roi to hsv, not the full frame
        # Convert from BGR to HSV
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

        # Create an orange mask from HSV image
        mask = cv.inRange(hsv, lower_orange, upper_orange)

        # Bitwise-AND mask ROI
        res = cv.bitwise_and(roi, roi, mask=mask)

        # Find contours from the mask
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # loop through the contours
        for contour in contours:
            # get the area
            area = cv.contourArea(contour)

            # draw contour with a smaller test threshold
            if area < 100:
                continue

            # computer permiter and circularity inside the contour loop
            perimeter = cv.arcLength(contour, True)

            if perimeter == 0:
                continue

            circularity = 4 * math.pi * area / (perimeter * perimeter)

            # print circularity
            print(f"area={area:.1f}, circularity={circularity:.2f}")

            # First Test filter
            # If it's not circular enough move on 
            if circularity < 0.45:
                continue

            x, y, w, h = cv.boundingRect(contour)

            # Compute the aspect ratio
            aspect_ratio = w / h

            full_x = roi_x1 + x
            full_y = roi_y1 + y

            print(f"area={area:.1f}, w={w}, h={h}, aspect_ratio={aspect_ratio:.2f}")

            # update to use full frame
            # blue bounding box
            cv.rectangle(debug_frame, (full_x, full_y), (full_x + w, full_y + h), (255, 0, 0), 3)
            cv.putText(
                debug_frame,
                f"{aspect_ratio:.2f}",
                (full_x, max(full_y - 10, 20)),
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
