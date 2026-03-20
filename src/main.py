import math
import cv2 as cv
import numpy as np

clicked_frame = None


def on_mouse(event, x, y, flags, param):
    global clicked_frame

    # protect agaisnt accidental clicks before a frame is assigned
    if clicked_frame is None:
        print("No frame available for sampling yet.")
        return

    if event == cv.EVENT_LBUTTONDOWN:
        bgr_pixel = clicked_frame[y, x]
        hsv_frame = cv.cvtColor(clicked_frame, cv.COLOR_BGR2HSV)
        hsv_pixel = hsv_frame[y, x]

        print(f"Clicked at (x={x}, y={y})")
        print(f"BGR pixel: {bgr_pixel}")
        print(f"HSV pixel: {hsv_pixel}")

        patch_size = 5
        half = patch_size // 2

        y1 = max(0, y - half)
        y2 = min(clicked_frame.shape[0], y + half + 1)
        x1 = max(0, x - half)
        x2 = min(clicked_frame.shape[1], x + half + 1)

        bgr_patch = clicked_frame[y1:y2, x1:x2]
        hsv_patch = hsv_frame[y1:y2, x1:x2]

        avg_bgr = np.mean(bgr_patch, axis=(0, 1))
        avg_hsv = np.mean(hsv_patch, axis=(0, 1))

        print(f"Average BGR in {patch_size}x{patch_size} patch: {avg_bgr}")
        print(f"Average HSV in {patch_size}x{patch_size} patch: {avg_hsv}")
        print("-" * 50)


def playVideoFrameFile():
    global clicked_frame
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

    # Track the ball_path
    ball_path = []
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

        # Updated lower orange range it was more yellow before
        lower_orange = np.array([2, 100, 90])

        # Define Upper orange range
        upper_orange = np.array([12, 255, 255])

        # # convert roi to hsv, not the full frame
        # Convert from BGR to HSV
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

        # Create an orange mask from HSV image
        mask = cv.inRange(hsv, lower_orange, upper_orange)

        kernel = np.ones((5, 5), np.uint8)

        # use morphology close to connect broken white region and fill gaps
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

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

            # Second Test filter
            # If it's not circular enough move on
            if circularity < 0.25:
                continue

            x, y, w, h = cv.boundingRect(contour)

            # Compute the aspect ratio
            aspect_ratio = w / h

            full_x = roi_x1 + x
            full_y = roi_y1 + y

            # Center Point
            center_x = full_x + w // 2
            center_y = full_y + h // 2

            cv.circle(debug_frame, (center_x, center_y), 6, (0, 0, 255), - 1)

            # When a contour is a ball candidate append it into our list
            ball_path.append((center_x, center_y))

            # print circularity
            print(f"area={area:.1f}, circularity={circularity:.2f}",
                  f"w={w}, h={h}, aspect_ratio={aspect_ratio:.2f}")

            # update to use full frame
            # blue bounding box
            cv.rectangle(debug_frame, (full_x, full_y), (full_x + w, full_y + h), (255, 0, 0), 4)

            # Show aspect ratio and circularity
            label = f"a:{aspect_ratio:.2f} c:{circularity:.2f}"
            cv.putText(
                debug_frame,
                label,
                (full_x, max(full_y - 10, 20)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        cv.imwrite("debug_frame.jpg", debug_frame)

        if display_mode == "frame":
            cv.imshow("ShotTracker", debug_frame)
            clicked_frame = frame.copy()
            cv.setMouseCallback("ShotTracker", on_mouse)

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
