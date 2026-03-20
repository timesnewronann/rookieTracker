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

    # track jump distance between tracked balls
    MAX_JUMP_DISTANCE = 120

    # Track the missed consecutive frames
    missed_frames = 0
    MAX_MISSED_FRAMES = 3

    # Cap the trail length to avoid long trail history cluttering the frame
    MAX_TRAIL_POINTS = 30

    # hardcoded hoop region
    hoop_roi_x1 = 1040
    hoop_roi_y1 = 240
    hoop_roi_x2 = 1130
    hoop_roi_y2 = 320

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

        # Updating ROI for dynamic ROI
        frame_height, frame_width = frame.shape[:2]

        # ---- DYNAMIC ROI LOGIC ----
        # Start with either static or dynamic ball ROI
        if not ball_path:
            roi_x1 = 250
            roi_y1 = 300
            roi_x2 = 1050
            roi_y2 = 720
        else:
            last_x, last_y = ball_path[-1]
            margin = 160

            # clamp the ROI
            roi_x1 = max(0, last_x - margin)
            roi_y1 = max(0, last_y - margin)
            roi_x2 = min(frame_width, last_x + margin)
            roi_y2 = min(frame_height, last_y + margin)

            # If ball is entering late flight, include hoop region too
            # Making the hoop a late-flight expansion target for the ROI
            if last_x > 900:
                # left/ top
                roi_x1 = min(roi_x1, hoop_roi_x1)
                roi_y1 = min(roi_y1, hoop_roi_y1)

                # right/ bottom
                roi_x2 = max(roi_x2, hoop_roi_x2)
                roi_y2 = max(roi_y2, hoop_roi_y2)

        # Final Clamp
        roi_x1 = max(0, roi_x1)
        roi_y1 = max(0, roi_y1)
        roi_x2 = min(frame_width, roi_x2)
        roi_y2 = min(frame_height, roi_y2)

        # Draw the manual hoop on the debug frame
        cv.rectangle(debug_frame, (hoop_roi_x1, hoop_roi_y1), (hoop_roi_x2, hoop_roi_y2), (0, 255, 255), 2)

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

        # Track the best candidates for tracking
        best_candidate = None
        # best_circularity = 0
        best_score = None

        # loop through the contours
        for contour in contours:

            # Filter by area
            # get the area
            area = cv.contourArea(contour)

            # draw contour with a smaller test threshold
            if area < 100:
                continue

            # computer permiter and circularity inside the contour loop
            perimeter = cv.arcLength(contour, True)

            if perimeter == 0:
                continue

            # Filter by circularity
            circularity = 4 * math.pi * area / (perimeter * perimeter)

            # Second Test filter
            if circularity < 0.25:
                continue

            x, y, w, h = cv.boundingRect(contour)

            # Compute the aspect ratio
            aspect_ratio = w / h

            full_x = roi_x1 + x
            full_y = roi_y1 + y

            # Compute the center point
            center_x = full_x + w // 2
            center_y = full_y + h // 2

            # print circularity
            print(f"area={area:.1f}, circularity={circularity:.2f}",
                  f"w={w}, h={h}, aspect_ratio={aspect_ratio:.2f}")

            # Case 1: first tracked point
            # Track the candidate with highest circularity
            candidate = (full_x, full_y, w, h, center_x, center_y, aspect_ratio, circularity)

            if not ball_path:
                # pick best by circularity
                score = circularity
                if best_score is None or score > best_score:
                    best_score = score
                    best_candidate = candidate
            else:
                # Case 2: already tracking
                # Track the candidate with the smallest distance to ball_path[-1]
                # use distance to last tracked point as the score
                # For each surviving candidate compute distance to last_center
                # Lower distance == better
                last_x, last_y = ball_path[-1]
                distance = math.hypot(center_x - last_x, center_y - last_y)

                if distance > MAX_JUMP_DISTANCE:
                    continue

                if best_score is None or distance < best_score:
                    best_score = distance
                    best_candidate = candidate

        if best_candidate:
            missed_frames = 0
            # Unpack best_candidate and use best_candidates values
            full_x, full_y, w, h, center_x, center_y, aspect_ratio, circularity = best_candidate

            ball_path.append((center_x, center_y))

            # If we exceed our trail length remove the oldest point
            if len(ball_path) > MAX_TRAIL_POINTS:
                ball_path.pop(0)

            # blue bounding box
            cv.rectangle(debug_frame, (full_x, full_y), (full_x + w, full_y + h), (255, 0, 0), 4)

            cv.circle(debug_frame, (center_x, center_y), 6, (0, 0, 255), - 1)
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
        else:
            missed_frames += 1

        if missed_frames > MAX_MISSED_FRAMES:
            ball_path = []

        # Draw the trail
        for i in range(1, len(ball_path)):
            cv.line(debug_frame, ball_path[i - 1], ball_path[i], (0, 255, 255), 2)

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
