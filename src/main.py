import math
import cv2 as cv
import numpy as np

# Save the current frame on click
# By clicking with mouse, we can see the pixel's color at that exact spot
clicked_frame = None


def on_mouse(event, x, y, flags, param):
    """
    Debug Assitance:
    Let's us click on the video and inspect the pixel's BGR + HSV values.

    Why this matters:
    Ball detection depends on color thresholding in HSV space.
    If the ball is not showing up in the mask correctly, we can use this tool to help us inspect
    the real HSV values of the ball and tune the orange ranges
    """
    global clicked_frame

    # protect agaisnt accidental clicks before a frame is assigned
    if clicked_frame is None:
        print("No frame available for sampling yet.")
        return

    if event == cv.EVENT_LBUTTONDOWN:
        # Get the exact pixel in BGR space
        bgr_pixel = clicked_frame[y, x]

        # Convert the full frame to HSV to allow inspection of the same pixel
        hsv_frame = cv.cvtColor(clicked_frame, cv.COLOR_BGR2HSV)
        hsv_pixel = hsv_frame[y, x]

        print(f"Clicked at (x={x}, y={y})")
        print(f"BGR pixel: {bgr_pixel}")
        print(f"HSV pixel: {hsv_pixel}")

        # Inspect a small path around the clicked point
        # A single pixel can be noisy so a local average can be more helpful
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


def detect_player(frame):
    """
    For now:
    return the current manual player box.
    Later:
    replace with YOLO person detection.
    """
    return (470, 260, 660, 620)


def build_player_regions(player_box, frame_shape):
    """
    Takes a player box and returns dynamic regions derived from it.
    """

    # unpack the box
    x1, y1, x2, y2 = player_box
    # get the height and width of the frame
    frame_h, frame_w = frame_shape[:2]

    # Refactor into player_size
    player_width = x2 - x1
    player_height = y2 - y1

    # get search zone coordinates
    search_x1 = max(0, x1 + int(player_width * 0.05))

    # This search zone is too large -> far above the player
    # Should start a little below the very top of the player box not at the very top
    search_y1 = max(0, y1 + int(player_height * 0.15))
    search_x2 = min(frame_w, x2 + int(player_width * 0.15))

    # Should usually stop around the lower torso / thigh / knee area because that is a better v1 guess for where the ball is likely to be during gather and release
    # This is too generous includes too much of the floor
    search_y2 = min(frame_h, y1 + int(player_height * 0.80))

    # return a dictionary of the player's box and the area to search for the basketball
    return {
        "player_box": player_box,
        "ball_search_zone": (search_x1, search_y1, search_x2, search_y2),
    }


def build_search_roi(ball_path, frame_width, frame_height, startup_roi, margin):
    """
    Decide WHERE we should search for the ball.

    Two modes:
    1. Startup Mode:
        If we do not have any tracked ball points, use a fixed startup ROI.
        Keeps our search small and avoids scanning the whole frame.

    2. Tracking Mode:
        If we already know where the ball was in the last frame,
        search near the last known positon using a margin.

    Why is this helpful:
    - A smaller search area == less false orange object
    - Once tracking starts, the ball should not teleport across the frame,
      so searching near the last point is more stable.
    """

    # using a fixed startup_roi when ball_path is empty
    if not ball_path:
        return startup_roi

    # get the last x, y coordinate on the ball path
    last_x, last_y = ball_path[-1]

    # get the Region of interest coordinates
    roi_x1 = max(0, last_x - margin)
    roi_y1 = max(0, last_y - margin)
    roi_x2 = min(frame_width, last_x + margin)
    roi_y2 = min(frame_height, last_y + margin)

    return roi_x1, roi_y1, roi_x2, roi_y2


def get_ball_candidates(mask, roi_x1, roi_y1):
    """
    Find all contours in the mask that LOOK like a ball

    Idea:
    This function should not decide which contour is "the ball".
    It only builds a list of potential candidates

    - get_ball_candidates() = filtering step
    - choose_best_candidate() = decision sstep

    Inputs:
    - mask: binary image where orange-ish pixels are white
    - roi_x1, roi_y1: offesets we can convert ROI-local contour coordinates
      back into full-frame coordinates

    Returns:
    A list of dictionaries.
    Each dictionary stores information about one candidate contour:
    - bounding box
    - center
    - area
    - circularity
    - etc
    """
    # Find contours from the mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    candidates = []

    # Loop through the contours
    for contour in contours:
        # Area tells us how big the contour is
        # Very tiny contours usually == noise so we can ignore them.
        # get the area
        area = cv.contourArea(contour)

        if area < 100:
            continue

        # compute permiter and circularity inside the contour loop
        perimeter = cv.arcLength(contour, True)

        # If permiter is 0, circularity formula would break
        if perimeter == 0:
            continue

        # Ciruclarity measures how close the contour is to a circle
        # Near 1.0 == very circular
        # Filter by circularity
        circularity = 4 * math.pi * area / (perimeter * perimeter)

        # Second Test filter
        if circularity < 0.25:
            continue

        # Bounding box gives us the width and height
        # Helps us computer aspect ratio and center point
        x, y, w, h = cv.boundingRect(contour)

        if h == 0:
            continue

        # Compute the aspect ratio
        # Aspect ratio near 1 means width and height are similar
        aspect_ratio = w / h

        if not (0.7 <= aspect_ratio <= 1.3):
            continue

        # The contour was found inside the cropped ROI,
        # So x and y are only local to that ROI.
        # We convert them back to full-frame coordinates
        full_x = roi_x1 + x
        full_y = roi_y1 + y

        # Compute the center point
        center_x = full_x + w // 2
        center_y = full_y + h // 2

        # build our dictionairy of potential candidates
        candidates.append({
            "x": full_x,
            "y": full_y,
            "w": w,
            "h": h,
            "center_x": center_x,
            "center_y": center_y,
            "area": area,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
        })

    return candidates


# TODO: Use ball_search zone during startup mode
def choose_best_candidate(candidates, ball_path, player_regions):
    """
    Choose ONE candidate from the candidates list.

    This is where scoring(rating which candidate is the best) occurs

    Why is this separated this function from ball_candidates()?
    Because of two questions:
    1. Which contours are plausible
    2. Out of these plausible contours, which one should we trust most?

    We use two scoring modes:

    A) Startup mode (ball_path is empty)
    We do not have a previous ball location yet.
    So we prefer a candidate that:
    - is inside the player box
    - is close to the player center
    - is circular
    - has an aspect ratio near 1

    B) Tracking mode (ball_path has points)
    We already know where the ball was last frame.
    So we prefer a candidate that :
    - is close to the last known position
    - still looks ball-like
    - does not jump unrealistically far
    """

    # Unpack the player_region dictionary
    player_box = player_regions["player_box"]

    # get the ball_search_zone from the dict
    ball_search_zone = player_regions["ball_search_zone"]
    search_x1, search_y1, search_x2, search_y2 = ball_search_zone

    if not candidates:
        return None

        # Player init box
    player_box_x1, player_box_y1, player_box_x2, player_box_y2 = player_box
    player_center_x = (player_box_x1 + player_box_x2) // 2
    player_center_y = (player_box_y1 + player_box_y2) // 2

    # Computer the center of ball_search_zone
    search_center_x = (search_x1 + search_x2) // 2
    search_center_y = (search_y1 + search_y2) // 2

    best_candidate = None
    best_score = None

    # go through our candidates
    for candidate in candidates:
        cx = candidate["center_x"]
        cy = candidate["center_y"]
        circularity = candidate["circularity"]
        aspect_ratio = candidate["aspect_ratio"]

        # --------------------------
        # STARTUP MODE
        # --------------------------
        # We do not have a tracked ball -> get a good first guess
        if not ball_path:
            # replace inside_player_box with inside_ball_search zone
            inside_ball_search_zone = (
                # player_box_x1 <= cx <= player_box_x2 and
                # player_box_y1 <= cy <= player_box_y2
                search_x1 <= cx <= search_x2 and
                search_y1 <= cy <= search_y2
            )

            # On startup, only consider candidates near the player
            # We avoid random orange regions on the floor/background/shorts
            # from winning the first detection.
            if not inside_ball_search_zone:
                continue

            # Distance to the player's center is used a rough location
            # smaller distance == better
            # distance_to_player_center = math.hypot(
            #     cx - player_center_x,
            #     cy - player_center_y
            # )

            distance_to_search_center = math.hypot(
                cx - search_center_x,
                cy - search_center_y

            )

            # Penalize shapes that are less round.
            # Example:
            # aspect_ratio = 1.0 -> no penalty
            # aspect_ratio = 1.4 -> larger penalty
            aspect_penalty = abs(1.0 - aspect_ratio) * 100

            # Lower score == better
            # Subrtract circularity because Hight circularity is good
            # More circular candidates get rewarded with a lower scorer
            # score = distance_to_player_center + aspect_penalty - (80 * circularity)
            score = distance_to_search_center + aspect_penalty - (80 * circularity)

            # --------------------------
            # TRACKING MODE
            # --------------------------

            # We have a ball point, so we prioritize tracking the ball
        else:
            last_x, last_y = ball_path[-1]
            distance_to_last = math.hypot(cx - last_x, cy - last_y)

            # Reject candidates that jumpt too far from the previous point.
            # A real ball can move quickly, but not usually teleport.
            if distance_to_last > 120:
                continue

            aspect_penalty = abs(1.0 - aspect_ratio) * 60

            # During tracking, closeness to the last point is important
            # Shape is still important, but location continuity is more important
            score = distance_to_last + aspect_penalty - (40 * circularity)

        if best_score is None or score < best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


def draw_debug(frame, roi, player_regions, hoop_box, best_candidate, ball_path):
    """
    Draw overlays on the frame so we can visually debug the tracked points

    This function does not perform detection
    It only draws
    """
    roi_x1, roi_y1, roi_x2, roi_y2 = roi

    # Unpack the player_region dictionary
    player_box = player_regions["player_box"]

    # get the ball_search_zone from the dict
    ball_search_zone = player_regions["ball_search_zone"]
    search_x1, search_y1, search_x2, search_y2 = ball_search_zone

    player_box_x1, player_box_y1, player_box_x2, player_box_y2 = player_box
    hoop_roi_x1, hoop_roi_y1, hoop_roi_x2, hoop_roi_y2 = hoop_box

    debug_frame = frame.copy()

    # Yellow rectange == current search ROI
    # cv.rectangle(debug_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)

    # Cyan Rectangle == startup player box
    cv.rectangle(
        debug_frame, (player_box_x1, player_box_y1), (player_box_x2,
                                                      player_box_y2), (255, 255, 0), 2
    )

    # Draw search zone in green
    cv.rectangle(
        debug_frame, (search_x1, search_y1), (search_x2, search_y2), (0, 255, 0), 2
    )

    # Yellow rectangle = hoop region
    # We are not using it to track shots made/missed yet but we are keeping it for context
    cv.rectangle(
        debug_frame, (hoop_roi_x1, hoop_roi_y1), (hoop_roi_x2, hoop_roi_y2), (0, 255, 255), 2
    )

    if best_candidate:
        x = best_candidate["x"]
        y = best_candidate["y"]
        w = best_candidate["w"]
        h = best_candidate["h"]
        cx = best_candidate["center_x"]
        cy = best_candidate["center_y"]
        aspect_ratio = best_candidate["aspect_ratio"]
        circularity = best_candidate["circularity"]

        # Blue rectangle == chosen ball candidate
        cv.rectangle(debug_frame,
                     (x, y), (x + w, y + h), (255, 0, 0), 3)

        # Red center dot == center of our chosen candidate
        cv.circle(debug_frame, (cx, cy), 5, (0, 0, 255), -1)

        # Show shape states so we understand why a contour was selected
        label = f"a:{aspect_ratio:.2f} c:{circularity:.2f}"
        cv.putText(
            debug_frame,
            label,
            (x, max(y - 10, 20)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    # Draw the tracked path
    for i in range(1, len(ball_path)):
        cv.line(debug_frame, ball_path[i - 1], ball_path[i], (0, 255, 255), 2)

    return debug_frame


def playVideoFrameFile():
    """
    Main coordinator function

    "Game manager"
    It does not contain the logic
    It orchestrates the steps in the proper order

    Frame Loop Order:
    1. Read Frame
    2. Choose ROI
    3. Build Orange Mask
    4. Extract plausible ball candidates
    5. Choose best candidate
    6. Update path
    7. Draw debug overlays
    8. Show Frame
    """
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

    # Display mode lets us inspect different parts of the pipeline
    # "frame" = full debug frame
    # "mask" = binary orange mask
    # "res" = masked color result

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

    # Startup ROI
    # Fixed search area before the first ball point is found
    startup_roi = (420, 220, 760, 680)

    # Hoop box:
    # Drawn for context
    hoop_box = (1040, 240, 1130, 320)

    # Updated lower orange range it was more yellow before
    lower_orange = np.array([2, 100, 90])

    # Define Upper orange range
    upper_orange = np.array([12, 255, 255])

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

        # Updating ROI for dynamic ROI
        frame_height, frame_width = frame.shape[:2]

        # Player box:
        # Used during startup mode to bias the first detection near the shooter
        # player_box = (470, 260, 660, 620)
        player_box = detect_player(frame)
        player_regions = build_player_regions(player_box, frame.shape)

        # copy for drawing
        debug_frame = frame.copy()

        # Figure out where to search on this frame
        roi = build_search_roi(ball_path, frame_width, frame_height, startup_roi, margin=160)
        roi_x1, roi_y1, roi_x2, roi_y2 = roi

        # Crop the roi to only process relevant area
        cropped_roi = frame[roi_y1: roi_y2, roi_x1:roi_x2]

        # convert roi to hsv, not the full frame
        # Convert from BGR to HSV
        hsv = cv.cvtColor(cropped_roi, cv.COLOR_BGR2HSV)

        # Build a binary mask:
        # white = orange pixels
        # black = everything else
        mask = cv.inRange(hsv, lower_orange, upper_orange)

        # Morphological closing
        kernel = np.ones((5, 5), np.uint8)

        # use morphology close to connect broken white region and fill gaps
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        # Bitwise-AND mask ROI
        res = cv.bitwise_and(cropped_roi, cropped_roi, mask=mask)

        # Gather potential candidates
        candidates = get_ball_candidates(mask, roi_x1, roi_y1)

        # Select the best candidate
        best_candidate = choose_best_candidate(candidates, ball_path, player_regions)

        # go through the best candidate
        if best_candidate:
            missed_frames = 0

            # Save the chosen center point into our tracking path
            ball_path.append((best_candidate["center_x"], best_candidate["center_y"]))

            # Keep only the most recent trail points
            if len(ball_path) > MAX_TRAIL_POINTS:
                ball_path.pop(0)
        else:
            missed_frames += 1

        # if we lose the ball too many frames in a row
        # Reset the path and go back to startup
        if missed_frames > MAX_MISSED_FRAMES:
            ball_path = []

        # Draw overlays for debugging
        debug_frame = draw_debug(frame, roi, player_regions, hoop_box, best_candidate, ball_path)

        if display_mode == "frame":
            cv.imshow("ShotTracker", debug_frame)

            # save a copy so the mouse callbak can inspect clicked pixels
            clicked_frame = frame.copy()
            cv.setMouseCallback("ShotTracker", on_mouse)

        elif display_mode == "mask":
            cv.imshow("ShotTracker", mask)

        elif display_mode == "res":
            cv.imshow("ShotTracker", res)

        # waitKey(0) == press a key to go to the next frame
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
