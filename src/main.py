import math
import cv2 as cv
import numpy as np

# import player_detection function
from player_detection import detect_player

from ball_detection import build_search_roi, get_ball_candidates, choose_best_candidate

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


def build_player_regions(player_box, frame_shape):
    """
    Build dynamic player-relative regions.

    player_box:
        Detected player rectangle from YOLOX.

    ball_search_zone:
        Broad area whwere the ball is allowed to be.

    ball_preference_zone:
        Tighter centered possesion zone used to prefer likely
        player-owned ball candidates during startup.
    """

    # unpack the box
    x1, y1, x2, y2 = player_box
    # get the height and width of the frame
    frame_h, frame_w = frame_shape[:2]

    player_width = x2 - x1
    player_height = y2 - y1

    # Broad search zone
    search_x1 = max(0, x1 + int(player_width * 0.05))
    search_y1 = max(0, y1 + int(player_height * 0.15))
    search_x2 = min(frame_w, x2 + int(player_width * 0.15))
    search_y2 = min(frame_h, y1 + int(player_height * 0.80))

    # Tighten up the preference zone
    pref_x1 = max(0, x1 + int(player_width * 0.18))
    pref_x2 = min(frame_w, x1 + int(player_width * 0.82))
    pref_y1 = max(0, y1 + int(player_height * 0.28))
    pref_y2 = min(frame_h, y1 + int(player_height * 0.72))

    # return a dictionary of the player's box and the area to search for the basketball
    return {
        "player_box": player_box,
        "ball_search_zone": (search_x1, search_y1, search_x2, search_y2),
        "ball_preference_zone": (pref_x1, pref_y1, pref_x2, pref_y2)
    }


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
