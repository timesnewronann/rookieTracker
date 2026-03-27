import math
import cv2 as cv
import numpy as np


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
    MAX_JUMP_DISTANCE = 120
    # Unpack the player_region dictionary
    player_box = player_regions["player_box"]

    # get the ball_search_zone from the dict
    ball_search_zone = player_regions["ball_search_zone"]
    search_x1, search_y1, search_x2, search_y2 = ball_search_zone

    # Unpack ball_preference_zone
    ball_preference_zone = player_regions["ball_preference_zone"]
    pref_x1, pref_y1, pref_x2, pref_y2 = ball_preference_zone

    if not candidates:
        return None

    # Computer the center of ball_search_zone
    search_center_x = (search_x1 + search_x2) // 2
    search_center_y = (search_y1 + search_y2) // 2

    # Compute the center of the ball_preference_zone
    pref_center_x = (pref_x1 + pref_x2) // 2
    pref_center_y = (pref_y1 + pref_y2) // 2

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
            inside_ball_search_zone = (
                search_x1 <= cx <= search_x2 and
                search_y1 <= cy <= search_y2
            )

            # Calculate the inside ball_pref_zone
            inside_ball_preference_zone = (
                pref_x1 <= cx <= pref_x2 and
                pref_y1 <= cy <= pref_y1
            )

            # On startup, only consider candidates near the player
            # We avoid random orange regions on the floor/background/shorts
            # from winning the first detection.
            if not inside_ball_search_zone:
                continue

            distance_to_search_center = math.hypot(
                cx - search_center_x,
                cy - search_center_y

            )

            distance_to_pref_center = math.hypot(
                cx - pref_center_x,
                cy - pref_center_y
            )

            preference_bonus = 0
            if inside_ball_preference_zone:
                preference_bonus = 35

            # Penalize shapes that are less round.
            # Example:
            # aspect_ratio = 1.0 -> no penalty
            # aspect_ratio = 1.4 -> larger penalty
            aspect_penalty = abs(1.0 - aspect_ratio) * 100

            # Lower score == better
            # Subrtract circularity because Hight circularity is good
            # More circular candidates get rewarded with a lower scorer
            # score = distance_to_player_center + aspect_penalty - (80 * circularity)
            score = (
                distance_to_search_center
                + (0.5 * distance_to_pref_center)
                + aspect_penalty
                - (80 * circularity)
                - preference_bonus
            )

            # --------------------------
            # TRACKING MODE
            # --------------------------

            # We have a ball point, so we prioritize tracking the ball
        else:
            last_x, last_y = ball_path[-1]
            distance_to_last = math.hypot(cx - last_x, cy - last_y)

            # Reject candidates that jumpt too far from the previous point.
            # A real ball can move quickly, but not usually teleport.
            if distance_to_last > MAX_JUMP_DISTANCE:
                continue

            aspect_penalty = abs(1.0 - aspect_ratio) * 60

            # During tracking, closeness to the last point is important
            # Shape is still important, but location continuity is more important
            score = distance_to_last + aspect_penalty - (40 * circularity)

        if best_score is None or score < best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate
