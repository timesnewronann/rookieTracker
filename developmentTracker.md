# 3/25/2026

## Player-First detection milestone with YOLOX

### What I accomplished

- Cleaned up the tracker architecture so player detection lives in its own `player_detection.py` module instead of overloading `main.py`.
- Installed YOLOX locally in a Python 3.11 virtual environment and learned how editable installs, dependency issues, and package imports work.
- Confirmed the difference between:
  - `yolox_tiny.py` = model config / architecture
  - `yolox_tiny.pth` = trained checkpoint weights
- Ran the official YOLOX demo on a saved test frame and verified that person detection works.
- Integrated YOLOX into my app through `detect_player(frame)`
- Built helper functions to:
  - load the YOLOX detector once
  - run the inference on a frame
  - return only `person` detections
  - select the main on-court player using simple heuristics
- Fixed a bug where YOLOX detection were using resized model-input coordinates instead of the original frame
- Successfully got a moving player box in the basketball tracker that follows the real shooter instead of the mural/background people

# To Do:

- Improve ball association by introducing a tighter preference region inside the broader ball search zone.

- Keep the broad zone for eligibility.

- Use the tighter zone for startup scoring so the tracker prefers the most likely player-owned ball
