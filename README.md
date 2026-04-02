# rookieTracker

# Current State of the Tracker:

- Implemented YOLOX to track the player
- Tried to train YOLOX to track the basketball
- Need CUDA to train the model
- Transitioning this project to Collab to access a GPU

## Old state

- The cyan player box now follows the real player.
- The green zone is still only a player-derived possesion/search zone.
- The green zone does not understand release, shot pocket, or hands.
- The blue ball box can still pick the wrong orange object because the system still does not strongly model "this is the player's ball."

# How to Run this Project

cd shottracker
source venv/bin/activate
python src/main.py

## Local setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip install -v -e . --no-build-isolation --no-deps
pip install opencv-python loguru tqdm tabulate psutil pillow matplotlib torchvision thop pycocotools
cd ..
```

### Weights note

```md
## Model weights

Download pretrained YOLOX weights into:

models/weights/

Example:

- models/weights/yolox_tiny.pth

## Third-party Dependency

This project uses YOLOX for player detection.

- YOLOX repository: https://github.com/Megvii-BaseDetection/YOLOX
- License: Apache-2.0

YOLOX is installed locally as a separate dependency and is not committed into this repository.
Downloaded pretrained weights are also kept out of version control.
```
