import torch  # loads the checkpoint and runs the model

# ValTransform preprocesses the image the way YOLOX expects
from YOLOX.yolox.data.data_augment import ValTransform

# COCO_CLASSES lets us check whether a detection is "person"
from YOLOX.yolox.data.datasets import COCO_CLASSES

# get_exp loads the YOLOX config
from YOLOX.yolox.exp import get_exp

# postprocess turns raw model output into usable detection boxes
from YOLOX.yolox.utils import postprocess

# Load the model once, if we reload YOLOX every frame this app will be super slow
PLAYER_PREDICTOR = None


def load_player_detector():
    """
    load YOLOX config once and reuse it across frames
    load weights
    build the model
    cache it so we do not reload every frame
    """
    global PLAYER_PREDICTOR

    # If player predictor is already loaded reuse it
    if PLAYER_PREDICTOR is not None:
        return PLAYER_PREDICTOR

    # Load the YOLOX experiment/config file
    # .py = architecture / config
    exp = get_exp("YOLOX/exps/default/yolox_tiny.py", None)

    # Set inference-related values
    exp.test_conf = 0.25
    exp.nmsthre = 0.45
    exp.test_size = (416, 416)

    # Build the neural networkf structure
    model = exp.get_model()
    model.eval()

    # Load the pretrained checkpoint weights
    # .pth = learned weights
    checkpoint = torch.load("models/weights/yolox_tiny.pth", map_location="cpu")

    # Fills the network with the trained parameters
    model.load_state_dict(checkpoint["model"])

    # YOLOX preproccesing helper -> handles image preprocessing before inference
    preprocess = ValTransform(legacy=False)

    # Player predictor dict
    PLAYER_PREDICTOR = {
        "model": model,
        "exp": exp,
        "preprocess": preprocess,
    }

    return PLAYER_PREDICTOR


def get_person_detections(frame, predictor):
    """
    run inference on one frame
    return all detected person boxes
    does not decide who is the shooter

    Run YOLOX on one frame and return only person detections

    Returns:
        list of tuples: (x1, y1, x2, y2, conf)
    """

    # unpack the predictor dictionary
    model = predictor["model"]
    exp = predictor["exp"]
    preprocess = predictor["preprocess"]

    # Preprocess the frame into the format YOLOX expects
    


def choose_main_player():
    pass

# TODO: update with YOLO to detect player


def detect_player(frame):
    """
    run YOLOX
    get all person boxes
    select the real on-court player
    return one box
    """
    return (470, 260, 660, 620)
