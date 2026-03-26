import torch  # loads the checkpoint and runs the model

# ValTransform preprocesses the image the way YOLOX expects
from yolox.data.data_augment import ValTransform

# COCO_CLASSES lets us check whether a detection is "person"
from yolox.data.datasets import COCO_CLASSES

# get_exp loads the YOLOX config
from yolox.exp import get_exp

# postprocess turns raw model output into usable detection boxes
from yolox.utils import postprocess

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

    # Resize the frames to the original sizing instead of the extremely small scale
    frame_h, frame_w = frame.shape[:2]

    # Resizes the frame into YOLOX's expected input format
    img, _ = preprocess(frame, None, exp.test_size)

    # Convert the processed image into a PyTorch tensor
    # unsqueeze(0) == adds the batch dimension
    # ex: (channels, heights, width) -> (1, channels, height, width)
    img = torch.from_numpy(img).unsqueeze(0).float()

    # Run inference with gradients disabled.
    # Tells PyTorch
    # - do inferences only
    # - do not track gradients
    # - use less memory
    with torch.no_grad():
        outputs = model(img)
        outputs = postprocess(
            outputs,
            num_classes=len(COCO_CLASSES),
            conf_thre=exp.test_conf,
            nms_thre=exp.nmsthre,
            class_agnostic=True,
        )

    detections = []
    output = outputs[0]

    # If no detections in the frame
    if output is None:
        return detections

    output = output.cpu().numpy()

    # Scale detections from YOLO test_size to original frame size
    ratio = min(exp.test_size[0] / frame.shape[0], exp.test_size[1] / frame.shape[1])

    for row in output:
        x1, y1, x2, y2, obj_conf, cls_conf, cls_id = row[:7]
        cls_id = int(cls_id)

        # divide the coordinates by the ratio
        x1 /= ratio
        y1 /= ratio
        x2 /= ratio
        y2 /= ratio

        # keep only person detections
        if COCO_CLASSES[cls_id] != "person":
            continue

        conf = float(obj_conf * cls_conf)

        detections.append((int(x1), int(y1), int(x2), int(y2), conf))

    return detections


def choose_main_player():
    pass

# TODO: update with YOLO to detect player
# Temporarily updating to test the two functions


def detect_player(frame):
    """
    run YOLOX
    get all person boxes
    select the real on-court player
    return one box
    """
    predictor = load_player_detector()

    person_boxes = get_person_detections(frame, predictor)

    print(f"person_boxes: {person_boxes}")
    return (470, 260, 660, 620)
