from yolox.exp import Exp as MyExp

# Custom Training setup


class Exp(MyExp):
    def __init__(self):
        super().__init__()

        # -------------------------
        # Model
        # -------------------------
        self.depth = 0.33
        self.width = 0.375
        self.num_classes = 1
        self.input_size = (640, 640)
        self.test_size = (640, 640)

        # -------------------------
        # Training
        # -------------------------
        self.max_epoch = 50
        self.data_num_workers = 2
        self.eval_interval = 5

        # Turn off heavy augmentation for a first smoke test on CPU
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0
        self.hsv_prob = 0.0
        self.flip_prob = 0.5

        # -------------------------
        # Dataset
        # -------------------------
        self.data_dir = "../datasets/basketball"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "instances_val2017.json"

        self.train_name = "train2017"
        self.val_name = "val2017"

        # -------------------------
        # Output
        # -------------------------
        self.exp_name = "basketball_yolox_tiny"
