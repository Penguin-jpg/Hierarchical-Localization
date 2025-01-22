import sys
from pathlib import Path
import subprocess
import logging
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from ..utils.base_model import BaseModel

roma_path = Path(__file__).parent / "../../third_party/RoMa"
sys.path.append(str(roma_path))

from romatch.models.model_zoo.roma_models import roma_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


class Roma(BaseModel):
    default_conf = {
        "name": "two_view_pipeline",
        "model_name": "outdoor",
        "model_utils_name": "dinov2",
        "max_keypoints": 15000,
        'max_num_matches': None,
    }
    required_inputs = [
        "image0",
        "image1",
    ]
    weight_urls = {
        "roma": {
            "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
            "indoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
        },
        "tiny_roma_v1": {
            "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/tiny_roma_v1_outdoor.pth",
        },
        "dinov2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", #hopefully this doesnt change :D
    }

    # Initialize the line matcher
    def _init(self, conf):
        model_path = roma_path / "pretrained" / conf["model_name"]
        dinov2_weights = roma_path / "pretrained" / conf["model_utils_name"]

        # Download the model.
        if not model_path.exists():
            model_path.parent.mkdir(exist_ok=True)
            link = self.weight_urls["roma"][conf["model_name"]]
            cmd = ["wget", link, "-O", str(model_path)]
            logger.info(f"Downloading the Roma model with `{cmd}`.")
            subprocess.run(cmd, check=True)

        if not dinov2_weights.exists():
            dinov2_weights.parent.mkdir(exist_ok=True)
            link = self.weight_urls[conf["model_utils_name"]]
            cmd = ["wget", link, "-O", str(dinov2_weights)]
            logger.info(f"Downloading the dinov2 model with `{cmd}`.")
            subprocess.run(cmd, check=True)

        logger.info("Loading Roma model...")
        # load the model
        weights = torch.load(model_path, map_location="cpu")
        dinov2_weights = torch.load(dinov2_weights, map_location="cpu")

        self.net = roma_model(
            resolution=(14 * 8 * 6, 14 * 8 * 6),
            upsample_preds=False,
            weights=weights,
            dinov2_weights=dinov2_weights,
            device=device,
        )
        logger.info("Load Roma model done.")

    def _forward(self, data):
        image0 = data["image0"].cpu().numpy().squeeze(0) * 255
        image1 = data["image1"].cpu().numpy().squeeze(0) * 255
        image0 = image0.transpose(1, 2, 0)
        image1 = image1.transpose(1, 2, 0)
        image0 = Image.fromarray(image0.astype("uint8"))
        image1 = Image.fromarray(image1.astype("uint8"))
        W_A, H_A = image0.size
        W_B, H_B = image1.size

        # Match
        warp, certainty = self.net.match(image0, image1, device=device)
        # Sample matches for estimation
        matches, certainty = self.net.sample(
            warp, certainty, num=self.conf["max_keypoints"]
        )
        kpts1, kpts2 = self.net.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
        pred = {}
        pred["keypoints0"], pred["keypoints1"] = kpts1, kpts2
        pred['scores'] = certainty

        return pred
