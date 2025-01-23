import argparse
from pathlib import Path

from hloc import (
    extract_features,
    match_dense,
    reconstruction,
    visualization,
    triangulation,
    pairs_from_retrieval,
    pairs_from_exhaustive,
)
from hloc.utils.io import read_image
from hloc.utils.read_write_model import read_cameras_binary, read_images_binary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default="datasets/aachen_v1.1",
                        help="Path to the dataset, default: %(default)s")
    parser.add_argument("--outputs", type=Path, default="outputs/aachen_v1.1",
                        help="Path to the output directory, default: %(default)s")
    args = parser.parse_args()

    images_path = args.dataset / "images"
    reference_sfm_path = args.dataset / "sparse/0"
    pairs_path = args.outputs / "pairs-netvlad.txt"
    # pairs_path = args.outputs / "pairs-exhaustive.txt"
    sfm_dir = args.outputs / "sfm"
    args.outputs.mkdir(parents=True, exist_ok=True)

    image_list = list(map(lambda x: x.name, images_path.glob("*.*")))

    image = read_image(images_path / image_list[0])
    height, width = image.shape[:2]
    extrinsics = read_images_binary(reference_sfm_path / "images.bin")
    intrinsics = read_cameras_binary(reference_sfm_path / "cameras.bin")
    opts = {}
    for intr in intrinsics.values():
        if intr.model=="SIMPLE_PINHOLE":
            f = intr.params[0]
            cx = intr.params[1]
            cy = intr.params[2]
            opts = dict(camera_model="SIMPLE_PINHOLE", camera_params=','.join(map(str, (f, cx, cy))))
        elif intr.model=="SIMPLE_RADIAL":
            f = intr.params[0]
            cx = intr.params[1]
            cy = intr.params[2]
            k = intr.params[3]
            opts = dict(camera_model="SIMPLE_RADIAL", camera_params=','.join(map(str, (f, cx, cy, k))))
        elif intr.model=="PINHOLE":
            fx = intr.params[0]
            fy = intr.params[1]
            cx = intr.params[2]
            cy = intr.params[3]
            opts = dict(camera_model="PINHOLE", camera_params=','.join(map(str, (fx, fy, cx, cy))))
        else:
            assert False, "Not supported!"

    retrieval_conf = extract_features.confs["netvlad"]
    matcher_conf = match_dense.confs["roma"]

    # retrieval_path = extract_features.main(retrieval_conf, images_path, args.outputs)
    # pairs_from_retrieval.main(retrieval_path, pairs_path, num_matched=3)

    pairs_from_exhaustive.main(pairs_path, list(map(lambda x: x.name, images_path.glob("*.*"))))

    features_path, matches_path = match_dense.main(
        matcher_conf,
        pairs_path,
        images_path,
        args.outputs,
        max_kps=None
    )

    # triangulation.main(
    #     sfm_dir,
    #     reference_sfm_path,
    #     images_path,
    #     pairs_path,
    #     features_path,
    #     matches_path
    # )

    model = reconstruction.main(
        sfm_dir,
        images_path,
        pairs_path,
        features_path,
        matches_path,
        image_options=opts,
        mapper_options=dict(ba_refine_focal_length=False, ba_refine_extra_params=False),
    )
