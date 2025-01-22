import argparse
from pathlib import Path

from hloc import (
    extract_features,
    match_dense,
    reconstruction,
    visualization,
    pairs_from_retrieval,
    triangulation,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default="datasets/aachen_v1.1",
                        help="Path to the dataset, default: %(default)s")
    parser.add_argument("--outputs", type=Path, default="outputs/aachen_v1.1",
                        help="Path to the output directory, default: %(default)s")
    args = parser.parse_args()

    images = args.dataset / "images"
    reference_sfm = args.dataset / "sparse/0"
    pairs_path = args.outputs / "pairs-netvlad.txt"
    sfm_dir = args.outputs / "sfm"
    args.outputs.mkdir(parents=True, exist_ok=True)

    retrieval_conf = extract_features.confs["netvlad"]
    matcher_conf = match_dense.confs["roma"]

    retrieval_path = extract_features.main(retrieval_conf, images, args.outputs)
    pairs_from_retrieval.main(retrieval_path, pairs_path, num_matched=3)

    features_path, matches_path = match_dense.main(matcher_conf, pairs_path, images, args.outputs, max_kps=None)

    triangulation.main(
        sfm_dir,
        reference_sfm,
        images,
        pairs_path,
        features_path,
        matches_path
    )

    # model = reconstruction.main(sfm_dir, images, pairs_path, features_path, matches_path)
