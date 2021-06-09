import argparse

from utils import create_input_files


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--karpathy-json", required=True, type=str, help="the path to karpathy json")
    parser.add_argument("--image-folder", required=True, type=str, help="the path to image folder")
    parser.add_argument("--output-folder", required=True, type=str, help="the path to output folder")
    return parser.parse_args()


if __name__ == '__main__':
    # Create input files (along with word map)
    args = get_args()
    create_input_files(dataset='coco', karpathy_json_path=args.karpathy_json, image_folder=args.image_folder,
                       captions_per_image=5, min_word_freq=5, output_folder=args.output_folder, max_len=50)
