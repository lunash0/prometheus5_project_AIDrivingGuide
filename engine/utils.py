import yaml 
import argparse

def default_argument_parser():
    parser = argparse.ArgumentParser("AI Driving Guide project")
    parser.add_argument("--task_type",
                        required=True,
                        type=str,
                        help="Video type : choose 'all' or 'message'")
    parser.add_argument("-c", "--CFG_DIR",
                    required=True,
                    type=str,
                    help="configuration directory")
    
    parser.add_argument("-o", "--OUTPUT_DIR",
                    required=True,
                    type=str,
                    help="Output directory")
    
    parser.add_argument("-v", "--video", 
                        required=False,
                        type=str,
                        default=None,
                        help = "Path for the video.")
    
    parser.add_argument("-i", "--image",
                        required=False,
                        type=str,
                        default=None,
                        help="Path for the image") 
    parser.add_argument("-ped_thr", "--ped_score_threshold",
                        required=False,
                        type=float,
                        default=0.25,
                        help="Score threshold for pedestrian detection.")
    parser.add_argument("-tl_thr", "--tl_score_threshold",
                        required=False,
                        type=float,
                        default=0.35,
                        help="Score threshold for traffic lights detection.")
    return parser.parse_args()

def load_yaml(file_path: str) -> dict:
    """
    Loads a YAML configuration file.
    """
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def setup(args):
    task_type = args.task_type 
    assert task_type in ['all', 'message'], "Mode should be either 'all' or 'message'."

    cfg_dir = args.CFG_DIR
    output_dir = args.OUTPUT_DIR 

    video_path = args.video
    image_path = args.image

    ped_score_threshold = args.ped_score_threshold
    tl_score_threshold = args.tl_score_threshold

    if image_path is None and video_path is None:
        print('Error: Either video_path or image_path should be given.')
        exit(1)

    return task_type, cfg_dir, output_dir, video_path, image_path, ped_score_threshold, tl_score_threshold