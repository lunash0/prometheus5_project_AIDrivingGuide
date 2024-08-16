import yaml 
import argparse

def default_argument_parser():
    parser = argparse.ArgumentParser("AI Driving Guide project")
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
                        help = "Path for the video.")
    
    parser.add_argument("-i", "--image",
                        required=False,
                        type=str,
                        help="Path for the image") 
    parser.add_argument("-s", "--score_threshold",
                        required=False,
                        type=float,
                        default=0.25,
                        help="Score threshold for detection.")
    return parser.parse_args()

def load_yaml(file_path: str) -> dict:
    """
    Loads a YAML configuration file.
    """
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
