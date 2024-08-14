import argparse
import wandb 

def default_argument_parser():
    parser = argparse.ArgumentParser("Pedestrain-Detection with RetineNet")
    parser.add_argument("--mode", 
                        required=True,
                        type=str,
                        help = "Log for the current mode (Train/Test).")
    
    parser.add_argument("-c", "--config_file",
                        required=True,
                        type=str,
                        help="yaml file that contains train/test configurations.") 

    parser.add_argument("-o", "--OUTPUT_DIR",
                        required=True,
                        type=str,
                        help="Output directory")

    parser.add_argument("-rn", "--run_name",
                        required=False,                     
                        type=str,
                        default="",
                        help="Wandb name of Run of the current project.")
    
    parser.add_argument("-v", "--video_name",
                        required=False,
                        type=str,
                        default="",
                        help="Name of the output video.")

    parser.add_argument("-m", "--model",
                        required=False,
                        type=str,
                        default="",
                        help="Path of model ckpt.")
    
    return parser.parse_args()

def setup(args):
    MODE = args.mode # train, test
    assert MODE in ['train', 'test'], "Mode should be either 'train' or 'test'."
    print('[INFO] Mode:', MODE)

    cfg_dir = args.config_file
    output_dir = args.OUTPUT_DIR 

    if MODE == 'train':
        run_name =args.run_name 
        return cfg_dir, output_dir, run_name
    elif MODE == 'test':
        video_name = args.video_name 
        model_path = args.model
        return cfg_dir, output_dir, model_path, video_name
      
def get_log(run_name, configs):
    wandb.login()
    wandb.init(project=configs['project'],
               entity=configs['entity'],
               name =run_name )