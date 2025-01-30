import configparser
import subprocess
import argparse

def modify_config(config_file, train_mode, data=None, gpu=0, result=None):
    """
    Modifies the config.cfg file based on train or apply mode.
    
    Parameters:
        - config_file (str): Path to the config file.
        - train_mode (bool): Whether to run in training mode (True) or application mode (False).
        - train_data_file (str): Path to training data file (if in training mode).
        - apply_data_file (str): Path to application data file (if in apply mode).
    """

    # Load existing configuration
    config = configparser.ConfigParser()
    config.read(config_file)

    # Update [Task] section
    config.set("Task", "train_mode", str(train_mode))
    
    if train_mode:
        # Training mode: specify training data file
        config.set("Task", "train_data_file", data if data else "train.csv")
    else:
        # Apply mode: specify apply data file
        config.set("Task", "apply_data_file", data if data else "apply.csv")
        config.set("Task", "result_file", result if result else "result.csv")

    config.set("Details", "gpu", str(gpu))

    # Save updated configuration
    with open(config_file, "w") as configfile:
        config.write(configfile)

    print(f"Updated {config_file} successfully.")
    

def run_model(script_path, config_file):
    """
    Executes the model using the modified config file.

    Parameters:
        - script_path (str): Path to the script that runs the model.
        - config_file (str): Path to the modified config file.
    """
    
    command = f"python {script_path} {config_file}"
    
    try:
        print(f"Executing command: {command}")
        subprocess.run(command, shell=True, check=True)
        print("Model execution completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing model: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify config and run model.")
    
    parser.add_argument("--config", type=str, default="config.cfg", help="Path to the config file.")
    parser.add_argument("--mode", type=str, choices=["train", "apply"], required=True, help="Run mode: 'train' or 'apply'.")
    parser.add_argument("--script", type=str, default="transformer-cnnv6.py", help="Path to the script that runs the model.")
    parser.add_argument("--data", type=str, help="Path to  data file")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id set to zero as default.")
    parser.add_argument("--result", "-r",type=str, default='result.csv', help="gpu id set to zero as default.")

    args = parser.parse_args()

    # Determine mode and modify config accordingly
    is_train_mode = args.mode == "train"
    
    modify_config(args.config, is_train_mode, args.data, args.gpu, args.result)
    
    # Execute the model with the modified config file
    run_model(args.script, args.config)