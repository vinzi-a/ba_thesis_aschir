import argparse
import torch
from pytorchyolo import models, train
from datetime import datetime
import time
import os 
import tensorflow as tf 

#for tracking the compilation time
start_time = datetime.now()
# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Path to model config file')
parser.add_argument('--data', type=str, required=True, help='Path to data file')
parser.add_argument('--epochs', type=str,default=100, required=True, help='number of epochs trained for')
parser.add_argument('--n_cpu', type=int, default=8, help='Number of CPU threads to use during batch generation')
parser.add_argument('--verbose', action='store_true', help='Makes the training more verbose')
parser.add_argument('--pretrained_weights', type=str, help='Path to checkpoint file (.weights or .pth). Starts training from checkpoint model')
parser.add_argument('--checkpoint_interval', type=int, default=1, help='Interval of epochs between saving model weights')
parser.add_argument('--evaluation_interval', type=int, default=1, help='Interval of epochs between evaluations on validation set')
parser.add_argument('--multiscale_training', action='store_true', help='Allow multi-scale training')
parser.add_argument('--iou_thres', type=float, default=0.5, help='Evaluation: IOU threshold required to qualify as detected')
parser.add_argument('--conf_thres', type=float, default=0.1, help='Evaluation: Object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.5, help='Evaluation: IOU threshold for non-maximum suppression')
parser.add_argument('--logdir', type=str, default='logs', help='Directory for training log files (e.g. for TensorBoard)')
parser.add_argument('--seed', type=int, default=-1, help='Makes results reproducible. Set -1 to disable.')
args = parser.parse_args()
print(args)
#train.run() -h
model = models.load_model(args.model)
model.to("cuda")
print(f"training is running on: {next(model.parameters()).device}")
# Train the model
train.run()

#stop the compilation time
end_time = datetime.now()
duration = end_time - start_time 

# Calculate duration in d/h/m/s format
days = duration.days
hours, remainder = divmod(duration.seconds, 3600)
minutes, seconds = divmod(remainder, 60)
duration_str = f"{days}d/{hours}h/{minutes}m/{seconds}s"

# Print the duration
print('Duration:', duration_str)

if os.path.exists("train.tmp"):
    raise FileExistsError(f"The file {tmp_file_path} already exists remove it first before creating.")

# Write the duration to train.tmp file
with open('train.tmp', 'w') as f:
    f.write(duration_str)

#for seing the training process with tensorboard:
# be aware of being in the right environment: 
# $ conda deactivate
# $ poetry shell 
# $ poetry run tensorboard --logdir='logs' --port=6006
#open in browser: http://localhost:6006/   #timeseries 