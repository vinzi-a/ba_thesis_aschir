### Build the setup ### 
#cell 
# import all the libraries:  
#for the end file it would be great to have all the libraries in one cell
import cv2
import numpy as np 
import os
import matplotlib.pyplot as plt
import glob

from pytorchyolo import detect, models, train, test
from scipy.stats import gaussian_kde
from natsort import natsorted
from itertools import groupby
import pandas as pd
import subprocess
from tools import *
import csv
import importlib
from datetime import datetime
import torch

# Set environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
# ! export PATH=/usr/local/cuda/bin:$PATH
# ! export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

  #to not run the whole extracting cell for storing: 
xseed=42

# write train and valid file: (90% train, 10% validation)

# edits and comments: 

# k-folded cross validation (we are not doing that) 

def write_train_valid(
    directory="./data/custom_rot", 
    val=0.1, 
    cust_only=True,
    seed=42):
        
    f1 = open(directory + "/train.txt", "w")
    f2 = open(directory + "/valid.txt", "w")
    allim = glob.glob(root_dir= directory + '/images', pathname='*')
    np.random.seed(seed)
    np.random.shuffle(allim)
    nrep = len(allim)
    print("total amount of images: ", nrep, "validation amount: ", round(val*nrep), "training amount: ", round((1-val)*nrep)) 

    if cust_only:
        custom_data = glob.glob(root_dir = directory + '/images', pathname='R[0-9]*_P[0-9]*_*.jpg')
        
        np.random.shuffle(custom_data)
        for c in range(int(nrep)):
            if c < (1-val)*nrep:
                f1.write(directory + "/images/" + allim[c] + "\n")
            else:
                f2.write(directory + "/images/" + allim[c] + "\n")
    else:
        for c in range(int(nrep)):
            if c < 0.9*nrep:
                f1.write(directory + "/images/" + allim[c] + "\n")
            else:
                f2.write(directory + "/images/" + allim[c] + "\n")

    f1.close()
    f2.close()
    return nrep, val, cust_only

x_n_data, xval, xcust_only = write_train_valid(directory="./data/custom_rot",
                                            val=0.1,
                                            cust_only=True,
                                            seed=xseed)

print("train and valid file written")

boxsize = 0.06
rotate = True
k_cross_val = 1
x_bb_gen=None 
x_n_gen=None
#x_pretrained_weights=None
k_cross_val=1
xcust_only=True
key="train_conf_thres"
### train the model: 
for i_conf_thres in [0.10,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00]:
    print("train with: ", i_conf_thres)
    # parameters for the training:
    x_model = "config/yolov3-tiny-custom.cfg" 
    x_data = "config/custom_rot.data"
    x_epochs = 30
    x_iou_thres = 0.5 
    x_conf_thres = i_conf_thres
    x_nms_thres = 0.5
    x_pretrained_weights = None # "checkpoints/yolov3_ckpt_25.pth"
    x_multi_scale = True

    training_cmd = [ #the command we wanna execute in a bash environment in a subprocess. 
        "python", "train_yolo.py",
        "--model", x_model,
        "--data", x_data,
        "--epochs", str(x_epochs),
        "--iou_thres", str(x_iou_thres),
        "--conf_thres", str(x_conf_thres),
        "--nms_thres", str(x_nms_thres),
        "--seed", str(xseed),
        "--n_cpu", str(1), #this is nessesary for not killing the data loader in the training.
        '--checkpoint_interval', str(1),
        #'--pretrained_weights', str(x_pretrained_weights), # comment this out if pretrained_weights is none.   
        "--multiscale_training" if x_multi_scale else ''
    ]
    #start the training in a subprocess: 
    process = subprocess.Popen(training_cmd) # create a new subprocess that the training doesn't affect the notebook. 

    process.wait() # waiting until the subprocess and the training ended. 

    tmp_file_path = 'train.tmp'
    if os.path.exists(tmp_file_path): 
        with open(tmp_file_path, 'r') as f:
            x_train_time = f.read().strip()
            os.remove(tmp_file_path)
    else:        
        raise FileNotFoundError("The subprocess hasn't ended yet. The tmp file does not exist.")

    ### evaluation:   

    #for tracking the compilation time
    start_time = datetime.now()

    model = models.load_model("config/yolov3-tiny-custom.cfg", "checkpoints/yolov3_ckpt_30.pth") #load the trained model with its weights

    all_stats = statistix() # create a statistic object 

    file_val = open("data/custom_rot/valid.txt", "r")
    val_paths = file_val.readlines()
    file_val.close()
    #now we go through all the pictures we wanna add to the statistics: 

    # Parameter:
    ev_conf_thres=0.01
    ev_nms_thres=0.1
    ev_accepted_distance=15

    for val_file in val_paths:
        val_file = val_file.split('/')[-1]
        detect_picture(filename= val_file.rstrip(),
                       img_path = "data/custom_rot/images",
                       pred_label_path = "./data/custom_rot/predicted_labels",
                       model = model, #we use the once loaded model from the cell above to not waist time. 
                       conf_thres=ev_conf_thres,
                       nms_thres=ev_nms_thres,
                       accepted_distance = ev_accepted_distance,
                       stat_all_pics= all_stats, 
                       show = False
                    )
    #stop the compilation time
    end_time = datetime.now()
    duration = end_time - start_time 

    # Calculate duration in d/h/m/s format
    days = duration.days
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_str = f"{days}d/{hours}h/{minutes}m/{seconds}s"
    ev_time=duration_str
    # Print the duration
    #print('Duration:', duration_str)
    ### hier müssen wir alle daten noch in Variablen abspeichern um sie in eine xlx datei zu speichern. 
    all_stats.print_values()
    x_n_not_annotated, x_sens, x_over_det_rate, x_no_hit_p_rate = all_stats.x_get_values() 

    #current Date, Gpu, is wsl, cwd in windows

    # current time 
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S") #format the time 

    # current gpu 
    x_gpu =torch.cuda.get_device_name(0) # extract the gpu name 

    #current working env 
    def detect_environment():
        in_wsl = False
        in_windows = False

        # Prüfen, ob WSL läuft
        try:
            with open("/proc/sys/kernel/osrelease", "r") as f:
                in_wsl = "WSL" in f.read()
        except FileNotFoundError:
            pass

        # Prüfen, ob das Working Directory in Windows ist
        cwd = os.getcwd()
        in_windows = cwd.startswith("/mnt/")

        return in_wsl, in_windows

    in_wsl, in_windows = detect_environment()
    #print("shell in wsl: ", in_wsl, "cwd in windows: ", in_windows)
    #print(x_gpu)

    ##paramter for csv file: 
    # x_data= "config/custom_rot.data"
    # x_train_time =None # "0d/3h/36m/18s"
    # x_model= "config/yolov3-tiny-custom.cfg"
    # x_epochs= 30
    # x_pretrained_weights=None
    # x_iou_thres= 0.85
    # x_conf_thres= 0.1
    # x_nms_thres= 0.5
    # x_multi_scale= True
    # xseed= 42

    #the key is a field for metadata, to group multiple rows belonging together. 

    ## add an line to the csv file: 
    # the key is a field for comments as metadata, for marking different cells belonging together. 
    with open('train_conf_thres.csv', 'a', newline='') as file:
        writer= csv.writer(file)
        writer.writerow([key,
                         x_data, rotate, boxsize, x_n_gen, x_bb_gen, x_n_data, xval, xcust_only,
                         x_model, x_epochs, x_pretrained_weights, x_multi_scale, x_iou_thres, x_conf_thres, x_nms_thres,
                         xseed, x_gpu, current_time, x_train_time,ev_time, k_cross_val, in_wsl, in_windows,
                         x_n_not_annotated, x_sens, x_over_det_rate, x_no_hit_p_rate,
                         ev_conf_thres, ev_nms_thres, ev_accepted_distance])


    #change dir from weigths 
    dest_dir = "./data/custom_rot/train_conf_thres/" + f"{i_conf_thres}"
    subprocess.run(["./change_dir.sh", "true", "./checkpoints/", dest_dir, "true"], check=True) 
