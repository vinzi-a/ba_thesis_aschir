# libraries
import cv2
import numpy as np
import pandas as pd 
import os
from PIL import Image
import matplotlib.pyplot as plt
import glob
from scipy.spatial import distance
from tools import * 
from pytorchyolo import detect, models 
#To Do: 
# kann man peaks auch nummerieren? um sie besser besprechen zu können. oder abweichungen zu bemerken.

#falls es mehrere Klassen gibt, dann sollten sie in unterschiedlichen Farben eingetragen werden. 

# function for visualizing the images with the bounding boxes
# if you wanna see the image with set show=True, make sure to call this function in a jupyter notebook. 
#needs numpy, opencv, cv2 and os
def vis_picture(filename="R3_P35_22_90",
                image_path="data/custom_rot/images", 
                label_path="data/custom_rot/labels",
                pred_label_path=None,
                output_path="output",
                show=False,
                bounding_box=True,
                ac_d=None
):
   #catch input errors
    if filename.endswith('.jpg') or filename.endswith('.png'): #check if the filename got also its ending. 
        filename= filename[:-len('.jpg')]
        
   #setting parameters 
    image=f"{image_path}/{filename}.jpg"
    label=f"{label_path}/{filename}.txt"

   # check the given paths contain the right files
    if image is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}") 
    if not os.path.exists(label):
        raise FileNotFoundError(f"Labeldatei nicht gefunden: {label}")
    
    if not output_path.startswith("/"): 
        output_path = f"./{output_path}"
    output=f"{output_path}/{filename}.jpg" 
    if bounding_box:
        output=f"{output_path}/{filename}_with_bounding_boxes.jpg" 
    if output_path is not None: 
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Ausgabeverzeichnis erstellt: {output_path}")   

    if pred_label_path is not None:
        pred_label=f"{pred_label_path}/{filename}.txt"
    

   # load the picture 
    image = cv2.imread(image) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    scaley, scalex = np.shape(image)[:2]

   # load the labels
    with open(label, "r") as file:
        content = file.readlines()
    
   # transmis the coordinates out of the label files. 
    x, y, w, h = [], [], [], []
    for line in content:
        sep = line.split(' ') #seperator
        if len(sep) >= 5:
            try:
                x.append(float(sep[1]))
                y.append(float(sep[2]))
                w.append(float(sep[3]))
                h.append(float(sep[4]))
            except ValueError:
                print(f"Error in file {label}: {sep}")  # Identify which file has bad data
                continue  # Skip bad line
        else: 
            print(f"the label file {label_path} consists of less than 5 values, check the content of these file.")
            raise ValueError(f"The {filename} should contain at least 5 values per line.") 
        
   # drawing of the bounding boxes
    if bounding_box:
        for i in range(len(x)):
            x0 = (x[i] - w[i] / 2) * scalex
            x1 = (x[i] + w[i] / 2) * scalex
            y0 = (y[i] - h[i] / 2) * scaley
            y1 = (y[i] + h[i] / 2) * scaley

            start_point = (int(x0), int(y0))
            end_point = (int(x1), int(y1))
            cv2.rectangle(image, start_point, end_point, color=(0, 255, 0), thickness=1)
    
   # drawing of the midpoints of the bounding boxes
    mid_points = np.c_[(np.array(x) * scalex), (np.array(y) * scaley)]
    for mitte in mid_points: 
        cv2.circle(image, (int(mitte[0]), int(mitte[1])), 2, color=(0,255, 0), thickness=2)

        #for finding the best accepted distance in mid_points: 
        #if ac_d is not None and bounding_box is not None:
        #   cv2.circle(image, (int(mitte[0]), int(mitte[1])), radius=ac_d, color=(0, 255,0), thickness=1)

   # drawing of the predicted points. 
    if pred_label_path is not None:
        with open(pred_label, "r") as file:
            content = file.readlines()
        x2, y2, w2, h2 = [], [], [], []
        for line in content:
            sep = line.split(' ')
            if len(sep) >= 5:
                x2.append(float(sep[1]))
                y2.append(float(sep[2]))
                w2.append(float(sep[3]))
                h2.append(float(sep[4]))
            else: 
                print(f"the label file {pred_label_path} consists of less than 5 values, check the content of these file.")
        if bounding_box:
            for i in range(len(x2)):
                x0 = (x2[i] - w2[i] / 2) * scalex
                x1 = (x2[i] + w2[i] / 2) * scalex
                y0 = (y2[i] - h2[i] / 2) * scaley
                y1 = (y2[i] + h2[i] / 2) 
                start_point = (int(x0), int(y0))
                end_point = (int(x1), int(y1))
                cv2.rectangle(image, start_point, end_point, color=(255, 165, 0), thickness=1)
        
       # drawing of the midpoints of the predicted points 
        mid_points = np.c_[(np.array(x2) * scalex), (np.array(y2) * scaley)]
        for mitte in mid_points: 
            cv2.circle(image, (int(mitte[0]), int(mitte[1])), 2, color=(255, 165, 0), thickness=2)
        
   # Add a legend if the picture has enough pixels for displaying the legend
    min_size = 250  # Minimum size in pixels to display the legend  
    if scalex > min_size and scaley > min_size: #otherwise the picture is to small to write in it.
        legend_x, legend_y = 10, 10  # Starting point of the legend
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)  # White color
        thickness = 1
        cv2.putText(image, 'Legend:', (legend_x, legend_y + 20), font, font_scale, font_color, thickness)
        cv2.circle(image, (legend_x + 10, legend_y + 40), 3, (0, 255, 0), -1)
        cv2.putText(image, 'Annotated', (legend_x + 30, legend_y + 45), font, font_scale, font_color, thickness)
        if pred_label_path is not None:
            cv2.circle(image, (legend_x + 10, legend_y + 60), 3, (255, 165, 0), -1)
            cv2.putText(image, 'Predicted', (legend_x + 30, legend_y + 65), font, font_scale, font_color, thickness)
    
        if bounding_box: 
           # Add hitbox label
            hitbox_x, hitbox_y = legend_x, legend_y + 80  # Position of the hitbox label
            text = "Bounding Box"
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            hitbox_width, hitbox_height = text_width + 10, text_height + 10  # Add some padding
            cv2.rectangle(image, (hitbox_x, hitbox_y), (hitbox_x + hitbox_width, hitbox_y + hitbox_height), (255, 255, 255), 1)  # White border rectangle
            cv2.putText(image, text, (hitbox_x + 5, hitbox_y + hitbox_height - 5), font, font_scale, font_color, thickness)

   # save the picture
    if output is not None: 
        cv2.imwrite(output, image)
        if show and output is not None:
            print(f"Image saved: {output_path}")
   
   # show
    if show:
        print("")# print(f"your input is: filename= {filename}, image_path= {image_path}, label_path= {label_path}, output_path= {output_path}, show= {show}")
        plt.figure() #diese Zeile ist notwendig, damit das Bild in einem neuen Fenster geöffnet wird. 
        plt.imshow(image)

#this class is used to store the values of a comparison between predicted and manual points over all pictures.
class statistix: 
    def __init__(self):
        self.num_pic = 0 
        self.total_manual_points = 0
        self.total_predicted_points = 0
        self.total_TP = 0 
        self.total_FP = 0
        self.total_FN = 0
        self.total_OD = 0
        self.total_NH = 0 # not-hit-points
        self.n_not_annotated_pics=0

        self.sensitivity = 0
        self.over_detection_rate = 0
        self.no_hit_point_rate = 0 
    
    #adds the values of one picture stored in a class stat_pic to the values of the whole statistic over all pictures. 
    def add_pic(self, stat_pic: "stat_pic"):
        self.num_pic += 1
        self.n_not_annotated_pics += stat_pic.annotated_pic
        self.total_manual_points += stat_pic.manual_p
        self.total_predicted_points += stat_pic.pred_p   
        self.total_TP += stat_pic.TP
        self.total_FP += stat_pic.FP
        self.total_FN += stat_pic.FN
        self.total_OD += stat_pic.OD
        self.total_NH += stat_pic.NH
        self.FP = self.total_NH + self.total_OD

        self.sensitivity = self.total_TP / self.total_manual_points if self.total_manual_points > 0 else 0
        self.over_detection_rate = self.total_OD / self.total_manual_points if self.total_manual_points > 0 else 0
        self.no_hit_point_rate = self.total_NH / self.total_predicted_points if self.total_predicted_points > 0 else 0 

        #self.sensitivity = (self.num_pic-1)/self.num_pic * self.sensitivity + 1/self.num_pic * stat_pic.sensitivity for big data this might be better.

    def x_get_values(self):
        return  self.n_not_annotated_pics, self.sensitivity, self.over_detection_rate, self.no_hit_point_rate
        
    def print_values(self):
        print("\n Number of pictures: ", self.num_pic, "Number of not annotated pictures:", self.n_not_annotated_pics, ", Sensitivity: ", round(self.sensitivity,6), ", over detection rate: ", round(self.over_detection_rate,6), ", no-hit-point-rate: ", round(self.no_hit_point_rate,6),"\n")

#this class is used to store the values of a comparison between predicted and manual points for one picture.
class stat_pic(statistix):
    def __init__(self):
        self.pred_p = 0 # amount of predicted points 
        self.manual_p = 0 # amount of manual points
        self.TP = 0 # true positives
        self.FP = 0 # false positives
        self.FN = 0 # false negatives 
        self.OD = 0 #overdetected points 
        self.NH = 0 #not hit points
        self.sensitivity = 0
        self.annotated_pic= 0

    def set_0(self,man):
        self.pred_p = 0
        self.manual_p = man
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.OD = 0
        self.NH = 0
        self.sensitivity = 0
        self.annotated_pic= 0
    
    #calculates the values of a comparison between predicted and manual points for one picture. 
    def calculate(self, distance_matrix, accepted_distance):
        self.pred_p = distance_matrix.shape[1]
        self.manual_p = distance_matrix.shape[0]
        if self.manual_p == 0:
            raise ValueError("There are no manual points in this picture.")
        for i in range(distance_matrix.shape[0]):
            pos = len(np.where(distance_matrix[i] < accepted_distance)[0])  # number of positives for one manual point (one row)
            if pos > 0:
                self.TP += 1
            if pos > 1:
                self.OD += pos - 1  # adds all the points that are in range but not the closest one.
            if pos == 0:
                self.FN += 1
        #self.NH = pred_p - self.TP - self.OD # this easier and faster method of calculating OP's doesn't work for the case one predicted point can be accepted by multiple annotated points.
        
        #its not super time costing to iterate a second time per column, this time to find predicted points that are not assigned to a manual point
        for j in range(distance_matrix.shape[1]):
            pos = len(np.where(distance_matrix[:,j] < accepted_distance)[0])
            if pos == 0: 
                self.NH += 1
        
        self.FP = self.NH + self.OD 

        if self.manual_p > 0: 
            self.sensitivity = self.TP / self.manual_p
        else: 
            self.sensitivity = 0 
            raise ValueError("There are no manual points in this picture.")
    
    def set_annotated(self):
        self.annotated_pic = 1 # in case the picture is not fully annotated. 
        
    def get_values(self):
        return self.TP, self.FP, self.FN, self.OD, self.NH, self.sensitivity, self.pred_p, self.manual_p, self.annotated_pic

    def print_values(self):
        if self.annotated_pic == 0: 
            print("TP: ", round(self.TP,5), ", FP: ", self.FP, ", FN: ", self.FN, ", OD: ", self.OD, ", NH: ", self.NH, ", Sensitivity: ", round(self.sensitivity,6), ", predicted points: ", self.pred_p, ", manual points: ", self.manual_p)

    #here we should add the definition of the statistic.    
    def explain(self):
        print("This class is used to evaluate the prediction between the manual and prediction points based on a allowed bias. \n")
        print("Test: The test is evaluated as positive for predicted point i if a predicted point i is in the minimum distance range to a manual point j.\n")
        print("Assessment: A prediction is assessed as true/correct if predicted point i is at the minimum distance to manual point j and point i is the next closest to point j. A prediction is assessed as false if this is not the case. \n")
        print("This class is used to calculate the values of a comparison between predicted and manual points. The values are TP, FP, FN, OP, Sensitivity, predicted points and manual points. The values are calculated with the calculate method. The values can be printed with the print_values method. The values can be accessed with the get_values method. ")

def rotate_coord(x, y, angle):
    """
    rotate coordinates by angle, angle in rad
    """
    return x*np.cos(angle) - y*np.sin(angle), x*np.sin(angle) + y*np.cos(angle)

# write train and valid file: (90% train, 10% validation)
def write_train_valid(
    directory="/mnt/c/Users/vinze/Dropbox/Universität/8.Bachelorarbeit/yolo2/PyTorch-YOLOv3/data/custom_rot", 
    val=0.1, # amount of data in the validation set
    cust_only=True, # can the validation set contain custom data? 
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
#To DO: The issue is that we are not only doing the statistics here of the points, we are also doing the prediction. 
# do we outsource the prediction to another function? storing them in a file so we can also visualize them with the visualise function? 

#in the statistics: we should look ip in the matrix which coloumn didnt didnt got a minimum. that we can mark all the points in a different color for example that were over detected. 
#we can give as an output then the number of overdetected points. 
#we can also give the number of points that were not detected. 
#later we an build an extimator which points were well detected but not right in the given data? 
# check if in bounding box is something white? -> claim point as over detected -> no show the picture and add the point to the given data. 
 
def detect_picture(filename="R1_P35_8_90",
                   img_path = "./data/custom_rot/images",
                   pred_label_path = "./data/custom_rot/predicted_labels2",
                   model = None, #models.load_model("config/yolov3-tiny-custom.cfg", "data/custom_philo_generated/checkpoints100/yolov3_ckpt_100.pth"),
                   conf_thres=0.01,
                   nms_thres=0.1,
                   accepted_distance = 35, # how many pixels are allowed between annotated and predicted point to be wrong?
                   stat_all_pics= None, # for statistics is there an statistix object to store over every picture? 
                   output=None, # for visualisation: where should we store the visualized picture? None is possible! 
                   bounding_box = False, # for visualisation: should we show the bounding box?
                   show = True,
                   ac_d = None):

    #catching common errors

    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.txt'): #check if the filename got also its ending. 
        filename= filename[:-len('.jpg')]

    pic_path= f"{img_path}/{filename}.jpg"

    if pred_label_path.endswith('.txt'):
        raise TypeError("in the variable pred_label_path should contain only the directory not specific files.")
    
    if img_path.endswith('.txt'):
        raise TypeError("in the variable img_path should contain only the directory not specific files.")

    if not os.path.exists(pic_path):
            raise FileNotFoundError(f"Bild nicht gefunden: {pic_path}")
 
    #statistics inizialisation: 
    # if stat_all_pics is None: 
    #         stat_all_pics = statistix()
    stat = stat_pic() #create a statistic object for one pic 

    if 'R3' not in filename: # heel filopodia are not annotated in the R3 manual data (this data is incomplete)
    #parameters: 
        
        #load the image
        img = cv2.imread(pic_path)#.split('\n')[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #get the absolute heights and widths of the image to scale the relative numbers back to the original size
        img_height, img_width =img.shape[0], img.shape[1]
        
        #creates the label path to the given image_path
        original_label_path = (pic_path.split('.jpg')[0] + '.txt').replace('images', 'labels')
        
        #load manual labels
        original_labels = np.genfromtxt(original_label_path, delimiter=' ')

    #load coordinates: 
        #put the relative coordinates back to absolute coordinates depending on their rotation:
        pos_man = np.c_[original_labels[:, 1] * img_width, original_labels[:, 2] * img_height]           
        
        boxes = detect.detect_image(model, img, conf_thres=conf_thres, nms_thres=nms_thres)
        if len(boxes) <= 0: 
            print(f"box: no predicted points in {filename}") #raising no error to not desturb the training loops. 
            stat.set_0(man=len(pos_man))
            if stat_all_pics is not None: 
                stat_all_pics.add_pic(stat)
            return stat_all_pics
            
        boxes=np.round(boxes,decimals=2)
        
        pos_pred = np.c_[(boxes[:, 0] + boxes[:, 2])/2, (boxes[:, 1] + boxes[:, 3])/2] #Umrechnung der Koordinaten auf die Mitte der Box.
        
        #store the predicted coordinates in a pred_label file: 
        predicted_labels = np.zeros((pos_pred.shape[0], 5))
        predicted_labels[:, 0] = boxes[:,5]  # Copy class labels
        predicted_labels[:, 2] = (boxes[:, 1] + boxes[:, 3]) / (2 * img_height) # Normalize x_center and y_center
        predicted_labels[:, 1] = (boxes[:, 0] + boxes[:, 2]) / (2 * img_width)
        predicted_labels[:, 3] = (boxes[:, 2] - boxes[:, 0]) / img_width # Normalize width and height
        predicted_labels[:, 4] = (boxes[:, 3] - boxes[:, 1]) / img_height

        #save the predicted labels in a new file
        if pred_label_path is None:
            pred_label_path = original_label_path.replace('labels', 'predicted_labels')
        else: 
            if not os.path.exists(pred_label_path): ### sollte das nicht evtl in eine übergeordnete funktion die nicht per bild geht? 
                print(f"Directory created: {pred_label_path}")
            
            pred_label_path_file = os.path.join(pred_label_path, filename) + '.txt'
        
         # Save predicted labels to a new file this way to influence the float length. 
        with open(pred_label_path_file, 'w') as f: 
            for row in predicted_labels:
                f.write(f"{int(row[0])} {row[1]:.6f} {row[2]:.6f} {row[3]:.2f} {row[4]:.2f}\n") #lets round the predicted coordinates.
    
        if show: 
            print(filename)
            #print(f"Predicted labels saved: {pred_label_path_file}") ### evtl. weglassen man muss nicht alles anzeigen. 
    
    #let's do the statistics:
        
        if not isinstance(stat_all_pics,statistix) and stat_all_pics is not None:
            raise TypeError("stat_all_pics has not the input of a statistix object")

        D = distance.cdist(pos_pred, pos_man, metric = 'euclidean') # distance matrix
        
        stat.calculate(D, accepted_distance)
        
        if stat_all_pics is not None: 
            stat_all_pics.add_pic(stat)
            
        if show: 
            stat.print_values() 
        if show and stat_all_pics is not None: 
            stat_all_pics.print_values() ### do we need them twice? maybe only per picture?
        
        filename=f"{filename}.jpg"
        
        vis_picture(filename=filename,
                    image_path=img_path,
                    label_path= os.path.dirname(original_label_path),
                    pred_label_path=pred_label_path,
                    output_path=output, 
                    bounding_box=bounding_box,
                    show=show,
                    ac_d=ac_d)
                    
    # #add predicted and actual philopodia ends to the image:
    #     for pm in pos_man: 
    #         cv2.circle(img, (int(pm[0]), int(pm[1])), 3, color=(0, 255, 0)) #green is the manual annotation
    #     for pa in pos_pred:
    #         cv2.circle(img, (int(pa[0]), int(pa[1])), 3, color=(0, 0, 255)) #blue is the predicted annotation
    #     plt.figure()
    #     plt.imshow(img)

    else:
        if show: 
            print(f'{filename} is R3 data and not annotated')
            print()
        stat.set_annotated()
        if stat_all_pics is not None:
            stat_all_pics.add_pic(stat)

    return stat_all_pics

#time functions 
# recalculates time from the format d/h/m/s to seconds
def time_to_seconds(time_str):
    days, hours, minutes, seconds = 0, 0, 0, 0
    if 'd' in time_str:
        days = int(time_str.split('d')[0])
        time_str = time_str.split('d')[1]
    if 'h' in time_str:
        hours = int(time_str.split('h')[0])
        time_str = time_str.split('h')[1]
    if 'm' in time_str:
        minutes = int(time_str.split('m')[0])
        time_str = time_str.split('m')[1]
    if 's' in time_str:
        seconds = int(time_str.split('s')[0])
    return days * 86400 + hours * 3600 + minutes * 60 + seconds

def clean_time_str(time_str):
    return time_str.replace('/', '')

# recalculates time from seconds to the format d/h/m/s
def seconds_to_time(seconds):
    days = int(seconds // 86400)
    seconds %= 86400
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds = round(seconds % 60, 2)
    return f"{days}d/{hours}h/{minutes}m/{seconds}s"

# cleans df and gives the wanted columns in the right format
def read_and_clean_csv(file_path):
    df = pd.read_csv(file_path)
    df['training_time'] = df['training_time'].apply(clean_time_str)
    df['ev_time'] = df['ev_time'].apply(clean_time_str)
    df['training_time_seconds'] = df['training_time'].apply(time_to_seconds)
    df['ev_time_seconds'] = df['ev_time'].apply(time_to_seconds)
    return df
### testing

#example of using the function vis_picture:
#vis_picture(filename="R6_P25_8" , output_path="output", show=True,bounding_box=False)#label_path="../../yolo_test/data/custom/labels",
#vis_picture() pred_label_path="data/custom_philo_generated/predicted_labels"
#creating a testing environment for the statistic classes. 
#matrix=np.matrix([[13,14,33,60],[12,18,40,70],[19,19,100,70]])
#statistic = statistic()
#stat = stat_pic()
#stat.calculate(matrix, 15)
#stat.print_values()
#statistic.add_pic(stat)
#statistic.print_values()
#statistic.add_pic(stat)
#statistic.print_values()

#example of using the the statistic class: 
# matrix=np.matrix([[13,14,33,60],[12,18,40,70],[19,19,100,70]])
# statistic = statistic()
# stat = stat_pic()
# stat.calculate(matrix, 15)
# stat.print_values()
# statistic.add_pic(stat)
# statistic.print_values()
# statistic.add_pic(stat)
# statistic.print_values()