#this is a test file for finding the right threshold for avoid coinciding points in the data generation and improving the creation of these points. 

####functions#####

def rotate_coord(x, y, angle):
    """
    rotate coordinates by angle, angle in rad
    """
    return x*np.cos(angle) - y*np.sin(angle), x*np.sin(angle) + y*np.cos(angle)



def d_coinciding(theta1, theta2, dist1, dist2):
    #print(1 - math.cos((theta1 - theta2)*(min(dist1, dist2))))
    return 1000*(1 - math.cos((theta1 - theta2)*(min(dist1, dist2)/0.7071))) # im Distanzteil wird durch die am weitesten entfternt mögliche Distanz genormt, die Distanz vom Ursprung (0.5, 0.5) zum Eckpunkt (1, 1) ist etwa 0.7071	



import math 
def add_non_coinciding_points(num_points=10, centerx=0.5, centery=0.5, threshold=35.06):
    #initialize the matrices:
    distance_matrix = np.zeros(num_points)
    angle_matrix = np.zeros(num_points)
    xfil, yfil = np.zeros(num_points), np.zeros(num_points) #initialize the arrays for coordinate 
    coinciding_score = np.zeros((num_points, num_points)) #initialize the array for the coinciding score

    for i in range(num_points): # add points consecutively         
        xfil[i], yfil[i] = 0.8*np.random.rand() + 0.1, 0.8*np.random.rand()+0.1 # the image will be [0, 1]
        distance_matrix[i] = np.sqrt((xfil[i]-centerx)**2 + (yfil[i]-centery)**2) # sollten wir hier xcenter und ycenter subtrahieren?
        angle_matrix[i] = np.arctan2(yfil[i]-centery, xfil[i]-centerx) #der Winkel zwischen dem Ortsvektor und der x-Achse
 
        for j in range(i-1): 
            while d_coinciding(angle_matrix[i], angle_matrix[j], distance_matrix[i], distance_matrix[j]) < threshold:
                print(f"coinciding points detected ({i}, {j}), with distance: {d_coinciding(angle_matrix[i], angle_matrix[j], distance_matrix[i], distance_matrix[j])}")

                #neuen Punkt generieren und Distanzen und Winkel neu berechnen	
                xfil[i], yfil[i] = 0.8*np.random.rand() + 0.1, 0.8*np.random.rand()+0.1
                distance_matrix[i] = np.sqrt((xfil[i])**2 + (yfil[i])**2)
                angle_matrix[i] = np.arctan2(yfil[i], xfil[i])    

    minima= np.zeros(num_points)
    for i in range(num_points):  
        for j in range(num_points): #to get a feeling of the distance measure
            if i == j: 
                coinciding_score[i, j] = 100
                continue
            coinciding_score[i, j] = round(d_coinciding(angle_matrix[i], angle_matrix[j], distance_matrix[i], distance_matrix[j]),7)
        #print(min(coinciding_score[i,:]))
        minima[i] = min(coinciding_score[i,:])

    #print(f"Spanne in der Distanz: {max(distance_matrix)}, {min(distance_matrix)}")
    #print(f"Spanne im Winkel: {max(angle_matrix)}, {min(angle_matrix)}")

    #generates 10 random points within the range of 0.1 and 0.9
    #print(create_dist_matrix(xfil, yfil))
    #print(distance_matrix)
    #print(angle_matrix) 
    return xfil, yfil, minima


#%%
# generate surrogate data (Ersatzdaten)
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import gaussian_kde
from scipy.spatial import distance
from tools import vis_picture
import time 
#functions: 


#hier könnte man noch eine add function einbauen, die die Punkte, die zu nah beieinander liegen, entfernt und neue kreiert. und die Distanz und winkelmatrix updatet. Das wäre nicht so zeitintensiv. 

start_time = time.time()

nrep = 20 #wir wollen 1000 Bilder generieren
count = 0
centerx, centery = 0.5, 0.5
threshold = 35.05#1.3 #initial threshold
cv=0 
re_p = 0 
while count < nrep:
    print("threshold", threshold)
    num_points=random.randint(
        6,10) #zufällige Anzahl von Punkten zwischen 6 und 10
    xfil, yfil, minima = add_non_coinciding_points(num_points=num_points, centerx=centerx, centery=centery, threshold=threshold) #generates the points
    print("num_points", num_points)  
    
    # write labels file
    f = open("/mnt/c/Users/vinze/Dropbox/Universität/8.Bachelorarbeit/yolo2/PyTorch-YOLOv3/data/generated/labels/"+ str('%04d' % count) + ".txt", "w") #ich habe die labels noch nicht. 
    for i in range(len(xfil)): #how the lables look alike inside the file:
        # label_idx x_center y_center width height 
        f.write("0 " + str(xfil[i]) + ' ' + str(yfil[i]) + ' 0.05 0.05\n')
    f.close()
    print("/mnt/c/Users/vinze/Dropbox/Universität/8.Bachelorarbeit/yolo2/PyTorch-YOLOv3/data/generated/labels/"+ str('%04d' % count) + ".txt") 
    # create and save image
    xfil_int = np.array([])
    yfil_int = np.array([])
    for i in range(len(xfil)): #draws the line in between the points and the center
        samp = int(np.sqrt((xfil[i]-centerx)**2 + (yfil[i]-centery)**2)//0.01)
        xfil_int = np.r_[xfil_int, np.linspace(centerx, xfil[i], samp)[1:]]
        yfil_int = np.r_[yfil_int, np.linspace(centery, yfil[i], samp)[1:]]

    kde = gaussian_kde(np.c_[xfil_int, yfil_int].T, bw_method=0.1)
    x_flat, y_flat = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
    x, y = np.meshgrid(x_flat, y_flat)
    grid_coords = np.append(x.reshape(-1,1), y.reshape(-1,1), axis=1)    
    z = kde(grid_coords.T)
    z = z.reshape(100, 100)
    # plt.figure()
    # plt.imshow(z)
    # plt.scatter(xfil*100, yfil*100)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(10, 10)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(z, cmap='Greys_r', aspect='equal')#, origin='lower')
    plt.xticks([])
    plt.yticks([])
    # plt.tight_layout()
    # plt.scatter(xfil*100, yfil*100, c='w')
    filename = str('%04d' % count + '.jpg')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    print(filename)
    plt.savefig('/mnt/c/Users/vinze/Dropbox/Universität/8.Bachelorarbeit/yolo2/PyTorch-YOLOv3/data/generated/images/' + filename, bbox_inches = 'tight',
    pad_inches = 0, dpi=10)
    plt.close(fig)
    
    #lets visualize the generated image to get a feeling of the threshold for the created distance measure: 
    vis_picture(filename=str('%04d' % count),
        image_path="data/generated/images", 
        label_path="data/generated/labels",
        pred_label_path=None,
        output_path="gen_output",
        show=True,
        bounding_box=False)
    
    print("Enter how many coinciding points you can see in the picture: ")
    num_coinciding = int(input())
    cv += num_coinciding
    if num_coinciding == 0:
        count += 1
        continue

    #lets update the threshold by finding the minimal distance of these coinciding points: 
    indices=np.argpartition(minima, num_coinciding)[:num_coinciding]
    #print("indices",indices)
    #print(minima[indices])
    if threshold < max(minima[indices]):
        threshold = max(minima[indices])
    count += 1   
print("coincident points: ", cv)
end_time = time.time()  

#für Distanzmaß: 1000*(1 - math.cos((theta1 - theta2)*(min(dist1, dist2))) mit arctan2 eines Punktes 
    #bei 20 Wiederholungen: threshold 0.8757452 -> 0.9
    #bei 20 Wiederholungen: threshold 0.9 -> 1.3
    #bei 20 Wiederholungen: threshold 0 -> 35.0542646 (12 coinciding points still detected)
    #bei 20 Wiederholungen: threshold 35.0542646 -> 35.0542646 (0 coinciding points detected)
    #Was ist das beste Distanzmaß? wenn man 20 Wiederholungen ein Threshold trainiert, wie viele Coinciding points werden dann bei 20 weiteren Wiederholungen nicht erkannt? 