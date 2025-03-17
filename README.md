# Filopodia DETECTION IN THE LAMINA Plexus OF Drosophila melanogaster USING THE NEURAL NETWORK ARCHITECTURE YOLOV3
In this GitHup, you will learn how to apply the YOLOv3 object detection algorithm to detect filopodia tips while nerve cell development in the Lamina Plexus of Drosophila Melanogaster[[1]](#q1). The algorithm is applied to a small (354) and inclompletely annotated dataset, which was acquired in vivo from living pupae but therefore reaches a sensitivity up to 0.9. 

This project is part of the bachelor's thesis of Vinzent Aschir. The corresponding paper and the analysed data are avaidable upon request. 

To initialize YOLOv3, the implementation of [[2]](#q2) was used and embetted in this project. YOLOv4 and YOLOv7 weights are also compatible with this implementation.

<p align="center"><img src="https://github.com/vinzi-a/ba_thesis_aschir/blob/main/visualisation/R6_P35_5.jpg?raw=true" width="480"\></p>

## Installation
For following this souce code, clone your repository with: 

```sh
git clone https://github.com/vinzi-a/ba_thesis_aschir.git
cd ba_thesis_aschir
```
Before installing all the required packages create a virtual environment:
 ```sh
python -m venv yolov3 # create a venv
source yolov3/bin/activate # activate it
 
```
Use this command for dactivating your virtual environment if necessary: 
```sh 
deactivate 
```
Please install jupypter in your virtual environment to be able to follow the implemented jupyter notepads
```sh 
pip install jupyter
```
##  **Filopodia Detection in *Drosophila*** 
For the YOLOv3 implementation on **filopodia tip detection**, follow the Jupyter Notebook:
🪰 **[yolov3.ipynb](./yolov3.ipynb)** – Implementation by Vinzent Aschir on the *Drosophila* dataset. 

## **General YOLOv3 Instructions** 
For general YOLOv3 instructions, refer to:
📜 **[general_instruction.md](./general_instruction.md)** from [2](#q2)

<p align="center"><img src="https://github.com/eriklindernoren/PyTorch-YOLOv3/raw/master/assets/messi.png" width="480"\></p>
<p align="center"><img src="https://github.com/eriklindernoren/PyTorch-YOLOv3/raw/master/assets/giraffe.png" width="480"\></p>
## sources 🔗

<a name="q1"></a> **[1] 🔗 initial YOLOv3-algorithm**: [P. J. Reddie](https://pjreddie.com/darknet/yolo/) 

<a name="q2"></a> **[2] 🔗 followed  YOLOv3-implementation**: [E. Lindernoren](https://github.com/eriklindernoren/PyTorch-YOLOv3) 

<a name="q2"></a> [3] 🔗 **CustomYOLOv3-custom-Tiny model arichtecture**: [Gunjan Chourasia](https://github.com/GunjanChourasia/pytorch-yolo-v3-custom?tab=readme-ov-file) 

<a name="q2"></a> [4] 🖥️ **Code and data from Eric Reifenstein**  

## contact
📧 contact: <a href="vinzent.aschir@web.de">if you have any questions write me a mail 😊</a>
