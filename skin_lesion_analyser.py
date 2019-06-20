##############################################################################
# Project Title: CNN approach for accurate detection of skin lesions
#
# Program name: Skin Lesion Analyser (SLA.exe)
#
# Author: Jonatan González Rodríguez
#
# Data: 2019
#
# Python version:  Python 3.6.8 under Windows 10
#
##############################################################################

# Module Imports

from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
from PIL import Image, ImageTk

# Functions and UI

root = Tk()
root.title("Skin Lesion Analyser v0.1")

file_path = ""

def load_image():
    """
    This function proceeds to load the image selected by the user and reshape it for prediction purposes
    """
    global file_path
    global img
    file_path = filedialog.askopenfilename()
    img = np.asarray(Image.open(file_path).resize((100,75)))
    img = np.reshape(img, [1,75,100,3])
    return img

def showImg(file_path):
    """
    This function takes the image supplied by the user and displays it in the UI of the app.
    """
    load = Image.open(file_path)
    load = load.resize((150,150), Image.ANTIALIAS) # Image.ANTIALIAS is used for the downsampling of the image in the resize
    render = ImageTk.PhotoImage(load)
    # labels can be text or images
    img = Label(root, image=render)
    img.image = render
    img.place(relx=0.5, rely=0.3)

def show_and_load():
    """
    This function takes into account both functions previously described. This was done to be used with the UI button due to the fact that the command argument only
    accepts one funcion as variable.
    """
    load_image()
    showImg(file_path)

def predict():
    """
    Load the model and predicts the lesion based on the dictionary provided. Uses tensorflow as the backend to predict.
    """
    label_classes = {0:'Actinic keratoses', 1:'Basal cell carcinoma', 2:'Benign keratosis-like lesions', 3:'Dermatofibroma', 4: 'Melanocytic nevi', 5:'Melanoma', 6:'Vascular lesions'}
    my_model = load_model('C:\\Users\\PCCom\\Desktop\\Python\\Data\\model.h5')
    pred = my_model.predict(img)
    pred = pred.argmax(axis=1)[0]
    label = label_classes.get(pred)
    label = Label(root, text="Type of skin lesion: \n" + label, background="#71c7e1", font=("Times New Roman", 12,'bold'))
    label.place(relx=0.1, rely=0.5)

ttk.Button(root, text="Browse file...", command=show_and_load).place(relx=0.99, rely=0.5, anchor=E, relheight=0.5) # Browse file button
ttk.Button(root, text="Predict lesion type", command=predict).place(relx=0.35, rely=0.99, anchor=S) # Predict button
ttk.Label(root, text="Skin Lesion Analyser", background="#71c7e1", font=("Times New Roman", 35,'bold')).place(relx=0.5, rely=0.1, anchor=CENTER) # UI title

root.configure(background="#71c7e1")
root.geometry("500x300+700+300") # Geometry and start up positions in the screen.
root.resizable(FALSE, FALSE) # Makes the size fixed and not resizable
root.iconbitmap(r'C:\Users\PCCom\Desktop\Python\Data\dataset-card.ico')
root.mainloop()