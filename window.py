from tkinter import *
from PIL import Image
import cv2
import numpy as np

import torch
from ConvNet import ConvNet

#LOADING MODEL
device = torch.device('cpu')
FILE = "model.pth"
loaded_model = ConvNet()
loaded_model.load_state_dict(torch.load(FILE,map_location=device))
loaded_model.eval()

#WINDOW
class PaintBox(Frame):
    def __init__(self):
        Frame.__init__(self)
        self.pack(expand=YES, fill=BOTH)
        self.master.title("A simple paint program")
        self.master.geometry("400x400")

        self.message = Label(self, text="Draw a number!!!")
        self.message.place(relx=0.39, rely=0.05)

        self.text = StringVar(self)
        self.text.set("Start drawing!")

        self.text2 = StringVar(self)

        self.answer = Label(self, textvariable=self.text, bg="white")
        self.answer.place(relx=0.45, rely=0.9)

        self.savebut = Button(self, text="Save", command=self.save)
        self.savebut.place(relx=0.35, rely=0.9)

        self.clear = Button(self, text="clear", command=self.clr)
        self.clear.place(relx=0.1, rely=0.9)

        # create Canvas component
        self.myCanvas = Canvas(self, width=300, height=300, bg="white")
        self.myCanvas.place(relx=0.5, rely=0.5, anchor=CENTER)

        # bind mouse dragging event to Canvas
        self.myCanvas.bind("<B1-Motion>", self.paint)

        #self.Image = Image.new("RGB",(300,300),"white")

    def clr(self):
        self.myCanvas.delete("all")
        self.text.set("Start drawing!")

    def paint(self, event):
        x1, y1 = (event.x - 4), (event.y - 4)
        x2, y2 = (event.x + 4), (event.y + 4)
        self.myCanvas.create_oval(x1, y1, x2, y2, fill="black",width=18)

    def save(self):
        self.myCanvas.postscript(file="post.eps")
        img = Image.open("post.eps")
        img.save("post.png", "png")

        img_array = cv2.imread("post.png",cv2.IMREAD_GRAYSCALE)
        img_array=abs(img_array-255)
        img_array=img_array.astype("float32")
        img_array=cv2.resize(img_array,(28,28))

        input_data = torch.from_numpy(img_array.reshape(1,1,28,28))
        outputs = loaded_model(input_data)
        class_probabilities = torch.softmax(outputs,1)
        prob,pred = torch.max(class_probabilities,1)
        self.text.set(f'You typed {pred.item()} with probability {prob.item()}')
        
def main():
    PaintBox().mainloop()

if __name__ == "__main__":
    main()