import gradio as gr
import os
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import tensorflow
import time
from transformers import pipeline


fix_spelling = pipeline("text2text-generation",model="oliverguhr/spelling-correction-english-base")
text = ""
recording = False
labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
def run_model():
    global text
    global recording
    text = ''

    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    offset = 20
    imgsize = 350
    classifier = Classifier('E:\ASLDecetor\\new_model\keras_model.h5', labelsPath='E:\ASLDecetor\\new_model\labels.txt')

    recording = True
    cur_time = time.time()
    while recording:
        success, img = cap.read()
        hands, img = detector.findHands(img)

        if hands:

            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape

            # imgWhite[:imgCropShape[0],:imgCropShape[1]] = imgCrop

            if h > w:
                k = imgsize / h
                w_cal = round(k * w)
                img_resize = cv2.resize(imgCrop, (w_cal, imgsize))
                img_resize_shape = img_resize.shape
                w_gap = round((imgsize - w_cal) / 2)
                imgWhite[:, w_gap:w_cal + w_gap] = img_resize
                pred, index = classifier.getPrediction(imgWhite)


            else:
                k = imgsize / w
                h_cal = round(k * h)
                img_resize = cv2.resize(imgCrop, (imgsize, h_cal))
                img_resize_shape = img_resize.shape
                h_gap = round((imgsize - h_cal) / 2)
                imgWhite[h_gap:h_cal + h_gap, :] = img_resize
                pred, index = classifier.getPrediction(imgWhite)

            cv2.imshow('ImageWhite', imgWhite)

            if time.time() - cur_time >= 5:
                cur_time = time.time()
                print(labels[index])
                text +=labels[index]
        cv2.imshow('Image', img)
        key = cv2.waitKey(1)
    return text

def stop():
    global recording
    global text
    recording = False
    return text

def correct(text):
    corrected_text = fix_spelling(text, max_length=2048)[0]['generated_text']
    return corrected_text

with gr.Blocks() as dmeo:
    output = gr.Text(label='output')
    with gr.Row():
        read_btn = gr.Button('read')
        stop_btn = gr.Button('stop')
        correct_btn = gr.Button('correct')
    correction = gr.Text(label='did you mean?')

    read_btn.click(fn =run_model , inputs=[] , outputs = output)
    stop_btn.click (fn = stop , inputs = [] , outputs=[output])
    correct_btn.click(fn = correct , inputs=[output] , outputs = [correction])
dmeo.launch()