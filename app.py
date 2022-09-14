import gradio as gr
import tensorflow as tf
import numpy as np
from numpy import asarray
from datetime import datetime


model = tf.keras.models.load_model("simple-CNN-model.2022-8-9.hdf5")

def image_predict(img):
    """
    Displays dominant colors beyond a given threshold.
    img : image input, for ex 'blue-car.jpg'
    isize: input image size, for ex. 227
    thr: chosen threshold value
    """
    thr=0
    global model
    if model is None:
        model = tf.keras.models.load_model("models/simple-CNN-model.2022-8-9.hdf5")
        
    data = np.asarray(img)
        
    ndata = np.expand_dims(data, axis=0)
    y_prob = model.predict(ndata/255)
    #y_prob.argmax(axis=-1)
    
    now = datetime.now()
    print("--------")
    print("data and time: ", now)
            
    colorlabels = ['beige', 'black', 'blue', 'brown', 'gold', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'silver', 'tan', 'white', 'yellow']
    coltags = [sorted(colorlabels)[i] for i in np.where(np.ravel(y_prob)>thr)[0]]
    colprob = [np.ravel(y_prob)[i] for i in list(np.where(np.ravel(y_prob)>thr)[0])]
    
    if len(coltags) > 0:
        response = []
        for i,j in zip(coltags, colprob):
            #print(i,j)
            resp = {}
            resp[i] = float(j)
            response.append(resp)
        d = dict(map(dict.popitem, response))
        print('colors: ', d)
               
        return dict(d)

    else:
        return str('No label was found')

iface = gr.Interface(
    title = "Object color tagging",
    description = "App classifying objects on different colors",
    article = "<p style='text-align: center'><a href='https://www.rrighart.com' target='_blank'>Webpage</a></p>",
    fn=image_predict,
    inputs=gr.Image(shape=(227,227)), 
    outputs=gr.Label(),
    examples=['shoes1.jpg', 'shoes2.jpg'],
    enable_queue=True,
    interpretation="default",
    debug=True
    )
iface.launch()
