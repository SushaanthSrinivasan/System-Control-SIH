from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import keyboard
import key
import tkinter, win32api, win32con, pywintypes
import webbrowser
from threading import Timer
from selenium import webdriver

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import key

app = Flask(__name__)

def get_keyboard(current, prev):
    key.do_action(current, prev)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_styled_landmarks(image, results):
    # Draw face connections
    '''mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
    
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) '''
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

dynamic_actions = np.array(['Hello', 'Scroll Up', 'Scroll Down', 'Maximize', 'Minimize', 'Show Desktop'])
static_actions = np.array(['Select All', 'Enter', 'Copy', 'Paste', 'Call Me', 'Backspace', 'Switch Windows', 'Backspace', 'Fist', 'Smile'])

def get_model():
    input_shape = (30,1662)
    dynamic_hand_model = Sequential()
    dynamic_hand_model.add(LSTM(128, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
    dynamic_hand_model.add(LSTM(64, recurrent_dropout=0.35, return_sequences=False))
    dynamic_hand_model.add(Dropout(0.2))
    dynamic_hand_model.add(Dense(128, activation='relu'))
    dynamic_hand_model.add(Dense(64, activation='relu'))
    dynamic_hand_model.add(Dropout(0.4))
    dynamic_hand_model.add(Dense(128, activation='relu'))
    dynamic_hand_model.add(Dense(64, activation='relu'))
    dynamic_hand_model.add(Dropout(0.4))
    dynamic_hand_model.add(Dense(48, activation='relu'))
    dynamic_hand_model.add(Dropout(0.4))
    dynamic_hand_model.add(Dense(dynamic_actions.shape[0], activation='softmax'))

    return dynamic_hand_model



static_hand_model = load_model('mp_hand_gesture')
dynamic_hand_model = get_model()
opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
dynamic_hand_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy', 'accuracy'])
dynamic_model_path = os.path.join("Weights\weights-improvement-1911-1.00.hdf5")
dynamic_hand_model.load_weights(dynamic_model_path)

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


sequence = []
sentence = []
predictions = []
threshold = 0.5
dynamic_prediction = ''

holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

#UI Logic

def display(className):
    label = tkinter.Label(text=className, font=('Helvetica','50'), fg='light green', bg='black')
    label.master.overrideredirect(True)
    label.master.geometry("+0+0")
    label.master.lift()
    label.master.wm_attributes("-topmost", True)
    label.master.wm_attributes("-disabled", True)
    label.master.wm_attributes("-transparentcolor", "white")

    hWindow = pywintypes.HANDLE(int(label.master.frame(), 16))
    # http://msdn.microsoft.com/en-us/library/windows/desktop/ff700543(v=vs.85).aspx
    # The WS_EX_TRANSPARENT flag makes events (like mouse clicks) fall through the window.
    exStyle = win32con.WS_EX_COMPOSITED | win32con.WS_EX_LAYERED | win32con.WS_EX_NOACTIVATE | win32con.WS_EX_TOPMOST | win32con.WS_EX_TRANSPARENT
    win32api.SetWindowLong(hWindow, win32con.GWL_EXSTYLE, exStyle)

    label.pack()
    label.after(1000, label.master.destroy)
    label.mainloop()

sequence = []
sentence = []
predictions = []
threshold = 0.5
dynamic_prediction = ''

holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

for cam in range(5):
    cap = cv2.VideoCapture(cam)

    ret, frame = cap.read()

    if(frame is not None):
        break

cap.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture(cam)

current = ''
prev = ''
count = 0

def classify():
    global cap, classNames, model, mpHands, dynamic_actions, mpDraw, hands, current, count, gesture_list, sequence, keypoints, prev, current, predictions
    while True:
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)

        if(results.right_hand_landmarks and (results.left_hand_landmarks is None)):
            sequence = []
            x, y, c = frame.shape

            # Flip the frame vertically
            frame = cv2.flip(frame, 1)
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            

            # Get hand landmark prediction
            result = hands.process(framergb)


            # print(result)
            
            
            className = ''

            # post process the result
            if result.multi_hand_landmarks:
                landmarks = []
                
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        # print(id, lm)
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)

                        landmarks.append([lmx, lmy])

                    # Drawing landmarks on frames
                    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                    # Predict gesture
                    prediction = static_hand_model.predict([landmarks])
                    # print(prediction)
                    classID = np.argmax(prediction)
                    className = static_actions[classID]
                    #print(className)
                    if(className != current):
                        count = 0
                        current = className

                    elif(className == current):
                        count += 1

                    if(count > 5):
                        count = 0
                        #keyboard.write(className + "\n")
                        #print(className)
                        current=className
                        get_keyboard(current, prev)
                        predictions.append(className)
                        #display(className)
                        prev = current              
                        
                        
                    

            # show the prediction on the frame
            frame = cv2.flip(frame, 1)

            cv2.putText(frame, className, (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)



        elif((results.right_hand_landmarks is None) and results.left_hand_landmarks):
            #print(results)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

                
            if len(sequence) == 30:
                res = dynamic_hand_model.predict(np.expand_dims(sequence, axis=0))[0]
                className = (dynamic_actions[np.argmax(res)])
                predictions.append(className)
                get_keyboard(className, "")
                #display(className)
                sequence = []
                #time.sleep(1)

            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            cv2.putText(image, className, (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            frame = image.copy()

        #cv2.imshow("Output", frame)

        #if cv2.waitKey(1) == ord('q'):
        #    break

        #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

      

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

@app.route('/video_feed')
def video_feed():
    return Response(classify(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/list')
def glist():
    return render_template('list.html', gesture_list=predictions[-15:])

@app.route('/info')
def info():
    return render_template('info.html')

if __name__ == '__main__':
    #Timer(1, open_browser).start();

    #if not os.environ.get("WERKZEUG_RUN_MAIN"):
    #    webbrowser.open_new('http://127.0.0.1:5000/')

    app.run(port = 8000, host = "127.0.0.1")
