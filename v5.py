###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################

#python3 v5.py

import cv2
from mediapipe import solutions
'''
pip install mediapipe
pip install protobuf==3.20.*
'''
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.python.solutions.pose import PoseLandmark
from typing import Mapping, Tuple
from pandas import DataFrame
from math import sqrt, sin
import time
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt




# Initialize mediapipe hands module
mphands = solutions.hands
mpdrawing = solutions.drawing_utils

# Specify the path to your video file
#vidpath = 'F:/Chloe/video_2.mp4'
#file_path = 'F:/Chloe/'

###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################

########################################################
###                    params hebb                   ###
######################################################## 

#toM = 0.35
#toS = 3.5
#Af = 1
#sigmaS = 2
#"sigmaF = 1.5"
#w_inj = 0.5 #poids synaptique I_inj
V0 = 0.01
q0 = 0.01
dt = 0.01
eps = 0.1 
w_inj_1= 0.01 

###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################

########################################################
###               params hand tracking               ###
######################################################## 

_RADIUS = 4
_RED = (48, 48, 255)
_GREEN = (48, 255, 48)
_BLUE = (192, 101, 21)
_YELLOW = (0, 204, 255)
_GRAY = (128, 128, 128)
_PURPLE = (128, 64, 128)
_PEACH = (180, 229, 255)
_WHITE = (224, 224, 224)
_CYAN = (192, 255, 48)
_MAGENTA = (192, 48, 255)

# Hands
_THICKNESS_WRIST_MCP = 10
_THICKNESS_FINGER = 8
_THICKNESS_DOT = 8 #-1

# Hand landmarks
_PALM_LANDMARKS = (HandLandmark.WRIST, HandLandmark.THUMB_CMC,
                   HandLandmark.INDEX_FINGER_MCP,
                   HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.RING_FINGER_MCP,
                   HandLandmark.PINKY_MCP)
_THUMP_LANDMARKS = (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP,HandLandmark.THUMB_TIP) 

_INDEX_FINGER_LANDMARKS = (HandLandmark.INDEX_FINGER_PIP, HandLandmark.INDEX_FINGER_DIP, HandLandmark.INDEX_FINGER_TIP)
_MIDDLE_FINGER_LANDMARKS = (HandLandmark.MIDDLE_FINGER_PIP, HandLandmark.MIDDLE_FINGER_DIP, HandLandmark.MIDDLE_FINGER_TIP)
_RING_FINGER_LANDMARKS = (HandLandmark.RING_FINGER_PIP, HandLandmark.RING_FINGER_DIP, HandLandmark.RING_FINGER_TIP)
_PINKY_FINGER_LANDMARKS = (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP)
_HAND_LANDMARK_STYLE = {
    _PALM_LANDMARKS:
        DrawingSpec( color=_RED, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _THUMP_LANDMARKS: DrawingSpec(
        color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _INDEX_FINGER_LANDMARKS: DrawingSpec( 
        color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _MIDDLE_FINGER_LANDMARKS: DrawingSpec(
            color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _RING_FINGER_LANDMARKS:
        DrawingSpec(
            color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _PINKY_FINGER_LANDMARKS:
        DrawingSpec(
            color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
}

# Hands connections
_HAND_CONNECTION_STYLE = {
    hands_connections.HAND_PALM_CONNECTIONS:
        DrawingSpec(color=_BLUE, thickness=_THICKNESS_WRIST_MCP),
    hands_connections.HAND_THUMB_CONNECTIONS:
        DrawingSpec(color=_CYAN, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_INDEX_FINGER_CONNECTIONS:
        DrawingSpec(color=_CYAN, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_MIDDLE_FINGER_CONNECTIONS:
        DrawingSpec(color=_CYAN, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_RING_FINGER_CONNECTIONS:
        DrawingSpec(color=_CYAN, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_PINKY_FINGER_CONNECTIONS:
        DrawingSpec(color=_CYAN, thickness=_THICKNESS_FINGER)
}


###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################

def get_hand_landmarks_style(): # -> Mapping[int, DrawingSpec]:
    """Returns the default hand landmarks drawing style.
    Returns:
    A mapping from each hand landmark to its default drawing spec.
    """
    hand_landmark_style = {}
    for k, v in _HAND_LANDMARK_STYLE.items():
        for landmark in k:
            hand_landmark_style[landmark] = v
    return hand_landmark_style


def get_hand_connections_style(): # -> Mapping[Tuple[int, int], DrawingSpec]:
    """Returns the default hand connections drawing style.
      Returns:
      A mapping from each hand connection to its default drawing spec.
    """
    hand_connection_style = {}
    for k, v in _HAND_CONNECTION_STYLE.items():
        for connection in k:
            hand_connection_style[connection] = v
    return hand_connection_style


# Set the desired window width and height
def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################

def hands_tracking():
    mp_drawing = solutions.drawing_utils
    #mp_drawing_styles = solutions.drawing_styles
    
    #initialize a neuron
    neur = create_NRS(nom='RS1', I_inj = 0.0, w_inj=w_inj_1, V=0.001, sigmaS=2.0 ,sigmaF=1.5,Af=1,q=0.01) #winj 1 def en haut!!!!!!!!!!!!!!!!!!!!
    
    #Create 5 empty lists
    L, list_V, list_T, list_I_inj, list_sigmaS = [],[],[],[],[]
    
    # Initialize video capture
    vidcap = cv2.VideoCapture(0) #(vidpath) #pour une vidéo enregistrée
    # Initialize hand tracking
    with mphands.Hands(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        start_time = time.time()
        while vidcap.isOpened():
            ret, frame = vidcap.read()
            if not ret:
                break

            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the frame for hand tracking
            processFrames = hands.process(rgb_frame)
            
            # Draw landmarks on the frame
            if processFrames.multi_hand_landmarks:
                for hand_landmarks in processFrames.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mphands.HAND_CONNECTIONS,
                        get_hand_landmarks_style(),
                        get_hand_connections_style()
                        )
                    #Follow position of the hand landmark[0] for point 0
                    wrist = hand_landmarks.landmark[0]
                    x_pixel = int(wrist.x * frame.shape[1])
                    y_pixel = int(wrist.y * frame.shape[0])
                    cv2.circle(frame, (x_pixel, y_pixel), 10, (200, 0, 200), -1) # Lilac color for the wrist point
                    
                    # Keep positions in a list
                    lmlist=(x_pixel,y_pixel)
                    
                    t = time.time() - start_time
                    I_inj = x_pixel #ici! ---------------------------------------------------------------------
                    neur.I_inj = I_inj
                    V, sigmaS, q = update_neuron(neur, t)
                    
                    list_V.append(V)
                    list_T.append(t)
                    list_I_inj.append(I_inj)
                    list_sigmaS.append(sigmaS)
                    L.append(lmlist)
        
            # Resize the frame to the desired window size
            #resized_frame = cv2.resize(frame, (winwidth, winheight))
            resized_frame = rescale_frame(frame, percent=150)

            # Display the resized frame
            cv2.imshow('Hand Tracking', resized_frame)
        
            # Exit loop by pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release the video capture and close windows
    vidcap.release()
    cv2.destroyAllWindows()  
         
    return L, list_V, list_T, list_I_inj, list_sigmaS

###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################

class NeuroneRS : pass  #car pas vraie prog obj

########################################################
###             état du neurone instant t            ###
########################################################  

def create_NRS(nom, I_inj, w_inj, V, sigmaS,sigmaF, Af, q):
    neurone = NeuroneRS()
    neurone.nom =  nom 
    neurone.I_inj = I_inj  # courant d'entrée:  peut etre une somme de courants
    neurone.w_inj =  w_inj  # poids synaptique courant d'entrée 
    neurone.V=  V    # output neurone- on rappelle V = Vs et la dérivée peut se noter y au lieu de Vs point-
    neurone.sigmaS =  sigmaS 
    neurone.sigmaF =  sigmaF 
    neurone.Af =Af
    neurone.q = q  #signal slow current
    neurone.toM = 0.35
    neurone.toS = 3.5    
    return neurone



########################################################
###          fonctions basiques nécessaire           ###
######################################################## 

#fonction crée pour raccourcir la notation, aussi appelée F dans les articles
def F(n):
    return n.V-n.Af*np.tanh((n.sigmaF/n.Af)*n.V)

#fonction input enlevée 

def f_V(n, t):
    return  - (F(n)+ n.q - n.w_inj *n.I_inj)/n.toM

def f_Q(n,t):
    return (-n.q + n.sigmaS*n.V)/n.toS

def f_sigmaS(n,t):
    y = f_V(n,t) #premier appel de la presque derivée 
    racine = sqrt(y**2 + (n.V)**2) 
    radical_1 = sqrt(n.toM * n.toS) #Constant value
    
    if racine == 0:
        quotient = 0
    else:
        quotient = y /racine   
        next_sigma = 2 * eps * n.I_inj * radical_1 * sqrt(abs(1+ n.sigmaS - n.sigmaF))*quotient
    
    return next_sigma

def update_neuron(neurone, t):
    neurone.sigmaS += dt * f_sigmaS(neurone, t)
    print (neurone.sigmaS) # test ---------------------------------------------------------
    neurone.V += dt * f_V(neurone, t)
    neurone.q += dt * f_Q(neurone, t)
    return neurone.V, neurone.sigmaS, neurone.q


########################################################
###                      dessin                      ###
########################################################  

#cette verson de plot a des problèmes d'échelle
'''
def plot(Vs1, Ts1, I_inj1, sigmaS):
    plt.figure(figsize=(10,6))
    plt.plot(Ts1, Vs1, '-m', label ='Vs - signal sortie')
    plt.plot(Ts1, I_inj1,'y--', label ='I_inj - signal de forçage')
    plt.plot(Ts1, sigmaS, 'b-', label= 'sigma S')
    
    # labels axes
    plt.xlabel('temps')
    plt.ylabel('intensité')

    # titre haut
    plt.suptitle(
        "Etude théorique CPGs",
        fontsize=16,
        fontweight="bold",
        x=0.126,
        y=0.98,
        ha="left",
        )
    # sous titre haut
    plt.title(
        "avec un neurone et apprentissage hebbien",
        fontsize=14,
        pad=10,
        loc="left",
        )
    # le carré avec les labels
    plt.legend(loc='upper left', fontsize=8)  

    plt.show()

'''

def plot(Vs1, Ts1, I_inj1, sigmaS):
    fig, axs = plt.subplots(2, 1, figsize=(10, 18))
    ##### titre haut
    plt.suptitle(
        "Etude pratique CPGs - 1neur - input webcam - w_inj= " + str(w_inj_1) + "  - eps = " + str(eps),
        fontsize=14,
        #fontweight="bold",
        x=0.09,
        y=1,
        ha="left",
    )
    '''
    #section qui ne marche plus avec les subplots
    ##### sous titre haut
    plt.title(
        '1 neurone, w_inj= ' + str(neur1.w_inj) + ', eps = ' + str(eps) ,
        fontsize=14,
        pad=10,
        x=0.08,
        y=1,
        loc="left",
    )'''

    # First subplot
    axs[0].plot(Ts1, I_inj1, 'y--', label='I_inj - signal de forçage')
    axs[0].plot(Ts1, Vs1, '-m', label='Vs - signal sortie')
    axs[0].set_xlabel('temps')
    axs[0].set_ylabel('potentiel')
    axs[0].legend(loc='upper right')
    #axs[1].set_title('I_inj - signal de forçage')
    
    '''
    # Second subplot
    axs[1].plot(Ts1, Vs1, '-m')           #label='Vs - signal sortie')
    axs[1].set_xlabel('temps')
    axs[1].set_ylabel('Vs - signal sortie')
    #axs[0].legend(loc='upper right')
    #axs[0].set_title('Vs - signal sortie')
    '''

    # Third subplot
    axs[1].plot(Ts1, sigmaS, 'b-')      #label='sigma S')
    axs[1].set_xlabel('temps')
    axs[1].set_ylabel('sigma S')
    #axs[2].legend(loc='upper right')
    #axs[2].set_title('sigma S')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plot
    plt.show()


###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################


def main():
    L, list_V, list_T, list_I_inj, list_sigmaS = hands_tracking()
    plot(list_V, list_T, list_I_inj, list_sigmaS)
    '''
    #Savinf csv file of positions
    df=DataFrame(L)
    print(df)
    df.to_csv(file_path+'position.csv', sep=';', index = True, header=None)
    '''
    
    
if __name__ == '__main__':
    main()

###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################