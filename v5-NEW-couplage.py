import cv2
from mediapipe import solutions
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.python.solutions.pose import PoseLandmark
from typing import Mapping, Tuple
from pandas import DataFrame
from math import sqrt, sin
import time
from random import randint
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

# Initialize mediapipe hands module
mphands = solutions.hands
mpdrawing = solutions.drawing_utils

# Specify the path to your video file
vidpath = 'F:/Chloe/video_2.mp4'
file_path = 'F:/Chloe/'

# parameters for Hebbian------------------------------------------------------------------------------#
V0 = 0.01
q0 = 0.01
dt = 0.01
eps = 0.5

# parameters for hands tracking-----------------------------------------------------------------------#
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
_THICKNESS_WRIST_MCP = 5
_THICKNESS_FINGER = 4
_THICKNESS_DOT = -1

# Hand landmarks
_PALM_LANDMARKS = (HandLandmark.WRIST, HandLandmark.THUMB_CMC,
                   HandLandmark.INDEX_FINGER_MCP,
                   HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.RING_FINGER_MCP,
                   HandLandmark.PINKY_MCP)
_THUMP_LANDMARKS = (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP)

_INDEX_FINGER_LANDMARKS = (HandLandmark.INDEX_FINGER_PIP, HandLandmark.INDEX_FINGER_DIP, HandLandmark.INDEX_FINGER_TIP)
_MIDDLE_FINGER_LANDMARKS = (HandLandmark.MIDDLE_FINGER_PIP, HandLandmark.MIDDLE_FINGER_DIP, HandLandmark.MIDDLE_FINGER_TIP)
_RING_FINGER_LANDMARKS = (HandLandmark.RING_FINGER_PIP, HandLandmark.RING_FINGER_DIP, HandLandmark.RING_FINGER_TIP)
_PINKY_FINGER_LANDMARKS = (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP)
_HAND_LANDMARK_STYLE = {
    _PALM_LANDMARKS:
        DrawingSpec(color=_RED, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
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


#---------------HEBBIAN------------------------------------------------------------------------------#
class NeuroneRS:
    pass  # car pas vraie prog obj

########################################################
###             état du neurone instant t            ###
########################################################
def create_NRS(nom, I_inj, w_inj, V, sigmaS, sigmaF, Af, q):
    neurone = NeuroneRS()
    neurone.nom = nom
    neurone.I_inj = I_inj  # courant d'entrée:  peut être une somme de courants
    neurone.w_inj = w_inj  # poids synaptique courant d'entrée 
    neurone.V = V     # output neurone- on rappelle V = Vs et la dérivée peut se noter y au lieu de Vs point-
    neurone.sigmaS = sigmaS
    neurone.sigmaF = sigmaF
    neurone.Af = Af
    neurone.q = q      # signal slow current
    neurone.toM = 0.2
    neurone.toS = 2
    return neurone

########################################################
###          fonctions basiques nécessaire           ###
########################################################

# signal de forçage, appelé I_inj mais une ou deux fois F dans les articles: ne pas confondre.
def Input(t):
    amplitude = 1
    freq = 2
    phase = 0
    # sin
    I = amplitude * sin(6.28 * freq * t + phase)
    return I

# fonction crée pour raccourcir la notation, aussi appelée F dans les articles
def F(n):
    return n.V - n.Af * np.tanh((n.sigmaF / n.Af) * n.V)

def f_V(n, t):
    return -(F(n) + n.q - n.w_inj * n.I_inj) / n.toM

def f_Q(n, t):
    return (-n.q + n.sigmaS * n.V) / n.toS

def f_sigmaS(n, t):
    dot_sigma = 400
    y = f_V(n, t)  # y = dV/dt
    racine = sqrt(np.float64(y**2 + n.V**2))
    print(racine, y)
    
    if racine == 0:
        quotient = 0
    elif n.sigmaF < 1 + n.sigmaS:
        quotient = y / racine   
        dot_sigma = 2 * eps * n.I_inj * sqrt(n.toM * n.toS) * sqrt(1 + n.sigmaS - n.sigmaF) * quotient            
    else:
        dot_sigma = np.clip(dot_sigma, 300, 500)
          
    return dot_sigma

def update_neuron(n, t):
    n.sigmaS = n.sigmaS + dt * f_sigmaS(n, t)
    n.V = n.V + dt * f_V(n, t)
    n.q = n.q + dt * f_Q(n, t)
    return n

def couple_neurons(n1, n2, t, coef1, coef2):
    # Update n2 with input from n1 and external input
    n2.I_inj = coef1 * n1.V + coef2 * Input(t)
    n2 = update_neuron(n2, t)
    
    # Update n1 with input from n2 and external input at t + 1
    n1.I_inj = coef1 * n2.V + coef2 * Input(t + dt)
    n1 = update_neuron(n1, t + dt)
    
    return n1, n2

def simulate(t_max, coef1, coef2):
    n1 = create_NRS("n1", 0, 0.5, 0, 0.5, 0.3, 0.5, 0.1)
    n2 = create_NRS("n2", 0, 0.5, 0, 0.5, 0.3, 0.5, 0.1)
    
    Vs1, Vs2 = [], []
    Ts = np.arange(0, t_max, dt)
    
    for t in Ts:
        n1, n2 = couple_neurons(n1, n2, t, coef1, coef2)
        Vs1.append(n1.V)
        Vs2.append(n2.V)
    plot(Vs1, Vs2, Ts)   
    return Vs1, Vs2, Ts

def plot_neurons(Vs1, Vs2, Ts):
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    plt.suptitle("Coupled Neurons Simulation", fontsize=14)
    
    # First subplot
    axs[0].plot(Ts, Vs1, '-m', label='Neuron 1 Output')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Potential')
    axs[0].legend(loc='upper right')
    
    # Second subplot
    axs[1].plot(Ts, Vs2, '-b', label='Neuron 2 Output')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Potential')
    axs[1].legend(loc='upper right')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Display the plot
    plt.show()

########################################################
###                      dessin                      ###
########################################################
def plot(Vs1, Ts1, I_inj1, w_inj, liste_sigma_s):
    fig, axs = plt.subplots(3, 1, figsize=(10, 16))
    plt.suptitle(
        "Theoritical study CPGs - 2 neurons - w_inj= " + str(w_inj) + " - eps = " + str(eps),
        fontsize=14,
        x=0.09,
        y=1,
        ha="left",
        )
    
    label_font_size = 9
    # First subplot
    axs[0].plot(Ts1, I_inj1, '-y', label='I_inj - Forcing signal')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('potential')
    axs[0].legend(loc='upper right')
    
    # Second subplot
    axs[1].plot(Ts1, Vs1, '-m', label='Vs - Output signal')
    axs[1].set_ylabel('potential')
    axs[1].legend(loc='upper right')
    
    # Third subplot
    axs[2].plot(Ts1, liste_sigma_s, 'b-')
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('sigma S')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Display the plot
    plt.show()

#------------------Hand tracking----------------------------------------------------------------------#

def get_hand_landmarks_style(): 
    """Returns the default hand landmarks drawing style.
    Returns:
    A mapping from each hand landmark to its default drawing spec.
    """
    hand_landmark_style = {}
    for k, v in _HAND_LANDMARK_STYLE.items():
        for landmark in k:
            hand_landmark_style[landmark] = v
    return hand_landmark_style

def get_hand_connections_style(): 
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
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def hands_tracking():
    mp_drawing = solutions.drawing_utils
    
    #initialize neurons
    neur1 = create_NRS(nom='RS1', I_inj=0.01, w_inj=0.05, V=0.01, sigmaS=20, sigmaF=1, Af=0.2, q=0.1)
    neur2 = create_NRS(nom='RS2', I_inj=0.0, w_inj=0.3, V=0.0, sigmaS=15, sigmaF=3, Af=0.2, q=0.0)
    #n2 = create_NRS("n2", 0, 0.5, 0, 0.5, 0.3, 0.5, 0.1)
    
    #Create 5 empty lists
    L, list_V1, list_V2, list_T, list_I_inj, list_sigmaS1, list_sigmaS2 = [], [], [], [], [], [], []
    
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
                    # Follow position of the hand landmark[0] for point 0
                    wrist = hand_landmarks.landmark[0]
                    x_pixel = int(wrist.x * frame.shape[1])
                    y_pixel = int(wrist.y * frame.shape[0])
                    cv2.circle(frame, (x_pixel, y_pixel), 6, (200, 0, 200), -1) # Lilac color for the wrist point
                    
                    # Keep positions in a list
                    lmlist = (x_pixel, y_pixel)
                    
                    t = time.time() - start_time
                    I_inj = normalise(x_pixel)
                    neur1.I_inj = I_inj
                
                    neur1, neur2 = couple_neurons(neur1, neur2, t, coef1=0.05, coef2=0.01)
                    
                    list_V1.append(neur1.V)
                    list_V2.append(neur2.V)
                    list_T.append(t)
                    list_I_inj.append(I_inj)
                    list_sigmaS1.append(neur1.sigmaS)
                    list_sigmaS2.append(neur2.sigmaS)
                    L.append(lmlist)
        
            # Resize the frame to the desired window size
            resized_frame = rescale_frame(frame, percent=120)

            # Display the resized frame
            cv2.imshow('Hand Tracking', resized_frame)
        
            # Exit loop by pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release the video capture and close windows
    vidcap.release()
    cv2.destroyAllWindows()
         
    return L, list_V1, list_V2, list_T, list_I_inj, list_sigmaS1, list_sigmaS2

def normalise(x):
    max_val = 1   # mettre la valeur max du csv
    min_val = 0   # mettre la valeur min du csv
    x_norm = (x - min_val) / (max_val - min_val)
    return x_norm

def main():
    w_inj = 0.5
    L, list_V1, list_V2, list_T, list_I_inj, list_sigmaS1, list_sigmaS2 = hands_tracking()
    plot(list_V1, list_T, list_I_inj, w_inj, list_sigmaS1)  
    
    # Saving csv file of positions
    #df = DataFrame(L)
    #print(df)
    #df.to_csv(file_path + 'position1.csv', sep=';', index=True, header=None)
    
if __name__ == '__main__':
    main()
