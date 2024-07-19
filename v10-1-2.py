import qi
import argparse
import sys
import cv2
from mediapipe import solutions
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.python.solutions.pose import PoseLandmark
from typing import Mapping, Tuple
from pandas import DataFrame
from math import sqrt, atan2, sin, pi
import time
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt


#  python3 v10-1-2.py

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
class NeuronRS:

########################################################
###             constructeur du neurone              ###
########################################################
    def __init__(self, nom, I_inj, w_inj, V, sigmaS, sigmaF, Af, q):
        self.nom = nom
        self.I_inj = I_inj  # courant d'entrée:  peut être une somme de courants
        self.w_inj = w_inj  # poids synaptique courant d'entrée 
        self.V = V          # output neurone- on rappelle V = Vs et la dérivée peut se noter y au lieu de Vs point-
        self.sigmaS = sigmaS
        self.sigmaF = sigmaF
        self.Af = Af
        self.q = q          # signal slow current
        self.toM = 0.2
        self.toS = 2


########################################################
###          fonctions basiques nécessaire           ###
########################################################

# signal de forçage, appelé I_inj mais une ou deux fois F dans les articles: ne pas confondre.
def Input(t):
    amplitude = 1
    freq = 25 #2
    phase = 0
    # sin
    I = amplitude * sin(2*pi * freq * t + phase)
    return I

# fonction F de l'Oscillateur de Rowat et Selverston
def F(n):
    return n.V - n.Af * np.tanh((n.sigmaF / n.Af) * n.V)

# fonction de dV/dt
def f_V(n, t):
    return -(F(n) + n.q - n.w_inj * n.I_inj) / n.toM

# fonction de dq/dt              
def f_Q(n, t):
    return (-n.q + n.sigmaS * n.V) / n.toS
     
def f_sigmaS(n, t):
    dot_sigma = 400
    y = f_V(n, t)  # y = dV/dt
    racine = sqrt(np.float64(y**2 + n.V**2))
    #print(racine, y)
    
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
    n1.I_inj = coef1 * n2.V + coef2 * Input(t + 1)
    n1 = update_neuron(n1, t + 1)
    
    return n1, n2

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

def plot_xy(L,list_T):
    # Unpack the x and y coordinates from L
    x_coords = [point[0] for point in L]
    y_coords = [point[1] for point in L]
    
     # Plot the x and y coordinates over time
    plt.plot(list_T, x_coords, label='x_pixel', color='blue')
    plt.plot(list_T, y_coords, label='y_pixel', color='orange')
    
    # Add titles and labels
    plt.title('Position over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (mm)')
    
    # Add a legend
    plt.legend()
    
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

def normalise(x):
    min = 0    
    max = 500  
    x_norm = (x-min)/(max - min)
    return x_norm

def calculate_angle(wrist_landmark_time1, wrist_landmark_time2):
        min_angle_rad = 0.27  # Minimum angle in radians
        max_angle_rad = 2.04  # Maximum angle in radians
        
        WL_t2y = wrist_landmark_time2.y
        WL_t1y = wrist_landmark_time1.y
        WL_t2x = wrist_landmark_time2.x
        WL_t1x = wrist_landmark_time1.x
        
        # Calculate the angle between two wrist positions over time
        angle_rad = atan2(WL_t2y - WL_t1y, WL_t2x - WL_t1x)
            
        # Clamp the angle between the min and max values 
        angle_rad = max(min_angle_rad, min(max_angle_rad, angle_rad)) 
        
        return angle_rad

def control_robot_hand(session, side, wrist_landmark_time1, wrist_landmark_time2):
    motion_service = session.service("ALMotion")
    posture_service = session.service("ALRobotPosture")
    motion_service.wakeUp()
    posture_service.goToPosture("StandInit", 0.5)
        
    try:
        # Calculate the angle for the movement
        angle = calculate_angle(wrist_landmark_time1, wrist_landmark_time2)
            
        # Create the name for the joint to be moved
        names = [side + 'ElbowYaw']
        maxSpeedFraction = 0.2
            
        # Perform the movement
        motion_service.changeAngles(names, angle, maxSpeedFraction)
            
    except Exception as e:
        print(f"Error controlling the robot hand: {e}")    


def hands_tracking(session):
    video_service = session.service("ALVideoDevice")
    video_client = video_service.subscribeCamera("python_client", 0, 2, 11, 30)
    mp_drawing = solutions.drawing_utils
    
    #initialize neurons
    neur1 = NeuronRS(nom='RS1', I_inj=0.01, w_inj=0.5, V=0.001, sigmaS=30, sigmaF=2, Af=0.2, q=0.01)
    neur2 = NeuronRS(nom='RS2', I_inj=0.0, w_inj=0.3, V=0.0, sigmaS=5, sigmaF=3, Af=0.2, q=0.0)
        
    #Create 5 empty lists
    L, list_V1, list_V2, list_T, list_I_inj, list_sigmaS1, list_sigmaS2 = [], [], [], [], [], [], []
    
    try:
        with mphands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            start_time = time.time()
            wrist_landmark_time1 = None
            while True:
                # Get image from Pepper's camera
                frame = video_service.getImageRemote(video_client)
                width = frame[0]
                height = frame[1]
                array = np.frombuffer(frame[6], dtype=np.uint8).reshape((height, width, 3))
                image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                    
                # Process the image and detect hands
                image.flags.writeable = False
                results = hands.process(image)

                # Draw the hand annotations on the image
                image.flags.writeable = True
                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        # Get the current wrist landmark
                        wrist_landmark_time2 = hand_landmarks.landmark[mphands.HandLandmark.WRIST]
                        x_pixel = int(wrist_landmark_time2.x * image.shape[1])
                        y_pixel = int(wrist_landmark_time2.y * image.shape[0])
                            
                        side = handedness.classification[0].label  # 'Left' or 'Right'
                        side = 'L' if side == 'Left' else 'R'
                            
                        # Keep positions in a list
                        lmlist=(x_pixel,y_pixel)
                        
                        t = time.time() - start_time
                        I_inj = normalise(x_pixel)
                            
                        neur1.I_inj = I_inj
                        neur1, neur2 = couple_neurons(neur1, neur2, t, coef1=0.05, coef2=0.02)
                            
                        list_V1.append(neur1.V)
                        list_V2.append(neur2.V)
                        list_T.append(t)
                        list_I_inj.append(I_inj)
                        list_sigmaS1.append(neur1.sigmaS)
                        list_sigmaS2.append(neur2.sigmaS)
                        L.append(lmlist)
                            
                        if wrist_landmark_time1 is not None:
                            # Control the robot hand based on previous and current wrist positions
                            control_robot_hand(session, side, wrist_landmark_time1, wrist_landmark_time2)  
            
                        # Update the previous wrist landmark
                        wrist_landmark_time1 = wrist_landmark_time2
                            
                # Display the image
                cv2.imshow('Hand Tracking', image)

                if cv2.waitKey(1) & 0xFF == ord('q'):   # could use 27:
                    break
    except Exception as e:
        print(f"Error in hand tracking: {e}")
    finally:
        video_service.unsubscribe(video_client)
        cv2.destroyAllWindows()
    
    #motion_service.rest()      
    return L, list_V1, list_V2, list_T, list_I_inj, list_sigmaS1, list_sigmaS2 


def main():
    # Create a new NAOqi session, that is used for connecting to robot.
    session = qi.Session()
    ip = "192.168.0.106"  # 0 106  pour pepper 4 en haut # 1 147 pour pepper 1 en bas
    port = 9559
    try:
        session.connect("tcp://" + ip + ":" + str(port))
    except RuntimeError:
            print ("Can't connect to Naoqi at ip \"" + ip + "\" on port " + str(port) +".\n"
                "Please check your script arguments. Run with -h option for help.")
            sys.exit(1)
        
    L, list_V1, list_V2, list_T, list_I_inj, list_sigmaS1, list_sigmaS2 = hands_tracking(session)
    plot(list_V1, list_T, list_I_inj, list_sigmaS1)
    
if __name__ == "__main__":
    main()