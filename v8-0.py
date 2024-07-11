import cv2
from mediapipe import solutions
import numpy as np
from math import sqrt, degrees, atan2
import time
import matplotlib.pyplot as plt
import argparse
import qi
# Initialize mediapipe hands module
mphands = solutions.hands
mpdrawing = solutions.drawing_utils

#parameters for Hebbian------------------------------------------------------------------------------#
V0 = 0.01
q0 = 0.01
dt = 0.01
eps = 0.5

# Define constants for hand tracking
_RADIUS = 4
_GREEN = (48, 255, 48)
_CYAN = (192, 255, 48)
_THICKNESS_WRIST_MCP = 10
_THICKNESS_FINGER = 8
_THICKNESS_DOT = 8

_HAND_LANDMARK_STYLE = {
    (solutions.hands.HandLandmark.WRIST, solutions.hands.HandLandmark.THUMB_CMC,
     solutions.hands.HandLandmark.INDEX_FINGER_MCP, solutions.hands.HandLandmark.MIDDLE_FINGER_MCP,
     solutions.hands.HandLandmark.RING_FINGER_MCP, solutions.hands.HandLandmark.PINKY_MCP):
        mpdrawing.DrawingSpec(color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    (solutions.hands.HandLandmark.THUMB_MCP, solutions.hands.HandLandmark.THUMB_IP,
     solutions.hands.HandLandmark.THUMB_TIP):
        mpdrawing.DrawingSpec(color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    (solutions.hands.HandLandmark.INDEX_FINGER_PIP, solutions.hands.HandLandmark.INDEX_FINGER_DIP,
     solutions.hands.HandLandmark.INDEX_FINGER_TIP):
        mpdrawing.DrawingSpec(color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    (solutions.hands.HandLandmark.MIDDLE_FINGER_PIP, solutions.hands.HandLandmark.MIDDLE_FINGER_DIP,
     solutions.hands.HandLandmark.MIDDLE_FINGER_TIP):
        mpdrawing.DrawingSpec(color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    (solutions.hands.HandLandmark.RING_FINGER_PIP, solutions.hands.HandLandmark.RING_FINGER_DIP,
     solutions.hands.HandLandmark.RING_FINGER_TIP):
        mpdrawing.DrawingSpec(color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    (solutions.hands.HandLandmark.PINKY_PIP, solutions.hands.HandLandmark.PINKY_DIP,
     solutions.hands.HandLandmark.PINKY_TIP):
        mpdrawing.DrawingSpec(color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
}

_HAND_CONNECTION_STYLE = {
    solutions.hands_connections.HAND_PALM_CONNECTIONS:
        mpdrawing.DrawingSpec(color=_CYAN, thickness=_THICKNESS_WRIST_MCP),
    solutions.hands_connections.HAND_THUMB_CONNECTIONS:
        mpdrawing.DrawingSpec(color=_CYAN, thickness=_THICKNESS_FINGER),
    solutions.hands_connections.HAND_INDEX_FINGER_CONNECTIONS:
        mpdrawing.DrawingSpec(color=_CYAN, thickness=_THICKNESS_FINGER),
    solutions.hands_connections.HAND_MIDDLE_FINGER_CONNECTIONS:
        mpdrawing.DrawingSpec(color=_CYAN, thickness=_THICKNESS_FINGER),
    solutions.hands_connections.HAND_RING_FINGER_CONNECTIONS:
        mpdrawing.DrawingSpec(color=_CYAN, thickness=_THICKNESS_FINGER),
    solutions.hands_connections.HAND_PINKY_FINGER_CONNECTIONS:
        mpdrawing.DrawingSpec(color=_CYAN, thickness=_THICKNESS_FINGER),
}

def get_hand_landmarks_style():
    """Returns the default hand landmarks drawing style."""
    hand_landmark_style = {}
    for k, v in _HAND_LANDMARK_STYLE.items():
        for landmark in k:
            hand_landmark_style[landmark] = v
    return hand_landmark_style

def get_hand_connections_style():
    """Returns the default hand connections drawing style."""
    hand_connection_style = {}
    for k, v in _HAND_CONNECTION_STYLE.items():
        for connection in k:
            hand_connection_style[connection] = v
    return hand_connection_style

def rescale_frame(frame, percent=75):
    """Rescale the frame to a certain percentage of the original size."""
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


class NeuronRS:
    def __init__(self, nom, I_inj, w_inj, V, sigmaS, sigmaF, Af, q):
        self.nom = nom
        self.I_inj = I_inj
        self.w_inj = w_inj
        self.V = V
        self.sigmaS = sigmaS
        self.sigmaF = sigmaF
        self.Af = Af
        self.q = q
        self.toM = 0.35
        self.toS = 3.5

def F(n):
    return n.V - n.Af * np.tanh((n.sigmaF / n.Af) * n.V)

def f_V(n, t):
    return -(F(n) + n.q - n.w_inj * n.I_inj) / n.toM

def f_Q(n, t):
    return (-n.q + n.sigmaS * n.V) / n.toS

def f_sigmaS(n, t):
    dot_sigma = 400
    y = f_V(n, t)  # y=dV/dt
    racine = sqrt(y**2 + n.V**2)
    
    if racine == 0:
        quotient = 0
    elif n.sigmaF < 1 + n.sigmaS:
        quotient = y / racine
        dot_sigma = 2 * 0.5 * n.I_inj * sqrt(n.toM * n.toS) * sqrt(1 + n.sigmaS - n.sigmaF) * quotient
    else:
        dot_sigma = np.clip(dot_sigma, 300, 500)
        
    return dot_sigma

def update_neuron(n, t):
    n.sigmaS = n.sigmaS + dt * f_sigmaS(n, eps, t)
    n.V = n.V + dt * f_V(n, t)
    n.q = n.q + dt * f_Q(n, t)
    return n.V, n.sigmaS, n.q


########################################################
###                      dessin                      ###
########################################################  

def plot(Vs1, Ts1, I_inj1, sigmaS):
    fig, axs = plt.subplots(2, 1, figsize=(10, 18))
    ##### titre haut
    plt.suptitle(
        "Etude pratique CPGs - 1neur - input webcam - I_inj= " + str(I_inj1) + "  - eps = " + str(eps),
        fontsize=14,
        #fontweight="bold",
        x=0.09,
        y=1,
        ha="left",
    )

    # First subplot
    axs[0].plot(Ts1, I_inj1, 'y--', label='I_inj - signal de forcage')
    axs[0].plot(Ts1, Vs1, '-m', label='Vs - signal sortie')
    axs[0].set_xlabel('temps')
    axs[0].set_ylabel('potentiel')
    axs[0].legend(loc='upper right')
    #axs[1].set_title('I_inj - signal de forçage')
    

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

def normalise(x):
    min = 0
    max = 550
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
        angle = calculate_angle(wrist_landmark_time1, wrist_landmark_time2)
        names = [side+'ElbowYaw']
        maxSpeedFraction = 0.2
        motion_service.changeAngles(names, angle, maxSpeedFraction)
        #motion_service.rest()
    except Exception as e:
        print(f"Error controlling the robot hand: {e}")
    

def hands_tracking(session):
    video_service = session.service("ALVideoDevice")
    video_client = video_service.subscribeCamera("python_client", 0, 2, 11, 30)
    
    mp_drawing = solutions.drawing_utils
    neur = NeuronRS(nom='RS1', I_inj=0.0, w_inj=0.5, V=0.001, sigmaS=30, sigmaF=2, Af=0.2, q=0.01)
    L, list_V, list_T, list_I_inj, list_sigmaS = [], [], [], [], []
    
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
                        '''
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            solutions.hands.HAND_CONNECTIONS,
                            hand_landmark_style = get_hand_landmarks_style(),
                            hand_connection_style = get_hand_connections_style())
                        '''
                        # Get the current wrist landmark
                        wrist_landmark_time2 = hand_landmarks.landmark[solutions.hands.HandLandmark.WRIST]
                        x_pixel = int(wrist_landmark_time2.x * image.shape[1])
                        y_pixel = int(wrist_landmark_time2.y * image.shape[0])
                        
                        side = handedness.classification[0].label  # 'Left' or 'Right'
                        side = 'L' if side == 'Left' else 'R'
                        
                        # Keep positions in a list
                        lmlist=(x_pixel,y_pixel)
                    
                        t = time.time() - start_time
                        I_inj = normalise(x_pixel)
                        neur.I_inj = I_inj
                        V, sigmaS, q = update_neuron(neur, t)
                        list_V.append(V)
                        list_T.append(t)
                        list_I_inj.append(I_inj)
                        list_sigmaS.append(sigmaS)
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
    return L, list_V, list_T, list_I_inj, list_sigmaS    
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.1.147", help="Robot IP address")
    parser.add_argument("--port", type=int, default=9559, help="Robot port number")
    args = parser.parse_args()

    session = qi.Session()
    try:
        session.connect(f"tcp://{args.ip}:{args.port}")
        L, list_V, list_T, list_I_inj, list_sigmaS = hands_tracking(session)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Connection error: {e}")
    
    plot(list_V, list_T, list_I_inj, list_sigmaS)

if __name__ == "__main__":
    main()
