

import qi
import argparse
import sys
import cv2
from mediapipe import solutions
import numpy as np
from math import sqrt, degrees, atan2, pi, sin
import time
import matplotlib.pyplot as plt


#  python3 v10-3.py


class HandTrackingApp:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.mphands = solutions.hands
        self.mpdrawing = solutions.drawing_utils
        self.V0 = 0.01
        self.q0 = 0.01
        self.dt = 0.01
        self.eps = 0.5
        self._initialize_hand_styles()
        self.session = None
        self.video_client = None
        self.motion_service = None

    def _initialize_hand_styles(self):
        _RADIUS = 4
        _GREEN = (48, 255, 48)
        _CYAN = (192, 255, 48)
        _THICKNESS_WRIST_MCP = 10
        _THICKNESS_FINGER = 8
        _THICKNESS_DOT = 8

        self._HAND_LANDMARK_STYLE = {
            (self.mphands.HandLandmark.WRIST, self.mphands.HandLandmark.THUMB_CMC,
             self.mphands.HandLandmark.INDEX_FINGER_MCP, self.mphands.HandLandmark.MIDDLE_FINGER_MCP,
             self.mphands.HandLandmark.RING_FINGER_MCP, self.mphands.HandLandmark.PINKY_MCP):
                self.mpdrawing.DrawingSpec(color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
            (self.mphands.HandLandmark.THUMB_MCP, self.mphands.HandLandmark.THUMB_IP,
             self.mphands.HandLandmark.THUMB_TIP):
                self.mpdrawing.DrawingSpec(color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
            (self.mphands.HandLandmark.INDEX_FINGER_PIP, self.mphands.HandLandmark.INDEX_FINGER_DIP,
             self.mphands.HandLandmark.INDEX_FINGER_TIP):
                self.mpdrawing.DrawingSpec(color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
            (self.mphands.HandLandmark.MIDDLE_FINGER_PIP, self.mphands.HandLandmark.MIDDLE_FINGER_DIP,
             self.mphands.HandLandmark.MIDDLE_FINGER_TIP):
                self.mpdrawing.DrawingSpec(color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
            (self.mphands.HandLandmark.RING_FINGER_PIP, self.mphands.HandLandmark.RING_FINGER_DIP,
             self.mphands.HandLandmark.RING_FINGER_TIP):
                self.mpdrawing.DrawingSpec(color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
            (self.mphands.HandLandmark.PINKY_PIP, self.mphands.HandLandmark.PINKY_DIP,
             self.mphands.HandLandmark.PINKY_TIP):
                self.mpdrawing.DrawingSpec(color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
        }

        self._HAND_CONNECTION_STYLE = {
            solutions.hands_connections.HAND_PALM_CONNECTIONS:
                self.mpdrawing.DrawingSpec(color=_CYAN, thickness=_THICKNESS_WRIST_MCP),
            solutions.hands_connections.HAND_THUMB_CONNECTIONS:
                self.mpdrawing.DrawingSpec(color=_CYAN, thickness=_THICKNESS_FINGER),
            solutions.hands_connections.HAND_INDEX_FINGER_CONNECTIONS:
                self.mpdrawing.DrawingSpec(color=_CYAN, thickness=_THICKNESS_FINGER),
            solutions.hands_connections.HAND_MIDDLE_FINGER_CONNECTIONS:
                self.mpdrawing.DrawingSpec(color=_CYAN, thickness=_THICKNESS_FINGER),
            solutions.hands_connections.HAND_RING_FINGER_CONNECTIONS:
                self.mpdrawing.DrawingSpec(color=_CYAN, thickness=_THICKNESS_FINGER),
            solutions.hands_connections.HAND_PINKY_FINGER_CONNECTIONS:
                self.mpdrawing.DrawingSpec(color=_CYAN, thickness=_THICKNESS_FINGER),
        }

    def get_hand_landmarks_style(self):
        """Returns custom hand landmarks drawing style."""
        hand_landmark_style = {}
        for k, v in self._HAND_LANDMARK_STYLE.items():
            for landmark in k:
                hand_landmark_style[landmark] = v
        return hand_landmark_style

    def get_hand_connections_style(self):
        """Returns custom hand connections drawing style."""
        hand_connection_style = {}
        for k, v in self._HAND_CONNECTION_STYLE.items():
            for connection in k:
                hand_connection_style[connection] = v
        return hand_connection_style

    def rescale_frame(self, frame, percent=75):
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
    
    # signal de forçage, appelé I_inj
    def Input(self, t):
        amplitude = 1
        freq = 2
        phase = 0
        I = amplitude * sin(2*pi * freq * t + phase)
        return I
    
    def F(self, n):
        return n.V - n.Af * np.tanh((n.sigmaF / n.Af) * n.V)

    def f_V(self, n, t):
        return -(self.F(n) + n.q - n.w_inj * n.I_inj) / n.toM

    def f_Q(self, n, t):
        return (-n.q + n.sigmaS * n.V) / n.toS

    def f_sigmaS(self, n, t):
        dot_sigma = 400
        eps = 0.5
        y = self.f_V(n, t)  # y=dV/dt
        racine = sqrt(y**2 + n.V**2)
        
        if racine == 0:
            quotient = 0
        elif n.sigmaF < 1 + n.sigmaS:
            quotient = y / racine
            dot_sigma = 2 * eps * n.I_inj * sqrt(n.toM * n.toS) * sqrt(1 + n.sigmaS - n.sigmaF) * quotient
        else:
            dot_sigma = np.clip(dot_sigma, 300, 500)
            
        return dot_sigma
    
    def couple_neurons(self, n1, n2, t, coef1, coef2):
        # Update n2 with input from n1 and external input
        n2.I_inj = coef1 * n1.V + coef2 * self.Input(t)
        n2 = self.update_neuron(n2, t)
    
        # Update n1 with input from n2 and external input at t + dt
        n1.I_inj = coef1 * n2.V + coef2 * self.Input(t + self.dt)
        n1 = self.update_neuron(n1, t + self.dt)
    
        return n1, n2
    
    def update_neuron(self, n, t):
        n.sigmaS = n.sigmaS + self.dt * self.f_sigmaS(n, t)
        n.V = n.V + self.dt * self.f_V(n, t)
        n.q = n.q + self.dt * self.f_Q(n, t)
        return n.V, n.sigmaS, n.q

    ########################################################
    ###                      dessin                      ###
    ########################################################  

    def plot(self, Vs1, Ts1, I_inj1, sigmaS):
        fig, axs = plt.subplots(2, 1, figsize=(10, 18))
        ##### titre haut
        plt.suptitle(
            "Etude pratique CPGs - 1neur - input webcam - I_inj= " + str(I_inj1) + "  - eps = " + str(self.eps),
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
        
        # Third subplot
        axs[1].plot(Ts1, sigmaS, 'b-')      #label='sigma S')
        axs[1].set_xlabel('temps')
        axs[1].set_ylabel('sigma S')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Display the plot
        plt.show()

    def normalise(self, x):
        min = 0
        max = 550
        x_norm = (x-min)/(max - min)
        return x_norm
        
    def calculate_angle(self, wrist_landmark_time1, wrist_landmark_time2):
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

    def control_robot_hand(self, motion_service, side, wrist_landmark_time1, wrist_landmark_time2):
        try:
            # Calculate the angle for the movement
            angle = self.calculate_angle(wrist_landmark_time1, wrist_landmark_time2)
            
            # Create the name for the joint to be moved
            names = [side + 'ElbowYaw']
            maxSpeedFraction = 0.2
            
            # Perform the movement
            motion_service.changeAngles(names, angle, maxSpeedFraction)
            
        except Exception as e:
            print(f"Error controlling the robot hand: {e}")    

    def hands_tracking(self, session, motion_service):
        video_service = session.service("ALVideoDevice")
        self.video_client = video_service.subscribeCamera("python_client", 0, 2, 11, 30)
        
        mp_drawing = solutions.drawing_utils
        #initialize neurons
        neur1 = self.NeuronRS(nom='RS1', I_inj=0.01, w_inj=0.5, V=0.001, sigmaS=30, sigmaF=2, Af=0.2, q=0.01)
        neur2 = self.NeuronRS(nom='RS2', I_inj=0.0, w_inj=0.3, V=0.0, sigmaS=5, sigmaF=3, Af=0.2, q=0.0)
        
        L, list_V1, list_V2, list_T, list_I_inj, list_sigmaS1, list_sigmaS2 = [], [], [], [], [], [], []
        
        try:
            with self.mphands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                start_time = time.time()
                wrist_landmark_time1 = None
                while True:
                    # Get image from Pepper's camera
                    frame = video_service.getImageRemote(self.video_client)
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
                            wrist_landmark_time2 = hand_landmarks.landmark[self.mphands.HandLandmark.WRIST]
                            x_pixel = int(wrist_landmark_time2.x * image.shape[1])
                            y_pixel = int(wrist_landmark_time2.y * image.shape[0])
                            
                            side = handedness.classification[0].label  # 'Left' or 'Right'
                            side = 'L' if side == 'Left' else 'R'
                            
                            # Keep positions in a list
                            lmlist=(x_pixel,y_pixel)
                        
                            t = time.time() - start_time
                            I_inj = self.normalise(x_pixel)
                            
                            neur1.I_inj = I_inj
                            neur1, neur2 = self.couple_neurons(neur1, neur2, t, coef1=0.05, coef2=0.02)
                            
                            list_V1.append(neur1.V)
                            list_V2.append(neur2.V)
                            list_T.append(t)
                            list_I_inj.append(I_inj)
                            list_sigmaS1.append(neur1.sigmaS)
                            list_sigmaS2.append(neur2.sigmaS)
                            L.append(lmlist)
                            
                            if wrist_landmark_time1 is not None:
                                # Control the robot hand based on previous and current wrist positions
                                self.control_robot_hand(motion_service, side, wrist_landmark_time1, wrist_landmark_time2)  
            
                            # Update the previous wrist landmark
                            wrist_landmark_time1 = wrist_landmark_time2
                            
                    # Display the image
                    cv2.imshow('Hand Tracking', image)

                    if cv2.waitKey(1) & 0xFF == ord('q'):   # could use 27:
                        break
        except Exception as e:
            print(f"Error in hand tracking: {e}")
        finally:
            video_service.unsubscribe(self.video_client)
            cv2.destroyAllWindows()
        #motion_service.rest()      
        
        return L, list_V1, list_V2, list_T, list_I_inj, list_sigmaS1, list_sigmaS2 

    def main(self):
        # Create a new NAOqi session, that is used for connecting to robot.
        self.session = qi.Session()
        
        try:
            self.session.connect("tcp://" + self.ip + ":" + str(self.port))
        except RuntimeError:
            print ("Can't connect to Naoqi at ip \"" + self.ip + "\" on port " + str(self.port) +".\n"
                "Please check your script arguments. Run with -h option for help.")
            sys.exit(1)
       
        # Initialize motion and posture services
        motion_service = self.session.service("ALMotion")
        posture_service = self.session.service("ALRobotPosture")
        motion_service.wakeUp()
        posture_service.goToPosture("StandInit", 0.5)
        
        L, list_V1, list_V2, list_T, list_I_inj, list_sigmaS1, list_sigmaS2 = self.hands_tracking(self.session, self.motion_service)
        self.plot(list_V1, list_T, list_I_inj, list_sigmaS1)

if __name__ == "__main__":
    # Create an object parser that will manage the arguments of command line (here the connectivity parameters to Pepper)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.1.147", help="Robot IP address")
    parser.add_argument("--port", type=int, default=9559, help="Robot port number")
    
    # Analyse arguments of the command line and stock them in objet args
    args = parser.parse_args()
    
    #
    
    # Initialize and run the hand tracking app
    app = HandTrackingApp(args.ip, args.port)
    app.main()