

#poignet au lieu de barycentre pour économiser. 
#pour économiser plus, pas de possibilité ni de néccéssité! de changer le code de media pipe.

#pour projeter, ne prendre que le x des valeurss. actuellement fait pour webcam fixe, 
#pour application robots, insérer maniip de changement de repère entre la récup de valaurs, 
#et la récup du x. (on veut le x changé), et ensuite, feeder au osc hebb.

#cv2 utilise le framerate donné par la vidéo, mais on peut modifier ce que donne le robot avec le code de maelic.


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

# Initialize mediapipe hands module
mphands = solutions.hands
mpdrawing = solutions.drawing_utils

# Specify the path to your video file
##vidpath =
#file_path = '/home/nootnoot/Documents/pepper-internship/'
file_path = '/home/pepper/Documents/pepper-internship/'

# Initialize video capture
vidcap = cv2.VideoCapture(0) #(vidpath)

# Set the desired window width and height
winwidth = 800
winheight = 600

#--------------------------------------------------------------------------------------------------
_RADIUS = 5
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
_THICKNESS_FINGER = 20
_THICKNESS_DOT = 10 #-1

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
        DrawingSpec(color=_GRAY, thickness=_THICKNESS_WRIST_MCP),
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

def findPosition(processFrames,lm,frame,handNo=0,draw=False):
    
    if processFrames.multi_hand_landmarks:
        myHand=processFrames.multi_hand_landmarks[handNo]
            
        for id,lm in enumerate(myHand.landmark):
            h,w,c = frame.shape
            cx,cy = int(lm.x*w),int(lm.y*h)
            if id==0:
                lmlist=(cx,cy)
            if draw:
                cv2.circle(frame,(cx,cy),15,(255,0,255),cv2.FILLED)        
    return lmlist
 
def followPosition(processFrames,lm,frame,handNo=0):
    if processFrames.multi_hand_landmarks:
        myHand=processFrames.multi_hand_landmarks[handNo]  
        for id,lm in enumerate(myHand.landmark):
            h,w,c = frame.shape
            cx,cy = int(lm.x*w),int(lm.y*h)
            if id==0:
                position=(cx,cy)       
    return position   

#-----------------------------------------------------------------------------------------------
mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles
df=DataFrame()
L=[]
# Initialize hand tracking
with mphands.Hands(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
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
            for lm in processFrames.multi_hand_landmarks:
                #mpdrawing.draw_landmarks(frame, lm, mphands.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    frame,
                    lm,
                    mphands.HAND_CONNECTIONS,
                    get_hand_landmarks_style(),
                    get_hand_connections_style()
                    )
                #Follow position of the hand
                position = followPosition(processFrames,lm,frame,handNo=0)
                # Keep positions in a list
                lmlist=findPosition(processFrames,lm, frame,handNo=0,draw=False)
                L.append(lmlist)
            print(position)
        
        
        # Resize the frame to the desired window size
        resized_frame = cv2.resize(frame, (winwidth, winheight))

        # Display the resized frame
        cv2.imshow('Hand Tracking', resized_frame)
        
        # Exit loop by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    df=DataFrame(L)
    print(df)
    df.to_csv(file_path+'position.csv', sep=';', index = True, header=None)
    # Release the video capture and close windows
    vidcap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
