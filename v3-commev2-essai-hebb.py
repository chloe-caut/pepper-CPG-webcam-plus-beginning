#python3 v3-commev2-essai-hebb.py



#poignet au lieu de barycentre pour économiser. 
#pour économiser plus, pas de possibilité ni de néccéssité! de changer le code de media pipe.

#pour projeter, ne prendre que le x des valeurss. actuellement fait pour webcam fixe, 
#pour application robots, insérer maniip de changement de repère entre la récup de valaurs, 
#et la récup du x. (on veut le x changé), et ensuite, feeder au osc hebb.

#cv2 utilise le framerate donné par la vidéo, mais on peut modifier ce que donne le robot avec le code de maelic.




import cv2
from mediapipe import solutions


#imports pour hebb
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal
#fin imports pour hebb

#reqs de v2
'''
pip install mediapipe
pip install protobuf==3.20.*
'''


#reqs à utiliser dans le futur, vienent de la version v0 compatible avec le robot- voir modifs
"""
    COMMANDE A LANCER SUR ROBOT:file_path = '/home/pepper/Documents/pepper-internship/'
    gst-launch-0.10 -v v4l2src device=/dev/video1 ! 'video/x-raw-yuv,width=640, height=480,framerate=30/1' ! ffmpegcolorspace ! jpegenc ! rtpjpegpay ! udpsink host=192.168.1.123 port=3001
    mettre l'ip du pc

    TO INSTALL:
    pip install opencv-python mediapipe
    sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
    sudo apt-get install libgirepository1.0-dev
    pip install PyGObject
"""



from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.python.solutions.pose import PoseLandmark
from typing import Mapping, Tuple
from pandas import DataFrame



#--------------------------------------------------------------------------------------------------

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

IinjListe=[]

###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################


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
                IinjListe.append(position[0]) #modif de remplisssage de liste!!!!!! Addon v2 to v3 ici !!!!!!!!!!!!!!!!!!!!!!!!! juste cette ligne
    return position   



###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################




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


###############################################################################################
###############################################################################################

def main():
    df=DataFrame(L)
    print(df)
    df.to_csv(file_path+'position.csv', sep=';', index = True, header=None)
   # Vs1,Ts1, I_inj1, liste_sigma_s = simul(neur1)  #addon pour hebb!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! euh jsp
    # Release the video capture and close windows
    vidcap.release()
    cv2.destroyAllWindows()

    return df[0] #added for passage v2 to v3


#dans df: x puis y. df[0] donne x. pour les tests AVANT changement de repère via ros, je modifie comme ça.
#A CHANGER ABSOLUMENT APRES OPTI POUR ROBOT CAD V4 ON EST EN V3 OPTI WEBCAM AVEC HEBB







###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
# add on pour mixer avec hebb ici
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################

#si on fait du real time, la valaur de IInj à utiliser est la dernière de la liste. je garde 
#tout en liste (comment gérer le temps? il fait que les coups d'horloge soient les frames non?) 
#ou je met en temporel l'horloge du pc, avec la bilbio time je crois, mais pour les pas d'apprentissage la 
#dans la fonction hebb uniquement, j'utilise les frames pour actualiser sigma s.



#toM = 0.35
#toS = 3.5
#Af = 1
#sigmaS = 2
#"sigmaF = 1.5"
#w_inj = 0.5 #poids synaptique I_inj
V0 = 0
q0 = 0
dt = 0.01
eps = 0.1  

class NeuroneRS : pass  #car pas vraie prog obj



########################################################
###             état du neurone instant t            ###
########################################################  


def create_NRS(nom='RS1', I_inj = 0.0,w_inj=0.5, V=0.0, sigmaS=2.0 ,sigmaF=1.5,Af = 1,q=0.0):
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

#signal de forçage, appelé I_inj mais une ou deux fois F dans les articles: ne pas confondre.
def Input(t) :
    #dirac
    '''
    I=0
    if t >=3 and t<=17:
        I = scipy.signal.square(t, duty= 0.5) '''
        
    #carré
    #I= scipy.signal.square(t, duty= 0.5) 

    #sin
     #-------------------------------------------------------------------------------ici i inj sinus a freq variable
    amplitude  = 1
    freq = 2
    phase=0
    I = amplitude * math.sin(6.28 * freq * t + phase )

    return I
     
#notation incrémentale à venir à cause de l'intégration    

def f_V(n, t):
    #attention ici j'utilise la fct input --------------------------------------------------------
    return  - (F(n)+ n.q - n.w_inj *n.I_inj)  /n.toM

def f_Q(n,t):
    return (-n.q + n.sigmaS*n.V) /n.toS



#créatrice de pbs: 
def f_sigmaS(n,t):
    #print("fv ",f_V(n,t))  #nan si Input sinus math, ok si input carré scipy
    #breakpoints pour voir d'ou vient le pb
    y= f_V(n,t) #premier appel la presque derivée 
    sousRacine =  y**2 + (n.V)**2 #les carrés
    racine = math.sqrt( sousRacine ) #la racine
    quotient = y/ racine #if racine else 0    #la division #################################################################################ici le if else 0
    sigmasuivant = 2 * eps * n.I_inj * math.sqrt(n.toM*n.toS) * math.sqrt(1+ n.sigmaS - n.sigmaF) *      quotient
    # 2 * eps * n.I_inj * math.sqrt(n.toM*n.toS) * math.sqrt(1+ n.sigmaS - n.sigmaF) *    f_V(n,t) /  math.sqrt( f_V(n,t)**2 + (n.V)**2  )
    return sigmasuivant


########################################################
###                   intégration                    ###
########################################################  


#faut il mettre à jour le sigmaS avant ou après calcul de Vs? 
#comparer les 2, un des cas peut diverger, 
#sigmaS calculé en fct de l'état courant donc à calculer d'abord car sigmaS de sortie est en t+1

#Methode d'euler pour résoudre les équations différentielles
#intégration / main / boucle de résolution - trouver nom



def simul(n):
    t = 0
    T = 20                                               # définition du temps de simulation
    list_V = []
    list_T = []
    list_I_inj = []
    list_sigmaS=[]
    newtime = len(IinjListe) * dt #nombre de frames en gros fois dt et la on est à nouveau sur des listes de la meme longueur je crois

    #----------------------------------------------------------------ici on +1 les neurones
   # while t < T :
    while t <  newtime  : #adapté pour traitement hebb une fois la vidoé arretée!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #n.I_inj = Input(t)
        i = int(t / dt)
        n.I_inj = float( IinjListe[i] )     #voila l'autre changement !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        n.sigmaS = n.sigmaS + dt* f_sigmaS(n,t)
        n.V      = n.V      + dt* f_V(n,t)
        n.q      = n.q      + dt* f_Q(n,t)
        #test
        '''
        print ("sortie  :   ",  n.V)
        print (" q  :   ",  n.q)
        print ("sigma :   ",  n.sigmaS)
        print ("i_inj :  ", Input(t))
        '''

        t += dt

        list_V.append(n.V)
        list_T.append(t)  
        #list_I_inj.append(Input(t))  # why did i do this in the first place??? more precise but not really
        #should be like this either way, no?
        list_I_inj.append(n.I_inj)    #qui est actuellement la sortie de la liste qui vient de l'autre programme qui a été frankensteiné à celui là
        list_sigmaS.append(n.sigmaS)
        

    return list_V, list_T, list_I_inj, list_sigmaS


neur1 = create_NRS(nom='RS1', I_inj = 0.0,w_inj=0.1, V=0.001, sigmaS=2.0 ,sigmaF=1.5)  
#neur2 = create_NRS(nom='RS2', I_inj = 0.0,w_inj=0.5, V=0.0, sigmaS=2.0 ,sigmaF=1.5)

#avant ici, maintenant dans main pour être fait à la fin pour commencer// et back
Vs1,Ts1, I_inj1, liste_sigma_s = simul(neur1)

#print(liste_sigma_s)

########################################################
###                      dessin                      ###
########################################################  

plt.plot(Ts1, Vs1,    '-m',        label ='Vs - signal sortie')
plt.plot(Ts1, I_inj1,'y--',    label ='I_inj - signal de forçage')
plt.plot(Ts1, liste_sigma_s, 'b-', label= 'sigma S')
#Vs2,Ts2 = Neurone_RS(neur2)
#plt.plot(Ts2, Vs2)

#### labels axes
plt.xlabel('temps')
plt.ylabel('intensité')


##### titre haut
plt.suptitle(
    "CPG ! ",
    fontsize=16,
    fontweight="bold",
    x=0.126,
    y=0.98,
    ha="left",
)

##### sous titre haut
plt.title(
    "un neurone, hebbian buggé, et I-inj sortie webcam hand tracking ",
    fontsize=14,
    pad=10,
    loc="left",
)

#### le carré avec les labels
plt.legend(loc='upper right')  


plt.show()






###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
#fin addon hebb

###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################







if __name__ == '__main__':
    main() 





