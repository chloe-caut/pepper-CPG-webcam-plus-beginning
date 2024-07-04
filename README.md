# pepper-internship
spring summer 2024

short term goal: have pepper mirror a person's hand movement. 

internship goal: evaluate the interaction based on sychronisation & who is leading the movement etc etc, see job desc for more.


versions 0 to 3: hand tracking, either with robot camera (mouth) or with laptop webcam.
v4 skipped

v5: on laptop, feeding hand movement to a CPG model with hebbian learning, real time, graph drawn at the end. 
next versions: using robot camera, feeding CPG output to the robot to guide pepper's hand movements. (ros needed)

see v5-NEW for version with coupled neurons

v8 etc for robot




#############

job desc: 

#############



Le sujet de ce stage concerne le renforcement de la coordination motrice d’enfants atteints du
trouble du spectre autistiques (ASD). En effet, ces enfants (et aussi les adultes qu’ils sont
devenus) ont aussi bien souvent des troubles de la coordination motrice qui impactent leurs
interactions sociales. On sait en effet que les postures et les mouvements rythmiques des
membres, supérieurs notamment, sont une des modalités importantes de l’interaction sociale
entre les humains.


Ce sujet de stage propose de développer des algorithmes de renforcement de la coordination
motrice pour les personnes ASD, en utilisant un robot humanoïde Pepper dont les
mouvements et postures seront commandés par des “central pattern génerators” ou CPG qui
sont de réseaux de neurones inspirés des structures neuronales situées dans la moelle
épinière des mammifères et qui sont en charge du contrôle des mouvements rythmiques.


L’objectif de ce travail est que le robot, en regardant la personne, aura la propriété de se
synchroniser sur ses mouvements, autant que la personne se synchronisera sur les
mouvements du robot. La capacité de coordination motrice du robot étant parfaitement
contrôlable et modulable selon le réglage de certains paramètres des CPG, on peut imaginer
une thérapie de rééducation de la coordination motrice de la personne. 


En effet, dans une telle interaction, le robot peut être réglé selon plusieurs rôles : un rôle de meneur dans lequel
la personne doit suivre les gestes du robot sans que celui-ci puisse faire converger son rythme
vers celui de de la personne, un rôle d'apprenant dans lequel il se synchronise sur les gestes
de la personne, aidant cette dernière à coordonner ses mouvements (synchronisation
mutuelle) et enfin, un rôle mixte dans lequel le robot prend alternativement un rôle puis l'autre.
Cette dernière condition perturbe ainsi la synchronisation mutuelle, ce qui permet de renforcer
les compétences motrices de la personne.


Un autre point important de ce travail concerne l’étude des mécanismes de l’attention de la
personne sur la tâche à réaliser. Des indicateurs comme le temps d’exercices, la fréquence
de travail, la qualité de l’exercice etc... doivent être mesurés pour quantifier l'engagement de
la personne lors de cette interaction. 
Il serait intéressant de regarder si l’engagement est plus important lorsque l’homme se synchronise sur le robot ou l’inverse. Une mesure concernant
la pertinence de l’usage du robot par rapport à un travail avec un thérapeute sera également
important pour identifier des avantages et des manques d’une thérapie de rééducation de la
coordination motrice avec un robot.


Ce travail s’appuie sur des travaux déjà réalisés sur ce sujet (du code existe déjà et les modèles de CPG sont bien maîtrisés)


Pour mener à bien ce travail, le stagiaire devra suivre les étapes suivantes (pas forcément de manière chronologique):


1-  Etude bibliographique sur les CPG et les mécanismes de l’attention dans l'interaction humain/robot
   
2-  Prise en main du modèle de CPG de Rowat et Selverstion [1] , simulation (code existant)

3-  Prise en main du robot Pepper : codage chorégraphe + python

4- Implémentation d’une commande motrice simple de type sinusoïdale, expérimentations

5- Implémentation de la détection de mouvements : caméra externe au robot, Openpose, MoveNet (TensorFlow), expérimentations

6- Implémentation de commande par CPG pour une articulation d’un bras,mise en oeuvre de la synchronisation du robot, expérimentations

7- Implémentation de commande par CPG pour deux bras, puis le torse, expérimentations

8- Analyse des mécanismes de synchronisation/désynchronisation.

9- Étude de l’attention : déterminer des indicateurs pour la mesure. Déterminer l’engagement de personne en fonction du rôle dans l’interaction. Comparer cet engagement par rapport à un exercice avec des thérapeutes.


Bibliographie
[1] Melanie Jouaiti, Patrick Henaff. Comparative Study of Forced Oscillators for the Adaptive Generation of Rhythmic Movements in Robot Controllers. Biological Cybernetics (Modeling), 2019, 113 (5-6), pp.547-560.
[2] Melanie Jouaiti, Patrick Henaff. Robot-Based Motor Rehabilitation in Autism: A Systematic Review. International Journal of Social Robotics, 2019, 11 (5), pp.753-764
[3] Baptiste Menges, Adrien Guenard, Patrick Henaff. Dynamic Oscillators to Compensate Master Devices Imperfections in Robots Teleoperation Tasks Requiring Dynamic Movements. CASE 2021 – IEEE 17th International Conference on Automation Science and Engineering, Aug 2021, Lyon, France.
[4] C. Jost, B. Le Pévédic, D. Duhaut : Étude de l’impact du couplage geste et parole sur un robot, ouvrage collectif - Interactions et Intercompréhension : une approche comparative Homme-Homme, Animal-Homme-Machine et Homme-Machine chez EME editions, collection échange, ISBN 978-2-8066-0859-8, mars 2013.
[5] Nicolas Spatola, l’interaction Homme-Robot, de l’anthropomorphisme à l’humanisation : l’Année psychologique 2019/4 (Vol. 119), pages 515 à 563
[6] Syed Khursheed Hasnain : Synchronisation et coordination interpersonnelle dans l'interaction Homme-robot ; Thèse de doctorat en STIC (sciences et technologies de l'information et de la communication) - Université de Cergy Pontoise, juillet 2014
