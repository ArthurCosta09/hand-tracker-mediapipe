import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import math


class HandTracking:


    def __init__(self, mode=False, maxHands=2, complexity=1,detectionCon=0.5, trackCon=0.5) -> None:
        self._mode = mode
        self._maxHands = maxHands
        self._complexity = complexity
        self._detectionCon = detectionCon
        self._trackCon = trackCon

        cv.namedWindow("Hand tracker")

        self._cam = cv.VideoCapture(0)
        self._cam.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self._cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self._cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        self._cam.set(cv.CAP_PROP_FPS, 5)

        self.mpHands = mp.solutions.hands

        """
        Parâmetros:
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1, (0 ou 1)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5

        Obs: A classe utiliza somente cores RGB. Portanto é necessário converter
        a imagem para RGB.
        """
        self.hands = self.mpHands.Hands(self._mode, self._maxHands, self._complexity, self._detectionCon, self._trackCon)

        # Serve para desenhar os 21 pontos da mão a ser detectada
        self.mp_draw = mp.solutions.drawing_utils

        self.prev_time = 0
        self.curr_time = 0

        # Coordenadas iniciais do retângulo (x, y)
        self.rect_coord1 = (10, 110)
        self.rect_coord2 = (70, 180)

        # Largura e altura do retângulo
        self.rect_width = 60
        self.rect_height = 70
       
        # Variável booleana para permitir a movimentação da caixa
        self.is_Moving = False


    def _findHands(self, frame) -> None:

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
 
        # Processar a imagem RGB
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hlm in results.multi_hand_landmarks:
                for id, lm in enumerate(hlm.landmark):
                    height, width, channel = frame.shape
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    
                    if id == 8: 
                        cv.circle(frame, (cx, cy), 15, (255, 0, 255), cv.FILLED)
                        self._moveBlock((cx, cy), self.rect_coord1, self.rect_coord2)
                                        
                self.mp_draw.draw_landmarks(frame, hlm, self.mpHands.HAND_CONNECTIONS)


    def _showFps(self, frame):
        self.curr_time = time.time()
        fps = int(1 / (self.curr_time - self.prev_time))

        self.prev_time = self.curr_time
        
        cv.putText(frame, f"fps: {fps}", (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        

    def _drawRect(self, frame, color: tuple, *coordinates: tuple):
        if len(coordinates) > 2:
            raise Exception('Insira somente duas coordenadas para desenhar o retângulo')

        coord1, coord2 = coordinates
        pts = np.array([[(coord1[0], coord1[1]), (coord2[0], coord1[1]), 
                        (coord2[0], coord2[1]), (coord1[0], coord2[1])]])

        return cv.fillPoly(frame, pts, color)


    def _drawButton(self, frame):


        def move_block(event, x, y, flags, params):
            if event == cv.EVENT_LBUTTONDOWN:
                pts = np.array([[(20, 300), (150, 300), (150, 350), (20, 350)]])

                is_inside = cv.pointPolygonTest(pts, (x, y), False)
                if is_inside > 0:
                    self.is_Moving = False if self.is_Moving else True
        

        pts = np.array([[(20, 300), (150, 300), (150, 350), (20, 350)]])
        color = (0, 255, 0) if self.is_Moving else (0, 0, 255)
        cv.setMouseCallback('Hand tracker', move_block)
        # cv.rectangle(frame, (20, 300), (150, 350), (0, 0, 255), cv.FILLED)
        cv.fillPoly(frame, pts, color)
        cv.putText(frame, "Move", (40, 340), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


    def _moveBlock(self, circle_center: tuple, *coordinates: tuple):
        if len(coordinates) > 2:
            raise Exception('Insira somente duas coordenadas para desenhar o retângulo')
        if self.is_Moving:
            coord1, coord2 = coordinates
            cx, cy = circle_center
            
            x1, y1 = self.rect_coord1
            x2, y2 = self.rect_coord2

            pts = np.array([[(coord1[0], coord1[1]), (coord2[0], coord1[1]), 
                        (coord2[0], coord2[1]), (coord1[0], coord2[1])]])
                                                         
            is_inside = cv.pointPolygonTest(pts, circle_center, False)
            if is_inside > 0:
                mid_point = (cx, cy)
                mid_x, mid_y = mid_point
                
                self.rect_coord1 = (int(mid_x - self.rect_width//2), int(mid_y - self.rect_height//2))
                self.rect_coord2 = (int(mid_x + self.rect_width//2), int(mid_y + self.rect_height//2))
             
           
    def run(self):
        
        while self._cam.isOpened():
            success, frame = self._cam.read()

            self._showFps(frame)

            self._findHands(frame)           
            
            self._drawButton(frame)

            frame = self._drawRect(frame, (255, 0, 0), self.rect_coord1, self.rect_coord2)
            
            cv.imshow('Hand tracker', frame)
                        
            if cv.waitKey(2) == ord('q'):
                break