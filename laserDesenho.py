# -*- coding: utf-8 -*-
import cv2 as cv

import numpy as np

verdeMinimo = (60, 64, 127)
verdeMaximo = (75, 255, 255)

video = cv.VideoCapture(1)


telaDesenho = np.zeros((400,400,3),np.uint8)

cores = [
    (255,255,255),
    (255,0,0),
    (255,255,0),
    (0,255,0),
    (0,0,255)
    ]


cor_linha = cores[np.random.randint(0,len(cores))]

filme = cv.VideoWriter('./desenho.mp4',cv.VideoWriter.fourcc(*'mp4v'),15,(800,400))

while True:
    
    isCap,frame = video.read()
    
    if isCap:
        
        frame = cv.resize(frame,(400,400))
        imagemTabalho = frame.copy()
        
        blur = cv.GaussianBlur(imagemTabalho,(11,11),0)
        # cv.imshow('Blur',imagemTabalho)
        
        hsv = cv.cvtColor(blur,cv.COLOR_BGR2HSV)
        
        # cv.imshow('HSV',hsv)

        mask = cv.inRange(hsv, verdeMinimo,verdeMaximo)
        
        mask = cv.erode(mask,None,iterations=2)
        
        mask = cv.dilate(mask,None,iterations=2)
        
        # cv.imshow('Mascara',mask)
        
        contours,_ = cv.findContours(mask.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        
        
        if len(contours) >0:
            
            c = max(contours, key=cv.contourArea)
                        
            M = cv.moments(c)
           
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
               
               
            cv.circle(telaDesenho,
                        (center),
                        1,
                        cor_linha,
                    2,
                    cv.FILLED)
        
        
        
        
        cv.putText(imagemTabalho,
           "====================================================",
           (0,350),
           cv.FONT_HERSHEY_COMPLEX_SMALL,
           0.7,
           (255,127,127),
           1,
           cv.LINE_AA)
        
        cv.putText(imagemTabalho,
           "( q : Sair) (c : Mudar Cor) (n : Limpar Tela)",
           (1,365),
           cv.FONT_HERSHEY_COMPLEX_SMALL,
           0.7,
           (255,127,127),
           1,
           cv.LINE_AA)
        
        
        resultado = cv.hconcat([imagemTabalho,telaDesenho])
        

        
        cv.imshow('Tela',resultado)
        filme.write(resultado)
        tecla = cv.waitKey(100)
        
        if tecla == ord('q'):
            break
        elif tecla == ord('n'):
            telaDesenho = np.zeros((400,400,3),np.uint8)
        elif tecla == ord('c'):
            cor_linha = cores[np.random.randint(0,len(cores))]

video.release()
filme.release()
cv.destroyAllWindows()


