# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 01:01:48 2022

@author: Propietario
"""

 
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# contenido = os.listdir('I:/datas/archive_COVID/COVID-19_Radiography_Dataset/COVID/images')
contenido = os.listdir('I:/datas/archive_COVID/COVID-19_Radiography_Dataset/Normal/images')
print('número de archivos leídos: ',len(contenido))

def fourier( ):
   
  for nombre in contenido:
    # Leemos cada archivo de imagen
    img = cv2.imread('I:/datas/archive_COVID/COVID-19_Radiography_Dataset/Normal/images/' + nombre,0)
    mascara=cv2.imread('I:/datas/archive_COVID/COVID-19_Radiography_Dataset/Normal/masks/' + nombre,0)
    
    mascara=cv2.resize(mascara, (299,299), interpolation = cv2.INTER_AREA)
    
    #img = cv2.imread('I:/datas/archive_COVID/COVID-19_Radiography_Dataset/Normal/images/' + nombre,0)
    #aplicamos Transformada de Fourier
    # img=cv2.bitwise_and(img, mask)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    
    dft_shift = np.fft.fftshift(dft)

    # magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    #se crea la máscara a aplicar
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols, 2), np.uint8)

    r = 1.1

    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0
    #se aplica la máscara
    fshift = dft_shift * mask

    # fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
    
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    # print(len(img_back), len(mask),len(img_back[0][0]),len(mask[0][0]))
    # print(img_back[298])
    # print(mask2)
    # result = cv2.bitwise_and(img_back, mask)
    #se obtiene la transformada inversa de Fourier
    
    rows2, cols2 = mascara.shape    
    mask = np.ones((rows2, cols2, 2), np.uint8)

    mask_area=np.ndarray(shape=(rows,cols), dtype=bool)
    for m in range(rows2):
        for n in range(cols2):
          if mascara[m][n]==0:
            mask_area[m][n]=True
          else:
            mask_area[m][n]=False

    mask[mask_area] = 0
    img_back=img_back * mask
    
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    # result = cv2.bitwise_and(img_back, mask2)
    
    
    
    fig = plt.figure(figsize=(4.16, 4.16))
    ax1 = fig.add_subplot(1,1,1)
    ax1 = fig.add_axes([0.0, 0.0, 1, 1])
    ax1.imshow(img_back, cmap='gray')
    plt.axis('equal')
    plt.axis('off')
    # ruta="I:/datas/modificados_covid/COVID-19_Radiography_Dataset/COVID/transformada_inversa_+_mascara/"+nombre
    ruta="I:/datas/modificados_covid/COVID-19_Radiography_Dataset/Normal/transformada_inversa_+_mascara_pulmon/"+nombre
    # ruta="I:/datas/modificados_covid/COVID-19_Radiography_Dataset/normal/images/"+nombre
    #ruta="I:/datas/modificados_covid/COVID-19_Radiography_Dataset/normal/transformada_inversa_+_mascara/"+nombre
    plt.savefig(ruta)
    plt.close(fig)
    
#llamada a la función
fourier( )
