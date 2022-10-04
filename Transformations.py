#!/usr/bin/env python
# coding: utf-8

# # Taller 3 - transformacion de imagenes
# Stephanie Gonzalez López

# ## Librerias

# In[2]:


from turtle import circle
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import Image


# ## Lista de funciones:

# - Mask (img_1_rgb) #Enmascarar
# - InvMask (img_1_mask) #Invertir Mascara
# - Dim (img_2_rgb) #Dimensionar
# - fg_mask (img_fg) #Filtrar imagen 1
# - bg_mask (img_bg) #Filtrar imagen 2
# - New_Image (img1_final) #Unir dos imagenes
# - Change_Color (img_f) #Cambiar escala de colores RGB
# - HSV (img_f_hsv) #Cambiar a escala de colores HSV
# - hsv_new (img_f_hsv) #Modificar canales de HSV y tener una imagen modificada
# - Crop (img_f_crop) #Recortar
# - Flip_h (img_f_rgb_flipped_horz) #Voltear horizontalmente
# - Flip_v (img_f_rgb_flipped_vert) #Voltear verticalmente
# - Flip_b (img_f_rgb_flipped_both) #Voltear de ambos lados
# - Rotate1 (img_f_rot1) #Rotar(1)
# - Rotate2 (img_f_rot2) #Rotar(2)
# - Des (img_f_des) #Desplazar
# - Line (img_f_line) #Dibujar una linea
# - Circle (img_f_circle) #Dibujar un circulo
# - Rect (img_f_rect) #Dibujar un rectangulo
# - Text (img_f_text) #Escribir un texto
# - Bright(img_f_b) #Brillo

# ## Logo (Imagen 1)

# In[9]:


img_1=cv.imread("logo.png")
img_1_rgb=cv.cvtColor(img_1, cv.COLOR_BGR2RGB)
plt.imshow(img_1_rgb)
print(img_1_rgb.shape)
logo_w=img_1_rgb.shape[0]
logo_h=img_1_rgb.shape[1]


# Mask para la primera imagen (logo)

# In[57]:


def Mask (img_1_rgb):
    img_gray=cv.cvtColor(img_1_rgb, cv.COLOR_RGB2GRAY)
    revtal, img_1_mask=cv.threshold(img_gray, 230, 255, cv.THRESH_BINARY)
    plt.imshow(img_1_mask, cmap="gray")
    print(img_1_mask.shape)
    return(img_1_rgb)
# Invertimos Mask

# In[47]:


def InvMask (img_1_mask):
    mask_inv=cv.bitwise_not(img_1_mask)
    plt.imshow(mask_inv, cmap="gray")
    return(img_1_mask)


# ## Fondo de la imagen (imagen 2)

# In[32]:


img_2=cv.imread("fondo2.jpg")
img_2_rgb=cv.cvtColor(img_2, cv.COLOR_BGR2RGB)
plt.imshow(img_2_rgb)
print(img_2_rgb.shape)


# Dimensionamiento de la imagen

# In[45]:


def Dim (img_2_rgb):
    img_2_rgb=cv.cvtColor(img_2, cv.COLOR_BGR2RGB)
    aspect_ratio=logo_w/img_2_rgb.shape[1]
    dim=(768,548)
    img_2_rgb=cv.resize(img_2_rgb, dim, interpolation=cv.INTER_AREA)
    plt.imshow(img_2_rgb)
    print(img_2_rgb.shape)
    return(img_2_rgb)


# Aplicamos el fondo de la imagen con Mask

# In[80]:


def bg_mask (img_bg):
    img_gray=cv.cvtColor(img_1_rgb, cv.COLOR_RGB2GRAY)
    revtal, img_1_mask=cv.threshold(img_gray, 230, 255, cv.THRESH_BINARY)
    mask_inv=cv.bitwise_not(img_1_mask)
    img_bg=cv.bitwise_and(img_2_rgb, img_2_rgb, mask=mask_inv)
    plt.imshow(img_bg)
    return(img_bg)


# In[88]:


def fg_mask (img_fg):
    img_gray=cv.cvtColor(img_1_rgb, cv.COLOR_RGB2GRAY)
    img_1_mask=cv.threshold(img_gray, 230, 255, cv.THRESH_BINARY)
    img_fg=cv.bitwise_and(img_1_rgb, img_1_rgb, mask=img_1_mask)
    plt.imshow(img_fg)
    return(img_fg)


# Logo y fondo de la imagen unidas

# In[91]:


def New_Image (img1_final):
    img_gray=cv.cvtColor(img_1_rgb, cv.COLOR_RGB2GRAY)
    revtal, img_1_mask=cv.threshold(img_gray, 230, 255, cv.THRESH_BINARY)
    mask_inv=cv.bitwise_not(img_1_mask)
    img_fg=cv.bitwise_and(img_1_rgb, img_1_rgb, mask=img_1_mask)
    img_bg=cv.bitwise_and(img_2_rgb, img_2_rgb, mask=mask_inv)
    img1_final= cv.add(img_bg, img_fg)
    plt.imshow(img1_final)
    cv.imwrite("logo_fondo.png", img1_final[:,:,::-1])
    return(img1_final)


# In[96]:


img_f= cv.imread("logo_fondo.png", cv.IMREAD_COLOR) 
img_f_rgb=cv.cvtColor(img_f, cv.COLOR_BGR2RGB)
plt.imshow(img_f_rgb)


# Cambiando la escala de colores de la imagen resultante

# In[101]:


def Change_Color (img_f):
    b,g,r = cv.split(img_f)
    plt.figure(figsize=[25,5])
    plt.subplot(151);plt.imshow(img_f[:,:,::-1],cmap='gray');plt.title("Original");
    plt.subplot(152);plt.imshow(r,cmap='gray');plt.title("Red Channel");
    plt.subplot(153);plt.imshow(g,cmap='gray');plt.title("Green Channel");
    plt.subplot(154);plt.imshow(b,cmap='gray');plt.title("Blue Channel");
    img_merged = cv.merge((b,g,r))
    plt.subplot(155);plt.imshow(img_merged[:,:,::-1]);plt.title("Merged Output");
    return(img_f)


# Cambiando a HSV color space

# In[110]:


def HSV (img_f_hsv):
    img_f_hsv = cv.cvtColor(img_f, cv.COLOR_BGR2HSV)
    h,s,v = cv.split(img_f_hsv)
    plt.figure(figsize=[20,5])
    plt.subplot(141);plt.imshow(h,cmap='gray');plt.title("H Channel");
    plt.subplot(142);plt.imshow(s,cmap='gray');plt.title("S Channel");
    plt.subplot(143);plt.imshow(v,cmap='gray');plt.title("V Channel");
    plt.subplot(144);plt.imshow(img_f_rgb);plt.title("Original");
    return (img_f_hsv)


# Modificamos cada canal

# In[131]:


def hsv_new (img_f_hsv):
    img_hsv = cv.cvtColor(img_f, cv.COLOR_BGR2HSV)
    h,s,v = cv.split(img_hsv)
    h_new = h+100
    img_merged_hsv = cv.merge((h_new,s,v))
    img_f_hsv = cv.cvtColor(img_merged_hsv, cv.COLOR_HSV2RGB)
    plt.figure(figsize=[20,5])
    plt.subplot(141);plt.imshow(h_new,cmap='gray');plt.title("H Channel");
    plt.subplot(142);plt.imshow(s,cmap='gray');plt.title("S Channel");
    plt.subplot(143);plt.imshow(v,cmap='gray');plt.title("V Channel");
    plt.subplot(144);plt.imshow(img_f);plt.title("Modified");
    cv.imwrite("HSV.png", img_f_hsv[:,:,::-1])
    return(img_f_hsv)


# Ahora haremos lo siguiente con la imagen logo_fondo.png: 
# - Desplazar
# - Rotar
# - Voltear
# - Cortar

# Recorte

# In[135]:


def Crop (img_f_crop):
    img_f_crop= img_f_rgb[200:350, 160:600]
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img_f_rgb)
    plt.subplot(1,2,2)
    plt.imshow(img_f_crop)
    cv.imwrite("recorte.png", img_f_crop[:,:,::-1])
    return(img_f_crop)


# Volteamos la imagen horizontalmente

# In[143]:


def Flip_h (img_f_rgb_flipped_horz):
    img_f_rgb_flipped_horz = cv.flip(img_f_rgb, 1)
    plt.figure(figsize=[18,5])
    plt.subplot(141);plt.imshow(img_f_rgb);plt.title("Original");
    plt.subplot(142);plt.imshow(img_f_rgb_flipped_horz);plt.title("Horizontal Flip");
    cv.imwrite("flip_h.png", img_f_rgb_flipped_horz[:,:,::-1])
    return(img_f_rgb_flipped_horz)


# Volteamos la imagen verticalmente

# In[144]:


def Flip_v (img_f_rgb_flipped_vert):
    img_f_rgb_flipped_vert = cv.flip(img_f_rgb, 0)
    plt.figure(figsize=[18,5])
    plt.subplot(141);plt.imshow(img_f_rgb);plt.title("Original");
    plt.subplot(142);plt.imshow(img_f_rgb_flipped_vert);plt.title("Vertical Flip");
    cv.imwrite("flip_v.png", img_f_rgb_flipped_vert[:,:,::-1])
    return(img_f_rgb_flipped_vert)


# Volteamos la imagen de ambos lados

# In[147]:


def Flip_b (img_f_rgb_flipped_both):
    img_f_rgb_flipped_both = cv.flip(img_f_rgb, -1)
    plt.figure(figsize=[18,5])
    plt.subplot(141);plt.imshow(img_f_rgb);plt.title("Original");
    plt.subplot(142);plt.imshow(img_f_rgb_flipped_both);plt.title("Both Flipped");
    cv.imwrite("flip_b.png", img_f_rgb_flipped_both[:,:,::-1])
    return(img_f_rgb_flipped_both)


# Rotamos la imagen 

# In[153]:


def Rotate1 (img_f_rot1):
    img_f_rot1 = cv.rotate(img_f_rgb, cv.ROTATE_90_CLOCKWISE)
    plt.figure(figsize=[18,5])
    plt.subplot(141);plt.imshow(img_f_rgb);plt.title("Original");
    plt.subplot(142);plt.imshow(img_f_rot1);plt.title("Rotated Image");
    cv.imwrite("rotate1.png", img_f_rot1[:,:,::-1])
    return(img_f_rot1)


# In[157]:


def Rotate2 (img_f_rot2):
    (h, w) = img_f_rgb.shape[:2]
    center = (w / 2, h / 2)
    angle = 30
    scale = 1
    M = cv.getRotationMatrix2D(center, angle, scale)
    img_f_rot2 = cv.warpAffine(img_f_rgb, M, (w, h))
    plt.figure(figsize=[18,5])
    plt.subplot(141);plt.imshow(img_f_rgb);plt.title("Original");
    plt.subplot(142);plt.imshow(img_f_rot2);plt.title("Rotated Image");
    cv.imwrite("rotate2.png", img_f_rot2[:,:,::-1])
    return(img_f_rot2)


# Desplazamiento de la imagen

# In[161]:


def Des (img_f_des):
    he = img_f_rgb.shape[1] #columnas
    we = img_f_rgb.shape[0] # filas
    M = np.float32([[1,0,100],[0,1,150]])
    img_f_des = cv.warpAffine(img_f_rgb,M,(he, we))
    plt.figure(figsize=[18,5])
    plt.subplot(141);plt.imshow(img_f_rgb);plt.title("Original");
    plt.subplot(142);plt.imshow(img_f_des);plt.title("Shifted Image");
    cv.imwrite("Desplazamiento.png", img_f_des[:,:,::-1])
    return(img_f_des)


# Dibujamos en la imagen logo_fondo.png (círculos, rectángulos, líneas, texto).

# In[167]:


#Dibujamos una linea
def Line (img_f_line):
    img_f_line = img_f_rgb.copy()
    cv.line(img_f_line, (150, 100), (300, 200),(0, 0, 255), thickness=5, lineType=cv.LINE_AA);
    plt.imshow(img_f_line[:,:,::-1])
    cv.imwrite("Line.png", img_f_line[:,:,::-1])
    return(img_f_line)


# In[170]:


#Dibujamos un circulo
def Circle (img_f_circle):
    img_f_circle = img_f_rgb.copy()
    cv.circle(img_f_circle, (500,350), 100, (0, 0, 255), thickness=5, lineType=cv.LINE_AA);
    plt.imshow(img_f_circle[:,:,::-1])
    cv.imwrite("circle.png", img_f_circle[:,:,::-1])
    return (img_f_circle)



# In[173]:


#Dibujamos un rectangulo
def Rect (img_f_rect):
    img_f_rect = img_f_rgb.copy()
    cv.rectangle(img_f_rect, (660, 350), (100,200), (0, 0, 255), thickness=5, lineType=cv.LINE_8);
    plt.imshow(img_f_rect[:,:,::-1])
    cv.imwrite("rectangle.png", img_f_rect[:,:,::-1])
    return (img_f_rect)


# In[177]:


#Agregamos un texto
def Text (img_f_text):
    img_f_text = img_f_rgb.copy()
    text = "Albums mas escuchados en Spotify 2021"
    fontScale = 1
    fontFace = cv.FONT_ITALIC 
    fontColor = (0, 0, 255)
    fontThickness = 2
    cv.putText(img_f_text, text, (50, 500), fontFace, fontScale, fontColor, fontThickness, cv.LINE_AA);
    plt.imshow(img_f_text[:,:,::-1])
    cv.imwrite("text.png", img_f_text[:,:,::-1])
    return(img_f_text)


# Brillo de la imagen

# In[199]:


def Bright(img_f_b):
    n=100
    matrix = np.ones(img_f_rgb.shape, dtype='uint8')*n
    n=(0,1)
    if n==1:  
        img_f_b = cv.add(img_f_rgb, matrix)
    else:
        img_f_b = cv.subtract(img_f_rgb, matrix) 
    plt.figure(figsize=[17, 10])
    plt.subplot(121); plt.imshow(img_f_rgb); plt.title('Original')
    plt.subplot(122); plt.imshow(img_f_b); plt.title('Bright change')
    cv.imwrite("Brillo.png", img_f_b[:,:,::-1])
    return(img_f_b)
    

