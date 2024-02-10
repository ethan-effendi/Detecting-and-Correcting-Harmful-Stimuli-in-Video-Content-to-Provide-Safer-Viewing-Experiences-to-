import multiprocessing
# !pip install tensorflow_decision_forests
import tensorflow as tf
import keras as keras
import pandas as pd
import numpy as np
import cv2
import PIL
from PIL import Image
from PIL import ImageColor
import os
import csv
import shutil
import math
import tensorflow_decision_forests as tfdf
import time
import multiprocessing

def getDimensions(img):
  img = Image.open(img)
  width = img.width
  height = img.height
  return width, height

def trainAndReturnModel():
  names=['blinking','f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30', 'f31', 'f32', 'f33', 'f34', 'f35', 'f36', 'f37', 'f38', 'f39', 'f40', 'f41', 'f42', 'f43', 'f44', 'f45', 'f46', 'f47', 'f48', 'f49', 'f50', 'f51', 'f52', 'f53', 'f54', 'f55', 'f56', 'f57', 'f58', 'f59', 'f60', 'f61', 'f62', 'f63', 'f64', 'f65', 'f66', 'f67', 'f68', 'f69', 'f70', 'f71', 'f72', 'f73', 'f74', 'f75', 'f76', 'f77', 'f78', 'f79', 'f80', 'f81', 'f82', 'f83', 'f84', 'f85', 'f86', 'f87', 'f88', 'f89', 'f90', 'f91', 'f92', 'f93', 'f94', 'f95', 'f96', 'f97', 'f98', 'f99', 'f100', 'f101', 'f102', 'f103', 'f104', 'f105', 'f106', 'f107', 'f108', 'f109', 'f110', 'f111', 'f112', 'f113', 'f114', 'f115', 'f116', 'f117', 'f118', 'f119', 'f120', 'f121', 'f122', 'f123', 'f124', 'f125', 'f126', 'f127', 'f128', 'f129', 'f130', 'f131', 'f132', 'f133', 'f134', 'f135', 'f136', 'f137', 'f138', 'f139', 'f140', 'f141', 'f142', 'f143', 'f144', 'f145', 'f146', 'f147', 'f148', 'f149', 'f150', 'f151', 'f152', 'f153', 'f154', 'f155', 'f156', 'f157', 'f158', 'f159', 'f160', 'f161', 'f162', 'f163', 'f164', 'f165', 'f166', 'f167', 'f168', 'f169', 'f170', 'f171', 'f172', 'f173', 'f174', 'f175', 'f176', 'f177', 'f178', 'f179', 'f180', 'f181', 'f182', 'f183', 'f184', 'f185', 'f186', 'f187', 'f188', 'f189', 'f190', 'f191', 'f192', 'f193', 'f194', 'f195', 'f196', 'f197', 'f198', 'f199', 'f200', 'f201', 'f202', 'f203', 'f204', 'f205', 'f206', 'f207', 'f208', 'f209', 'f210', 'f211', 'f212', 'f213', 'f214', 'f215', 'f216', 'f217', 'f218', 'f219', 'f220', 'f221', 'f222', 'f223', 'f224', 'f225', 'f226', 'f227', 'f228', 'f229', 'f230', 'f231', 'f232', 'f233', 'f234', 'f235', 'f236', 'f237', 'f238', 'f239', 'f240', 'f241', 'f242']
  trainingFile = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSQrx7ff664E_8DBPIWowS5VYlTkRKT6KJDGLAjv8FF7REhxzCwS4ceU-han-kWGKiTG8ZC3kRMCy0N/pub?gid=0&single=true&output=csv', names = names, skipinitialspace=True, skiprows=1)
  trainingFileFeatures = trainingFile.copy()
  trainingFileLabels = trainingFileFeatures.pop("blinking")
  trainingFileFeatures = np.array(trainingFileFeatures)
  trainingFileFeatures.shape
  model = tfdf.keras.GradientBoostedTreesModel()
  model.fit(trainingFileFeatures, trainingFileLabels)
  return model

def returnPixelVals(videoURL):
  returnTensor = []
  videoCapture = cv2.VideoCapture(videoURL)
  height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
  width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
  while True:
      success, currentImage = videoCapture.read()
      if success:
        currentImage = currentImage.tolist()
        for r in range(height):
          for c in range(width):
            currentImage[r][c][0] = int(('#%02x%02x%02x' % (currentImage[r][c][2], currentImage[r][c][1], currentImage[r][c][0]))[1:], 16)
            currentImage[r][c].pop(2)
            currentImage[r][c].pop(1)
            returnTensor.append(currentImage)
      else:
        break
  return np.array(returnTensor)

def createCSV(inputList):
  pad = 0
  values = [[]]
  if len(inputList) != 0:
    pad = inputList[-1]
  for i in range(len(inputList)):
    values[0].append(inputList[i])
  neededPad = 242-len(values[0])
  for i in range(neededPad):
    values[0].append(pad)
  return np.array(values)

def decode(integer):
  asHex = '{0:06X}'.format(integer)
  asHex = '#'+str(asHex)
  rgb = ImageColor.getcolor(asHex, "RGB")
  return rgb

def checkSetAgainstModel(model, inputList):
  lastCorrected = 0
  correctionColor = inputList[0]
  for i in range(len(inputList)):
    prediction = model.predict(createCSV(inputList[lastCorrected:i+1]),verbose = 0)
    if prediction >= 0.55:
      noCorrectionCount = 0
      for v in range(len(inputList[lastCorrected:i+1])):
        inputList[lastCorrected+v] = correctionColor
        lastCorrected = lastCorrected+1+v
    else:
      noCorrectionCount+=1
  return inputList


def blinkingCorrection(video, outputVidName):
  pixValsTensor = returnPixelVals(video)
  heightOfFrame = len(pixValsTensor[0])
  widthOfFrame = len(pixValsTensor[0][0])
  model = trainAndReturnModel()
  for w in range(widthOfFrame):
    for h in range(heightOfFrame):
      pixelValsList = []
      for i in range(len(pixValsTensor)):
        pixelValsList.append(pixValsTensor[i][w][h][0])
      correctedValsList = checkSetAgainstModel(model, pixelValsList.copy())
      if correctedValsList != pixelValsList:
        for t in range(len(pixValsTensor)):
          pixValsTensor[i][w][h][0] = correctedValsList[t]
  pixValsTensor.tolist()
  for z in range(pixValsTensor):
    for y in range(widthOfFrame):
      for j in range(heightOfFrame):
        r,g,b = decode(pixValsTensor[z][y][j][0])
        pixValsTensor[z][y][j].pop(0)
        pixValsTensor[z][y][j].append(r)
        pixValsTensor[z][y][j].append(g)
        pixValsTensor[z][y][j].append(b)
  pixValsTensor = np.array(pixValsTensor)
  video = cv2.VideoWriter(outputVidName, cv2.VideoWriter_fourcc(*'DIVX'), 60, (widthOfFrame, heightOfFrame))
  for p in range(len(pixValsTensor)):
    video.write(pixValsTensor[p])
  video.release()

blinkingCorrection('/content/29hz.mov', '/content/corr29hz.mov')
