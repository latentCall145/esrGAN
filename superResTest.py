import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os, cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
parentPath = 'models/tf_220' # change to tf_230 if your machine doesn't support TF 2.2.0

nameGens = ['srGAN', 'esrGAN_DB', 'esrGAN_RRDB', 'esrGAN_RRDB_v2'] # RRDB v2 model only compatible for TF 2.2.0
genList = []
for i in nameGens:
    genList.append(tf.keras.models.load_model(os.path.join(parentPath, i, 'gen')))
numGens = len(genList)

def bigPred(x, gen): # upscale non-32x32 images; x=np.array, gen=Keras model
  m, h, w, c = x.shape
  ret = np.zeros((m, 4*h, 4*w, c))
  for i in range(0, h//32):
    for j in range(0, w//32):
      ret[:, 128*i:128*(i+1), 128*j:128*(j+1), :] = gen.predict(x[:, 32*i:32*(i+1), 32*j:32*(j+1), :])
  return ret

os.chdir('testImages')

imageName = 'turtle.png'
img = cv2.imread(imageName) # BGR -> RGB, divide by 255 to normalize images
img = img[:, :, ::-1] / 255
img = np.expand_dims(img, 0)

ySR = []
for i in genList:
    ySR.append(bigPred(img, i))

fig, axes = plt.subplots(nrows=2, ncols=numGens)

for i in range(numGens):
  axes[0][i].imshow(img[0])
  axes[0][i].set_title('Source Image')
  axes[1][i].imshow(ySR[i][0])
  axes[1][i].set_title(nameGens[i])

plt.show()
