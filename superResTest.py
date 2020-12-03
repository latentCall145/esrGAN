import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os, cv2

parentPath = 'models/tf_230' # change to tf_220 if your machine doesn't support TF 2.3.0
# srGAN
srGAN = tf.keras.models.load_model(os.path.join(parentPath, 'srGAN/gen'))

# esrGAN
# no RRDB model - just dense blocks
esrGAN = tf.keras.models.load_model(os.path.join(parentPath, 'esrGAN_DB/gen'))

# RRDB model
esrGAN_RRDB = tf.keras.models.load_model(os.path.join(parentPath, 'esrGAN_RRDB/gen'))

def bigPred(x, gen): # upscale non-32x32 images; x=np.array, gen=Keras model
  m, h, w, c = x.shape
  ret = np.zeros((m, 4*h, 4*w, c))
  for i in range(0, h//32):
    for j in range(0, w//32):
      ret[:, 128*i:128*(i+1), 128*j:128*(j+1), :] = gen.predict(x[:, 32*i:32*(i+1), 32*j:32*(j+1), :])
  return ret

os.chdir('testImages')

imageName = 'turtle128.png'
img = cv2.imread(imageName) # BGR -> RGB, divide by 255 to normalize images
img = img[:, :, ::-1] / 255
img = np.expand_dims(img, 0)

y_sr = bigPred(img, srGAN)
y_esr = bigPred(img, esrGAN)
y_rrdb = bigPred(img, esrGAN_RRDB)

fig, axes = plt.subplots(nrows=2, ncols=3)

for i in range(3):
  axes[0][i].imshow(img[0])
  axes[0][i].set_title('Source Image')

axes[1][0].imshow(y_sr[0])
axes[1][1].imshow(y_esr[0])
axes[1][2].imshow(y_rrdb[0])

axes[1][0].set_title('srGAN Super-Res')
axes[1][1].set_title('esrGAN (Dense Block) Super-Res')
axes[1][2].set_title('esrGAN (RRDB) Super-Res')

plt.show()
