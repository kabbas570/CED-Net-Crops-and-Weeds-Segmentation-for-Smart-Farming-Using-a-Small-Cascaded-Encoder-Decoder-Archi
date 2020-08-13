import tensorflow.keras.backend as K
def dice_coef(y_true, y_pred, smooth=2):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred) 