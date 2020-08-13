import tensorflow.keras as keras
def Upsample_Conc(result,C):  
    D=keras.layers.UpSampling2D(size = (2,2))(result)
    x = keras.layers.concatenate([D, C],axis=-1)
    return x
