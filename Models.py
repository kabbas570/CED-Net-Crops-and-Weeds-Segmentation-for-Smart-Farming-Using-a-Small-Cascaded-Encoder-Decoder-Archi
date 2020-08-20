import tensorflow.keras as keras
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.layers import MaxPooling2D
def UNet(input_size = (896,896,3)):
    inputs = keras.layers.Input(input_size)
    conv1 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1=keras.layers.BatchNormalization()(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2=keras.layers.BatchNormalization()(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3=keras.layers.BatchNormalization()(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4=keras.layers.BatchNormalization()(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5=keras.layers.BatchNormalization()(conv5)
    drop5 = keras.layers.Dropout(0.5)(conv5)
    up6 = keras.layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(drop5))
    merge6 = keras.layers.concatenate([conv4,up6], axis = 3)
    conv6 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6=  keras.layers.BatchNormalization()(conv6)
    up7 = keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv6))
    merge7 = keras.layers.concatenate([conv3,up7], axis = 3)
    conv7 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7=keras.layers.BatchNormalization()(conv7)
    up8 = keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv7))
    merge8 = keras.layers.concatenate([conv2,up8], axis = 3)
    conv8 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8=keras.layers.BatchNormalization()(conv8)
    up9 = keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv8))
    merge9 = keras.layers.concatenate([conv1,up9], axis = 3)
    conv9 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 =keras.layers.Conv2D(2, 1, activation = 'sigmoid')(conv9)
    model = keras.models.Model(inputs = inputs, outputs = conv10)
    return model
def segnet(
        input_size = (896,896,3)):
    # Block 1
    inputs = keras.layers.Input(input_size)
    x = keras.layers.Conv2D(64, (3, 3),activation='relu',padding='same',name='block1_conv1',kernel_initializer = 'he_normal')(inputs)
    x = keras.layers.Conv2D(64, (3, 3),activation='relu',padding='same',name='block1_conv2',kernel_initializer = 'he_normal')(x)
    x=keras.layers.BatchNormalization()(x)
    pool_1 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # Block 2
    x = keras.layers.Conv2D(128, (3, 3), activation='relu',padding='same',name='block2_conv1',kernel_initializer = 'he_normal')(pool_1)
    x = keras.layers.Conv2D(128, (3, 3),activation='relu',padding='same',name='block2_conv2',kernel_initializer = 'he_normal')(x)
    x=keras.layers.BatchNormalization()(x)
    pool_2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
     # Block 3
    x = keras.layers.Conv2D(256, (3, 3),activation='relu',padding='same',name='block3_conv1',kernel_initializer = 'he_normal')(pool_2)
    x = keras.layers.Conv2D(256, (3, 3),activation='relu',padding='same',name='block3_conv2',kernel_initializer = 'he_normal')(x)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block3_conv3',kernel_initializer = 'he_normal')(x)
    x=keras.layers.BatchNormalization()(x)
    pool_3 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # Block 4
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv1',kernel_initializer = 'he_normal')(pool_3)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv2',kernel_initializer = 'he_normal')(x)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv3',kernel_initializer = 'he_normal')(x)
    x=keras.layers.BatchNormalization()(x)
    pool_4 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    # Block 5
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv1',kernel_initializer = 'he_normal')(pool_4)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv2',kernel_initializer = 'he_normal')(x)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu', padding='same',name='block5_conv3',kernel_initializer = 'he_normal')(x)
    x=keras.layers.BatchNormalization()(x)
    pool_5 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    #DECONV_BLOCK
    #Block_1
    unpool_1=keras.layers.UpSampling2D(size = (2,2))(pool_5)
    conv_14= keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(unpool_1)
    conv_15 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_14)
    conv_16 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_15)
    conv_16= keras.layers.BatchNormalization()(conv_16)
    #Block_2
    unpool_2 = keras.layers.UpSampling2D(size = (2,2))(conv_16)  
    conv_17= keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(unpool_2)
    conv_18 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_17)
    conv_19 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_18)
    conv_19= keras.layers.BatchNormalization()(conv_19)
    #Block_3
    unpool_3 =  keras.layers.UpSampling2D(size = (2,2))(conv_19)   
    conv_20= keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(unpool_3)
    conv_21 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_20)
    conv_22 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_21)
    conv_22= keras.layers.BatchNormalization()(conv_22)
    #Block_4
    unpool_4 = keras.layers.UpSampling2D(size = (2,2))(conv_22)  
    conv_23= keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(unpool_4)
    conv_24 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_23)
    conv_24 = keras.layers.BatchNormalization()(conv_24) 
    #BLock_5
    unpool_5 =keras.layers.UpSampling2D(size = (2,2))(conv_24) 
    conv_25= keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(unpool_5)
    conv_26 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_25)
    conv_26 = keras.layers.BatchNormalization()(conv_26)
    out=keras.layers.Conv2D(2,1, activation = 'sigmoid', padding = 'same',kernel_initializer = 'he_normal')(conv_26)
    print("Build decoder done..")
    model = keras.models.Model(inputs=inputs, outputs=out, name="SegNet")
    return model
def fcn_8(input_size = (896,896,3)): 
     # Block 1
    inputs = keras.layers.Input(input_size)
    x = keras.layers.Conv2D(64, (3, 3),activation='relu',padding='same',name='block1_conv1',kernel_initializer = 'he_normal')(inputs)
    x = keras.layers.Conv2D(64, (3, 3),activation='relu',padding='same',name='block1_conv2',kernel_initializer = 'he_normal')(x)
    x=keras.layers.BatchNormalization()(x)
    pool_1 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # Block 2
    x = keras.layers.Conv2D(128, (3, 3), activation='relu',padding='same',name='block2_conv1',kernel_initializer = 'he_normal')(pool_1)
    x = keras.layers.Conv2D(128, (3, 3),activation='relu',padding='same',name='block2_conv2',kernel_initializer = 'he_normal')(x)
    x=keras.layers.BatchNormalization()(x)
    pool_2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
     # Block 3
    x = keras.layers.Conv2D(256, (3, 3),activation='relu',padding='same',name='block3_conv1',kernel_initializer = 'he_normal')(pool_2)
    x = keras.layers.Conv2D(256, (3, 3),activation='relu',padding='same',name='block3_conv2',kernel_initializer = 'he_normal')(x)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block3_conv3',kernel_initializer = 'he_normal')(x)
    x=keras.layers.BatchNormalization()(x)
    pool_3 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # Block 4
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv1',kernel_initializer = 'he_normal')(pool_3)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv2',kernel_initializer = 'he_normal')(x)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv3',kernel_initializer = 'he_normal')(x)
    x=keras.layers.BatchNormalization()(x)
    pool_4 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    # Block 5
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv1',kernel_initializer = 'he_normal')(pool_4)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv2',kernel_initializer = 'he_normal')(x)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu', padding='same',name='block5_conv3',kernel_initializer = 'he_normal')(x)
    x=keras.layers.BatchNormalization()(x)
    pool_5 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
    # up convolutions and sum
    pool_5_U=keras.layers.UpSampling2D(size = (4,4),interpolation='bilinear')(pool_5)
    pool_5_U=keras.layers.BatchNormalization()(pool_5_U)
    pool_4_U=keras.layers.UpSampling2D(size = (2,2),interpolation='bilinear')(pool_4)
    pool_4_U=keras.layers.BatchNormalization()(pool_4_U)
    sum_1=keras.layers.Add()([pool_5_U,pool_4_U])  
    sum_2=keras.layers.Add()([sum_1,pool_3])  
    sum_2=keras.layers.BatchNormalization()(sum_2)
    x=keras.layers.UpSampling2D(size = (8,8),interpolation='bilinear')(sum_2)  
    x =keras.layers.Conv2D(2, 1, activation = 'sigmoid',kernel_initializer = 'he_normal')(x)
    model = keras.models.Model(inputs = inputs, outputs = x)
    return model
def Deeplab_v3(input_size = (896,896,3),  batchnorm = True):
    inputs = keras.layers.Input(input_size)
    
    x= Conv2D(3, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(inputs)
    c0 = BatchNormalization()(x)
    c0 = Activation('relu')(c0)
    c0 = Conv2D(32, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c0)
    c0 = Conv2D(32, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c0)
    c0 = BatchNormalization()(c0)
    c0 = Activation('relu')(c0)
    # Contracting Path
    c1 = Conv2D(64, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(c0)
    c1 =  Conv2D(64, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)  
    p1 = MaxPooling2D((2, 2))(c1)

    
    c2 = Conv2D(128, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(p1)
    c2 = Conv2D(128, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(c2)
    c2 = Conv2D(128, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2) 
    p2 = MaxPooling2D((2, 2))(c2)


    c3 = Conv2D(256, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(p2)
    c3 = Conv2D(256, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(c3)
    c3 = Conv2D(256, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3) 
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(p3)
    c4 = Conv2D(256, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(c4)
    c4 = Conv2D(256, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4) 
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(512, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(p4)
    c5 = Conv2D(512, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(c5)
    c5 = Conv2D(512, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5) 
    
    #################################################*****ASPP_v3+******##################################################
    x1 = Conv2D(512, kernel_size = (1, 1), kernel_initializer='he_normal',  padding = 'same')(c5)
    x1 = BatchNormalization()(x1)
    x8 = Conv2D(512, kernel_size = (3, 3), dilation_rate = 6, kernel_initializer='he_normal', padding = 'same')(c5)
    x8 = BatchNormalization()(x8)
    x16 = Conv2D(512, kernel_size = (3, 3), dilation_rate = 12, kernel_initializer='he_normal', padding = 'same')(c5)
    x16 = BatchNormalization()(x16)
    x24 = Conv2D(512, kernel_size = (3, 3), dilation_rate = 18, kernel_initializer='he_normal', padding = 'same')(c5)
    x24 = BatchNormalization()(x24)
    
    img = MaxPooling2D(pool_size=16, strides=16, padding='same')(inputs)
    c = concatenate([x1, x8, x16, x24, img])
    ctr = Conv2D(filters = 256, kernel_size = (1, 1), kernel_initializer = 'he_normal',  padding = 'same')(c)
    ###################################################################################################
    
    # Upsampling
    up = Conv2D(128, (1, 1), kernel_initializer = 'he_normal', activation='relu')(ctr)
    up = UpSampling2D(size=((4,4)), interpolation='bilinear')(up)#x4 times upsample 
    
    up1 = Conv2D(64, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(c3)
    upc = concatenate([up1, up])
    
    up2 = Conv2D(32, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(upc)
    up2 = UpSampling2D(size=((4,4)), interpolation='bilinear')(up2)#x4 times upsample 
    
    outputs = Conv2D(2, (1, 1), activation='sigmoid')(up2)
    model = Model(inputs=inputs, outputs=outputs)
    return model
   
    
def MYMODEL1(
        input_size = (448,448,3)):
    inputs = keras.layers.Input(input_size)
    
    conv1 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1=keras.layers.BatchNormalization()(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2=keras.layers.BatchNormalization()(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3=keras.layers.BatchNormalization()(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4=keras.layers.BatchNormalization()(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    
    u1=keras.layers.UpSampling2D(size = (2,2))(conv5)
    u1 = keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u1)
    c1 = keras.layers.concatenate([conv4,u1], axis = 3)
    c1 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c1)
    c1 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c1)
    c1=keras.layers.BatchNormalization()(c1)

    u2=keras.layers.UpSampling2D(size = (2,2))(c1)
    
    u2 = keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u2)
    c2 =keras.layers.concatenate([conv3,u2], axis = 3)
    c2 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c2)
    c2 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c2)
    c2=keras.layers.BatchNormalization()(c2)
    u3=keras.layers.UpSampling2D(size = (2,2))(c2)
    u3 = keras.layers.Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u3)
    c3 =keras.layers.concatenate([conv2,u3], axis = 3)
    c3 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c3)
    c3 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c3)
    c3=keras.layers.BatchNormalization()(c3)
    u4=keras.layers.UpSampling2D(size = (2,2))(c3)
    u4 = keras.layers.Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u4)
    c4 =keras.layers.concatenate([conv1,u4], axis = 3)
    c4 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c4)
    c4 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c4)
    out =keras.layers.Conv2D(1, 1, activation = 'sigmoid')(c4)
    model = keras.models.Model(inputs=inputs, outputs=out, name="SegNet")
    return model
def MYMODEL2(
        input_size = (896,896,4)):
    inputs = keras.layers.Input(input_size)
    
    conv1 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1=keras.layers.BatchNormalization()(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2=keras.layers.BatchNormalization()(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3=keras.layers.BatchNormalization()(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4=keras.layers.BatchNormalization()(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    
    u1=keras.layers.UpSampling2D(size = (2,2))(conv5)
    u1 = keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u1)
    c1 = keras.layers.concatenate([conv4,u1], axis = 3)
    c1 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c1)
    c1 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c1)
    c1=keras.layers.BatchNormalization()(c1)

    u2=keras.layers.UpSampling2D(size = (2,2))(c1)
    
    u2 = keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u2)
    c2 =keras.layers.concatenate([conv3,u2], axis = 3)
    c2 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c2)
    c2 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c2)
    c2=keras.layers.BatchNormalization()(c2)
    u3=keras.layers.UpSampling2D(size = (2,2))(c2)
    u3 = keras.layers.Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u3)
    c3 =keras.layers.concatenate([conv2,u3], axis = 3)
    c3 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c3)
    c3 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c3)
    c3=keras.layers.BatchNormalization()(c3)
    u4=keras.layers.UpSampling2D(size = (2,2))(c3)
    u4 = keras.layers.Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u4)
    c4 =keras.layers.concatenate([conv1,u4], axis = 3)
    c4 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c4)
    c4 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c4)
    out =keras.layers.Conv2D(1, 1, activation = 'sigmoid')(c4)
    model = keras.models.Model(inputs=inputs, outputs=out, name="SegNet")
    return model
def MYMODEL3(
        input_size = (448,448,3)):
    inputs = keras.layers.Input(input_size)
    
    conv1 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1=keras.layers.BatchNormalization()(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2=keras.layers.BatchNormalization()(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3=keras.layers.BatchNormalization()(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4=keras.layers.BatchNormalization()(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    
    u1=keras.layers.UpSampling2D(size = (2,2))(conv5)
    u1 = keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u1)
    c1 = keras.layers.concatenate([conv4,u1], axis = 3)
    c1 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c1)
    c1 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c1)
    c1=keras.layers.BatchNormalization()(c1)

    u2=keras.layers.UpSampling2D(size = (2,2))(c1)
    
    u2 = keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u2)
    c2 =keras.layers.concatenate([conv3,u2], axis = 3)
    c2 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c2)
    c2 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c2)
    c2=keras.layers.BatchNormalization()(c2)
    u3=keras.layers.UpSampling2D(size = (2,2))(c2)
    u3 = keras.layers.Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u3)
    c3 =keras.layers.concatenate([conv2,u3], axis = 3)
    c3 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c3)
    c3 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c3)
    c3=keras.layers.BatchNormalization()(c3)
    u4=keras.layers.UpSampling2D(size = (2,2))(c3)
    u4 = keras.layers.Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u4)
    c4 =keras.layers.concatenate([conv1,u4], axis = 3)
    c4 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c4)
    c4 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c4)
    out =keras.layers.Conv2D(1, 1, activation = 'sigmoid')(c4)
    model = keras.models.Model(inputs=inputs, outputs=out, name="SegNet")
    return model
def MYMODEL4(
        input_size = (896,896,4)):
    inputs = keras.layers.Input(input_size)
    
    conv1 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1=keras.layers.BatchNormalization()(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2=keras.layers.BatchNormalization()(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3=keras.layers.BatchNormalization()(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4=keras.layers.BatchNormalization()(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    
    u1=keras.layers.UpSampling2D(size = (2,2))(conv5)
    u1 = keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u1)
    c1 = keras.layers.concatenate([conv4,u1], axis = 3)
    c1 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c1)
    c1 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c1)
    c1=keras.layers.BatchNormalization()(c1)

    u2=keras.layers.UpSampling2D(size = (2,2))(c1)
    
    u2 = keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u2)
    c2 =keras.layers.concatenate([conv3,u2], axis = 3)
    c2 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c2)
    c2 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c2)
    c2=keras.layers.BatchNormalization()(c2)
    u3=keras.layers.UpSampling2D(size = (2,2))(c2)
    u3 = keras.layers.Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u3)
    c3 =keras.layers.concatenate([conv2,u3], axis = 3)
    c3 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c3)
    c3 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c3)
    c3=keras.layers.BatchNormalization()(c3)
    u4=keras.layers.UpSampling2D(size = (2,2))(c3)
    u4 = keras.layers.Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u4)
    c4 =keras.layers.concatenate([conv1,u4], axis = 3)
    c4 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c4)
    c4 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c4)
    out =keras.layers.Conv2D(1, 1, activation = 'sigmoid')(c4)
    model = keras.models.Model(inputs=inputs, outputs=out, name="SegNet")
    return model
