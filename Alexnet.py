import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,Conv3D,MaxPool2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from datetime import datetime
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator

#The path of the stored model
alex_model_relu_path = os.path.join(os.getcwd(),'saved_models','alex_model_relu.h5')
alex_model_tanh_path = os.path.join(os.getcwd(),'saved_models','alex_model_tanh.h5')
alex_model_sigmoid_path = os.path.join(os.getcwd(),'saved_models','alex_model_sigmoid.h5')
alex_model_adagrad_path = os.path.join(os.getcwd(),'saved_models','alex_model_adagrad.h5')

epochs = 75
batch_size = 32
learningRate = 0.01
data_augmentation = False

def plot_loss_2(history, history2):
    
    plt.subplot(2,1,1)
    plt.plot(history.history['val_accuracy'])
    plt.plot(history2.history['val_accuracy'])    

    plt.xticks(np.arange(0,epochs, (epochs/10)))
    plt.title('Validation Set Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Q1', 'Q2'], loc=0)
    
    plt.subplot(2,1,2)
    plt.plot(history.history['val_loss'])
    plt.plot(history2.history['val_loss'])

    
    plt.xticks(np.arange(0,epochs, (epochs/10) ))
    plt.title('Validation Set Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Q1', 'Q2'], loc=0)
    
    plt.subplot(2,1,3)
    plt.plot(history.history['accuracy'])
    plt.plot(history2.history['accuracy'])
    
    plt.xticks(np.arange(0,epochs, (epochs/10) ))
    plt.title('Training Set accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Q1', 'Q2'], loc=0)
    
    plt.subplot(2,1,4)
    plt.plot(history.history['loss'])
    plt.plot(history2.history['loss'])

    
    plt.xticks(np.arange(0,epochs, (epochs/10) ))
    plt.title('Training Set loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Q1', 'Q2'], loc=0)
    
    plt.show()

def plot_loss_3(history, history2, history3):
    
    plt.subplot(2,2,1)
    plt.plot(history.history['val_accuracy'])
    plt.plot(history2.history['val_accuracy'])
    plt.plot(history3.history['val_accuracy'])

    plt.xticks(np.arange(0,epochs, (epochs/10)))
    plt.title('Validation Set Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Q1', 'Q2', 'Q3'], loc=0)
    
    plt.subplot(2,2,2)
    plt.plot(history.history['val_loss'])
    plt.plot(history2.history['val_loss'])
    plt.plot(history3.history['val_loss'])

    
    plt.xticks(np.arange(0,epochs, (epochs/10) ))
    plt.title('Validation Set Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Q1', 'Q2', 'Q3'], loc=0)
    
    plt.subplot(2,2,3)
    plt.plot(history.history['accuracy'])
    plt.plot(history2.history['accuracy'])
    plt.plot(history3.history['accuracy'])
    
    plt.xticks(np.arange(0,epochs, (epochs/10) ))
    plt.title('Training Set accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Q1', 'Q2', 'Q3'], loc=0)
    
    plt.subplot(2,2,4)
    plt.plot(history.history['loss'])
    plt.plot(history2.history['loss'])
    plt.plot(history3.history['loss'])
    
    plt.xticks(np.arange(0,epochs, (epochs/10) ))
    plt.title('Training Set loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Q1', 'Q2', 'Q3'], loc=0)
    
    plt.show()


#RGB
def train_model(x_train,y_train,x_test,y_test,x_validation,y_validation,model_name, optimizer='sgd',activation='relu', data_aug = False,epochs = 100, batch_size=32):
	if optimizer == 'Adagrad':
		learningRate = 0.01
		opt = Adagrad(lr=learningRate)
	elif optimizer == 'adam':
		learningRate = 0.01
		opt = Adam(lr=learningRate)
	else:
		learningRate = 0.01
		opt = SGD(lr=learningRate)
	
	alex_model = Sequential()
	
	#Layer 1 
	alex_model.add( Conv2D(48, kernel_size=(3,3),strides=(1,1), activation=activation, padding='same', input_shape=x_train.shape[1:] ) )
	alex_model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )
	
	#Layer 2
	alex_model.add( Conv2D(96, kernel_size=(3,3), activation=activation, padding='same') )
	alex_model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )
	
	#Layer 3
	alex_model.add( Conv2D(192, kernel_size=(3,3), activation=activation, padding='same') )
	
	#Layer 4
	alex_model.add( Conv2D(192, kernel_size=(3,3), activation=activation, padding='same') )
	alex_model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )
	
	#Layer 5
	alex_model.add( Conv2D(256, kernel_size=(3,3), activation=activation, padding='same') )
	alex_model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )
	
	alex_model.add(Flatten())
	
	#Layer 6
	alex_model.add(Dense(512, activation=activation))
	
	#Layer 7 
	alex_model.add(Dense(256, activation=activation))
	
	#Prediction
	alex_model.add(Dense(10, activation='softmax'))
	
	
	alex_model.compile(loss=tf.keras.losses.categorical_crossentropy,
		      optimizer=opt,
		      metrics=['accuracy'])
	
	#log_to_csv = CSVLogger('alex_relu.csv', append=True, separator=';')
	logdir = model_name+"_logs_" + datetime.now().strftime("%Y%m%d-%H%M%S")
	tensorboard_callback = TensorBoard(log_dir=logdir)
	if not data_aug:
		print('Not using data augmentation.')
		alex_model_hist = alex_model.fit(x_train, y_train,
								  epochs=epochs,
								  batch_size= batch_size,
								  validation_data=(x_validation, y_validation),
								  callbacks=[tensorboard_callback])
	else:
		print('Using real-time data augmentation.')
		# This will do preprocessing and realtime data augmentation:
		datagen = ImageDataGenerator(
		        featurewise_center=False,  # set input mean to 0 over the dataset
		        samplewise_center=False,  # set each sample mean to 0
		        featurewise_std_normalization=False,  # divide inputs by std of the dataset
		        samplewise_std_normalization=False,  # divide each input by its std
		        zca_whitening=False,  # apply ZCA whitening
		        zca_epsilon=1e-06,  # epsilon for ZCA whitening
		        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
		        # randomly shift images horizontally (fraction of total width)
		        width_shift_range=0.1,
		        # randomly shift images vertically (fraction of total height)
		        height_shift_range=0.1,
		        shear_range=0.,  # set range for random shear
		        zoom_range=0.,  # set range for random zoom
		        channel_shift_range=0.,  # set range for random channel shifts
		        # set mode for filling points outside the input boundaries
		        fill_mode='nearest',
		        cval=0.,  # value used for fill_mode = "constant"
		        horizontal_flip=True,  # randomly flip images
		        vertical_flip=False,  # randomly flip images
		        # set rescaling factor (applied before any other transformation)
		        rescale=None,
		        # set function that will be applied on each input
		        preprocessing_function=None,
		        # image data format, either "channels_first" or "channels_last"
		        data_format=None,
		        # fraction of images reserved for validation (strictly between 0 and 1)
		        validation_split=0.0)

	    # Compute quantities required for feature-wise normalization
		# (std, mean, and principal components if ZCA whitening is applied).
		datagen.fit(x_train)
		

	    # Fit the model on the batches generated by datagen.flow().
		alex_model_hist = alex_model.fit_generator(datagen.flow(x_train, y_train,
									  batch_size=batch_size),epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4, callbacks=[tensorboard_callback])
		
	
	alex_model_path = os.path.join(os.getcwd(),'saved_models',model_name+'.h5')
	tf.keras.models.save_model(
		alex_model,
		alex_model_path,
		overwrite=True,
		include_optimizer=True
		)
	return alex_model_hist 
	
if __name__ =='__main__':
	cifar = tf.keras.datasets.cifar10
	(x_train, y_train), (x_test, y_test) = cifar.load_data()
	
	x_train, x_test = x_train / 255.0, x_test / 255.0 
	#To make the data between 0~1
	y_train=to_categorical(y_train,num_classes=10)
	y_test=to_categorical(y_test,num_classes=10)
	
	x_train, x_validation = x_train[0:40000],x_train[40000:50000]
	y_train, y_validation = y_train[0:40000],y_train[40000:50000]
	
	print('Shape of Training samples tensor: ',x_train.shape)
	print('Shape of Training Labels: ',y_train.shape)
	print('Shape of Validation samples: ',x_validation.shape)
	print('Shape of Validation labels: ',y_validation.shape)
	print('Shape of Test samples: ',x_test.shape)
	print('Shape of Test labels: ',y_test.shape)
	alex_model_relu_hist = train_model(x_train,y_train,x_test,y_test,x_validation,y_validation,'alex_model_relu', optimizer='sgd',activation='relu',epochs=epochs)
	alex_model_tanh_hist = train_model(x_train,y_train,x_test,y_test,x_validation,y_validation,'alex_model_tanh', optimizer='sgd',activation='tanh',epochs=epochs)
	alex_model_sigmoid_hist = train_model(x_train,y_train,x_test,y_test,x_validation,y_validation,'alex_model_sigmoid', optimizer='sgd',activation='sigmoid') 
	alex_model_adagrad_hist = train_model(x_train,y_train,x_test,y_test,x_validation,y_validation,'alex_model_adagrad', optimizer='adagrad',activation='relu',epochs=epochs)
	alex_model_adam_hist = train_model(x_train,y_train,x_test,y_test,x_validation,y_validation,'alex_model_adam', optimizer='adam',activation='relu',epochs=epochs)
	with open ('results.txt','w+') as fp:
		for model in [alex_model_relu_hist,alex_model_tanh_hist,alex_model_adagrad_hist,alex_model_adam_hist]:
			fp.write(model.history['loss'])
			fp.write(model.history['accuracy'])
			fp.write(model.history['val_loss'])
			fp.write(model.history['val_accuracy'])
	plot_loss_2(alex_model_relu_hist, alex_model_tanh_hist)
	plot_loss_3(alex_model_relu_hist,alex_model_adagrad_hist, alex_model_adam_hist)

'''
 to use pretrained models:
#try:
# 	model  = tf.keras.models.load_model(
# 	    model_path,
# 	    custom_objects=None,
# 	    compile=True
# 	)
# except Exception as e:

	
	
	# finally:
	# 	score = model.evaluate(x_test, y_test)
	
	# 	print('Test loss:', score[0])
	# 	print('Test accuracy:', score[1])
	# 	predict_classes = model.predict_classes(x_test)
	
		# plt.figure(figsize=(12,9))
		# for j in xrange(0,3):
		# 	for i in xrange(0,4):
		# 		plt.subplot(4,3,j*4+i+1)
		# 		plt.title('predict:{}/real:{}'.format(predict_classes[j*4+i] ,y_test_label[j*4+i]))
		# 		plt.axis('off')
		# 		plt.imshow(x_test[j*4+i].reshape(32,32,3),cmap=plt.cm.binary)   
	
		# plt.show()
'''
	
	
