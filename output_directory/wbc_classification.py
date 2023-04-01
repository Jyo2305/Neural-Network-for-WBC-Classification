import tensorflow as tf
from tensorflow.keras import models,layers,datasets
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import normalize, to_categorical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from tensorflow.keras.models import Sequential 
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import math 
import datetime
import time
import sys
import itertools

def parameters():
	try:
		epochs = int(input("Enter No. of epochs: "))
		tbs = int(input("Enter Training set batch size: "))
		vbs = int(input("Enter Validation set batch size: "))
		verbose = int(input("Enter verbose(default = 1): "))
		if verbose < 0 or verbose > 2:
			verbose = 1
	except:
		print("Wrong parameters!!!")
		main_menu()
	main1(epochs, tbs, vbs, verbose)


def predict_image(prediction_folder, model, classes=0):
	current = os.getcwd()
	pred_list = []
	prediction_folder_content = os.listdir(prediction_folder)
	model_prediction = []
	for jpg in prediction_folder_content:
		pred_list.append(jpg)
	for jpg in prediction_folder_content:
		path = os.path.join(prediction_folder,jpg)
		img = image.load_img(path, target_size=(150,150))
		x = image.img_to_array(img)
		x/=255
		x = np.expand_dims(x,axis=0)
		images_list = np.vstack([x])
		pred_name = model.predict(images_list, batch_size=10)
		if classes==0:
			classification_count = len(pred_name[0])
			class_by_level = range(1,classification_count+1,1)
		else:
			class_by_level = classes
		j = 0
		
		for i in range(len(pred_name[0])):
			if pred_name[0][i]==max(pred_name[0]):
				model_prediction.append(class_by_level[i])
		
      	
	df1 = pd.DataFrame({'Image filename': pred_list, 'Predicted Class': model_prediction
				})
	x = input("Enter predicted result save filename: ")
	save_file_path = current + "/results_for_classification/" + x + ".csv"
	df1.to_csv(save_file_path)

def export_result(images, test_result_name, test_pred_name):
	
	df =	 pd.DataFrame(
    				{'Filename': images,
    				 'Actual Class': test_result_name,
     				 'Predicted Class': test_pred_name
    				})
	index = 1
	current = os.getcwd()
	x = input("Enter result csv filename: ")
	path = current + "/results_for_classification/" + x + ".csv"

	if not os.path.exists(path):
		df.to_csv(path)
	else:
		new_path = path.split(".")
		tmp = new_path[0] + "{}.format(index)"
		index += 1
		new_path = tmp + ".csv"
		df.to_csv(new_path)
		
def cm_plot(result_path, val_result_name, val_pred_name, classes, images=0):
	outputs = np.unique(val_result_name)
	df_confusion_matrix = confusion_matrix(val_result_name, val_pred_name, labels = outputs)
	
	title = result_path.split("/")
	title = title[-1]
	###To plot confusion matrix
	cmap = plt.cm.Blues

	plt.figure(figsize=(8,8))
	df_confusion_matrix = (df_confusion_matrix.astype('float')/df_confusion_matrix.sum(axis=1)[:,np.newaxis])*100
	plt.imshow(df_confusion_matrix, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(outputs))
	plt.xticks(tick_marks, outputs)
	plt.yticks(tick_marks, outputs)

	fmt = '.2f'
	thresh = df_confusion_matrix.max() / 2.
	for i, j in itertools.product(range(df_confusion_matrix.shape[0]), range(df_confusion_matrix.shape[1])):
    		plt.text(j, i, format(df_confusion_matrix[i, j], fmt),
             	horizontalalignment="center",
             	color="white" if df_confusion_matrix[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig(result_path)
	plt.show()
	if images == 0:
		sys.exit()
	else:
		export_result(images, val_result_name, val_pred_name)

def network(classification_count):
	### Model Building
	### Layers for neural network and Model Building
	l1 = Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3))
	l2 = MaxPooling2D(2,2)
	l3 = Conv2D(32,(3,3),activation='relu')
	l4 = MaxPooling2D(2,2)
	l5 = Conv2D(64,(3,3),activation='relu')
	l6 = MaxPooling2D(2,2)
	l7 = Conv2D(128,(3,3),activation='relu')
	l8 = GlobalAveragePooling2D()
	l9 = Flatten()
	l10 = Dropout(0.2)
	l11 = Dense(512,activation='relu')
	l12 = Dropout(0.2)
	l13 = Dense(classification_count,activation='softmax')
	model = tf.keras.models.Sequential([l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13])
	return model

def generator(train_dir, train_batch_size, val_batch_size):
	### Image Data Generator
	train_datagen = ImageDataGenerator(rescale = 1.0/255, height_shift_range = 0.2, width_shift_range = 0.2, rotation_range=45, 									validation_split=0.2) 
	#rotation_range, width_shift_range, height_shift_range, shear_range, horizontal_flip

	train_generator = train_datagen.flow_from_directory(train_dir,
                                                   	 batch_size=train_batch_size,
                                                    	class_mode='categorical',
                                                    	target_size=(150,150),
									
                                                    	subset='training'
                                                   	 )

	val_generator = train_datagen.flow_from_directory(train_dir,
                                                    	batch_size=val_batch_size,
                                                    	class_mode='categorical',
                                                    	target_size=(150,150),
									shuffle=False,
                                                    	subset='validation'
                                                   	 )
	
	return train_generator, val_generator

def test_generator(test_dir):
	test_datagen = ImageDataGenerator(rescale = 1.0/255)
	test = test_datagen.flow_from_directory(test_dir,
                                                    	batch_size=5,
                                                    	class_mode='categorical',
                                                    	target_size=(150,150),
									shuffle = False
                                                    	)
	return test

def model_training(train, val, epoch, verbose, val_size, val_batch_size, train_size, train_batch_size, model, cp_callback, checkpoint_path, best_checkpoint, tensorboard_callback):
	model.save_weights(checkpoint_path.format(epoch=0))

	history = model.fit(train,
                    	steps_per_epoch=int(train_size/train_batch_size),
                    	epochs=epoch,
				callbacks=[tensorboard_callback, best_checkpoint],
                    	validation_data=val,
                    	validation_steps=int(val_size/val_batch_size),
                    	verbose=verbose
                  	 )
	return history

def plot_model_accuracy(acc,val_acc, result_path):
	epoch = range(len(acc))
	plt.plot(epoch,acc,'r',label = 'training accuracy')
	plt.plot(epoch,val_acc,'b',label = 'validation accuracy')
	plt.title("Training and Validation Accuracy")
	plt.legend(loc = 0)
	save_file = os.path.join(result_path, "accuracy_plot.jpg")
	plt.savefig(save_file)
	plt.show()

def select(options_name):
	try:
		selection = int(input("Enter option by index: "))
	except:
		print("Invalid option!!!\n")
		select(options_name)
	if selection not in range(1,len(options_name)+1):
		print("Invalid selection!!!\n")
		select(options_name)
	else:
		return selection

def main2(loaded_model, val_result, val_pred, classes, result_path):
	
	options_name = ['Confusion Matrix of Validation Set', 'Confusion Matrix of Test Set', 'Predict Images']
	index = 1
	for option in options_name:
		print("\n{}: {}".format(index, option))
		index += 1
	
	for i in range(0,len(val_pred),1):
		for j in range(0,len(val_pred[i]),1):
			if val_pred[i][j]==max(val_pred[i]):
				val_pred[i][j] = 1
			else:
				val_pred[i][j] = 0
	
	

	val_result_name = []
	val_pred_name = []
	index = 0
	for i in range(0,len(val_pred),1):
		for j in range(0,len(val_pred[i]),1):
			if val_pred[i][j] == 1:
				val_pred_name.append(classes[j])

	for k in range(0,len(val_result),1):
		for l in range(0,len(val_result[k]),1):
			if val_result[k][l] == 1:
				val_result_name.append(classes[l])

	selection = select(options_name)
	if selection==1:
		val_plot_result = os.path.join(result_path, "validation_confusion_matrix")
		cm_plot(val_plot_result, val_result_name, val_pred_name, classes)
	elif selection == 2:
		print("Load Test data...\n")
		current = os.getcwd()
		directories = os.listdir(current)
		
		index=1
		for file in directories:
			print("{}: {}".format(index,file))
			index += 1
		print("\nOnly jpg, png, jpeg file extensions allowed")
		try:
			target = int(input("Enter target folder by index: "))
		except ValueError:
			print("\nInput value must be among the options provided!!!\n ")
			sys.exit()
		try:	
			target_folder = os.path.join(current, directories[target-1])
		except IndexError:
			print("\nInvalid Option!!!\n")
			sys.exit()
		index = 1
		try:
			classes = os.listdir(target_folder)	#Different image classes
		except NotADirectoryError:
			print("\nTarget is not a directory!!!\n")
			sys.exit()
		print("Listing directory:\n{}".format(target_folder))
		for file in classes:
			print("{}: {}".format(index, file))
			index += 1
		test_dir = target_folder
		test = test_generator(test_dir)
		classification_count = len(classes)
		## Results of Validation set
		test_result = np.array([])
		for i in range(0,len(test)):
			test_result = np.append(test_result, test[i][-1])
		test_result = test_result.reshape(int(len(test_result)/classification_count), classification_count)

		test_dir_content = os.listdir(test_dir)
		images = []
		for img in test_dir_content:
			image_folder = os.path.join(test_dir, img)
			image_folder_content = os.listdir(image_folder)
			for jpg in image_folder_content:
				images.append(jpg)

		test_pred = loaded_model.predict(test)
	
		for i in range(0,len(test_pred),1):
			for j in range(0,len(test_pred[i]),1):
				if test_pred[i][j]==max(test_pred[i]):
					test_pred[i][j] = 1
				else:
					test_pred[i][j] = 0

		test_result_name = []
		test_pred_name = []
		index = 1
		for i in range(0,len(test_pred),1):
			for j in range(0,len(test_pred[i]),1):
				if test_pred[i][j] == 1:
					test_pred_name.append(classes[j])

		for k in range(0,len(test_result),1):
			for l in range(0,len(test_result[k]),1):
				if test_result[k][l] == 1:
					test_result_name.append(classes[l])

		print("{}: {}\n{}: {}\n".format(index, "Confusion Matrix Plot", index+1, "Export results"))
		n = input("Enter option by Index: ")
		if n=='1':
			x = input("Enter result plot filename: ")
			test_plot_result = current + "/results_for_classification/" + x
			cm_plot(test_plot_result, test_result_name, test_pred_name, classes, images)
		elif n=='2':
			export_result(images, test_result_name, test_pred_name)
		else:
			print("Reload the model from model/output_model.h5")
			main_menu()
	
	elif selection==3:
		current = os.getcwd()
		contents = os.listdir(current)
		index = 1
		for i in contents:
			print("\n{}: {}".format(index, i))
			index += 1
		try:
			s = int(input("Enter option by index: "))
		except:
			print("Invalid option!!!\n")
			print("Reload model from model/output_model.h5 or your model!!!")
			sys.exit()
		try:
			prediction_folder = os.path.join(current, contents[s-1])
		except:
			print("Invalid option!!!\n")
			print("Reload model from model/output_model.h5 or your model!!!")
			sys.exit()
		
		predict_image(prediction_folder, loaded_model, classes)
        		
		
	sys.exit()

def main1(epochs, tbs, vbs, verb=1):
	###Training parameters
	train_batch_size = tbs
	val_batch_size = vbs
	epoch=epochs
	verbose = verb

	current = os.getcwd()
	directories = os.listdir(current)
	index=1
	for file in directories:
		print("{}: {}".format(index,file))
		index += 1
	print("\nOnly jpg, png, jpeg file extensions allowed")
	try:
		target = int(input("Enter target folder by index: "))
	except ValueError:
		print("\nInput value must be among the options provided!!!\n ")
		main1()
	try:	
		target_folder = os.path.join(current, directories[target-1])
	except IndexError:
		print("\nInvalid Option!!!\n")
		main1()
	index = 1
	try:
		classes = os.listdir(target_folder)	#Different image classes
	except NotADirectoryError:
		print("\nTarget is not a directory!!!\n")
		main1()
	print("Listing directory:\n{}".format(target_folder))
	for file in classes:
		print("{}: {}".format(index, file))
		index += 1
	classification_count = len(classes)
	model = network(classification_count)

	### Model Compiling
	model.compile(optimizer=RMSprop(learning_rate=0.001),
              	loss='categorical_crossentropy',
              	metrics=['accuracy'])

	train, val = generator(target_folder, train_batch_size, val_batch_size)
	print("\n")
	sum=0
	for folders in os.listdir(target_folder):
		path = os.path.join(target_folder, folders)
		sum += len(os.listdir(path))
	val_size = int(0.2*sum)
	train_size = int(sum-val_size)

	## Results of Validation set
	val_result = np.array([])
	for i in range(0,len(val)):
		val_result = np.append(val_result, val[i][-1])
	val_result = val_result.reshape(int(len(val_result)/classification_count), classification_count)

	model_dir = os.path.join(current,"model")
	checkpoint_dir = os.path.join(model_dir, "checkpoint")

	### Checkpoint creation
	checkpoint_path = os.path.join(current,"model/checkpoint/cp-{epoch:04d}.ckpt")
	best_checkpoint_dir = os.path.join(checkpoint_dir, "best_checkpoint")
	best_checkpoint_path = os.path.join(best_checkpoint_dir) 
	checkpoint_dir = os.path.dirname(checkpoint_path)
	if not os.path.exists(model_dir):
		os.mkdir(model_dir)
	if not os.path.exists(checkpoint_dir):
		os.mkdir(checkpoint_dir)
	

	cp_callback = tf.keras.callbacks.ModelCheckpoint(
    				filepath=checkpoint_path, 
    				verbose=1, 
    				save_weights_only=True)

	best_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=best_checkpoint_dir,
                                           		 	save_best_only=True)
	log = os.path.join(model_dir, "logs")
	log_dir = log + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	if not os.path.exists(log):
		os.mkdir(log)
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

	history = model_training(train, val, epoch, verbose, val_size, val_batch_size, train_size, train_batch_size, model, cp_callback, 	checkpoint_path, best_checkpoint, tensorboard_callback)
	
	
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	result_path = os.path.join(current, "results_for_classification")
	if not os.path.exists(result_path):
		os.mkdir(result_path)
	plot_model_accuracy(acc, val_acc, result_path)
	
	### Loading best model
	model.save(model_dir+"/output_model.h5")
	tf.saved_model.save(model, model_dir+"/")
	loaded_model = tf.keras.models.load_model(model_dir+"/output_model.h5")
	
	
	val_pred = model.predict(val)
	
	main2(loaded_model, val_result, val_pred, classes, result_path)

def main3(self_loaded_model):
	print(self_loaded_model.summary())
	print("Load Test data...\n")
	current = os.getcwd()
	directories = os.listdir(current)
	
	index=1
	for file in directories:
		print("{}: {}".format(index,file))
		index += 1
	print("\nOnly jpg, png, jpeg file extensions allowed")
	try:
		target = int(input("Enter target folder by index: "))
	except ValueError:
		print("\nInput value must be among the options provided!!!\n ")
		sys.exit()
	try:	
		target_folder = os.path.join(current, directories[target-1])
	except IndexError:
		print("\nInvalid Option!!!\n")
		sys.exit()
	index = 1
	try:
		classes = os.listdir(target_folder)	#Different image classes
	except NotADirectoryError:
		print("\nTarget is not a directory!!!\n")
		sys.exit()
	print("Listing directory:\n{}".format(target_folder))
	for file in classes:
		print("{}: {}".format(index, file))
		index += 1
	test_dir = target_folder
	test = test_generator(test_dir)
	
	test_dir_content = os.listdir(test_dir)
	images = []
	for img in test_dir_content:
		image_folder = os.path.join(test_dir, img)
		image_folder_content = os.listdir(image_folder)
		for jpg in image_folder_content:
			images.append(jpg)
	
	classification_count = len(classes)
	## Results of Validation set
	test_result = np.array([])

	for i in range(0,len(test)):
		test_result = np.append(test_result, test[i][-1])
	test_result = test_result.reshape(int(len(test_result)/classification_count), classification_count)
	
	test_pred = self_loaded_model.predict(test)
	
	for i in range(0,len(test_pred),1):
		for j in range(0,len(test_pred[i]),1):
			if test_pred[i][j]==max(test_pred[i]):
				test_pred[i][j] = 1
			else:
				test_pred[i][j] = 0

	test_result_name = []
	test_pred_name = []
	index = 1
	for i in range(0,len(test_pred),1):
		for j in range(0,len(test_pred[i]),1):
			if test_pred[i][j] == 1:
				test_pred_name.append(classes[j])

	for k in range(0,len(test_result),1):
		for l in range(0,len(test_result[k]),1):
			if test_result[k][l] == 1:
				test_result_name.append(classes[l])

	
	print(len(images), len(test_result_name), len(test_pred_name))
	
	print("{}: {}\n{}: {}\n".format(index, "Confusion Matrix Plot", index+1, "Export results"))
	n = input("Enter option by Index: ")
	if n=='1':
		x = input("Enter result plot filename: ")
		test_plot_result = current + "/results_for_classification/" + x
		cm_plot(test_plot_result, test_result_name, test_pred_name, classes, images)
	elif n=='2':
		export_result(images, test_result_name, test_pred_name)
	else:
		print("Reload the model from model/output_model.h5")
		main_menu()
	sys.exit()

def main_menu():
	a = 1
	b = 2
	c = 3
	current = os.getcwd()
	model_dir = current + "/model"
	
	result_dir = current + "/results_for_classification"
	if not os.path.exists(model_dir):
		os.mkdir(model_dir)
	if not os.path.exists(result_dir):
		os.mkdir(result_dir)

	print("\n{}: {}\n{}: {}\n{}: {}\n".format(a,'Train Simple model', b, 'Load and Test Model', c, "Load Model and Predict"))
	try:
		n = int(input("\nEnter option by index: "))
	except ValueError:
		print("Invalid Option!!!")
		main_menu()

	if n==a or n==b or n==c:
		if n==a:
			parameters()
		elif n==b:
			print("Must be a trained model saved as .h5 type file")
			
			index=1
			current_dir_contents = os.listdir(current)
			for folder in current_dir_contents:
				print("\n{}: {}".format(index, folder))
				index += 1
			target = int(input("Enter option by index: "))
			try:
				target_model_dir = os.path.join(current, current_dir_contents[target-1])
			except (NotADirectoryError, ValueError) as error:
				print("Invalid Option!!!")
				main_menu()
			index = 1
			target_model_dir_content = os.listdir(target_model_dir)
			
			for file in target_model_dir_content:
				print("\n{}: {}".format(index, file))
				index += 1
			target_file = int(input("Enter option by index: "))
			try:
				target_model_file = os.path.join(target_model_dir, target_model_dir_content[target_file-1])
			except:
				print("Invalid file!!!")
				main_menu()

			print("Loading Model....")
			try:
				self_loaded_model = tf.keras.models.load_model(target_model_file)
			except:
				print("\nSome error occured!!!")
				main_menu()
			print("Loading Completed !!!")
			main3(self_loaded_model)
		
		elif n==c:
			print("Must be a trained model saved as .h5 type file")
			
			index=1
			current_dir_contents = os.listdir(current)
			for folder in current_dir_contents:
				print("\n{}: {}".format(index, folder))
				index += 1
			target = int(input("Enter option by index: "))
			try:
				target_model_dir = os.path.join(current, current_dir_contents[target-1])
			except (NotADirectoryError, ValueError) as error:
				print("Invalid Option!!!")
				main_menu()
			index = 1
			target_model_dir_content = os.listdir(target_model_dir)
			
			for file in target_model_dir_content:
				print("\n{}: {}".format(index, file))
				index += 1
			target_file = int(input("Enter option by index: "))
			try:
				target_model_file = os.path.join(target_model_dir, target_model_dir_content[target_file-1])
			except:
				print("Invalid file!!!")
				main_menu()

			print("Loading Model....")
			try:
				self_loaded_model = tf.keras.models.load_model(target_model_file)
			except:
				print("\nSome error occured!!!")
				main_menu()
			print("Loading Completed!!!\n")
			
			print("Select prediction images directory by index: ")
			contents = os.listdir(current)
			index = 1
			for i in contents:
				print("\n{}: {}".format(index, i))
				index += 1
			try:
				s = int(input("Enter option by index: "))
			except:
				print("Invalid option!!!\n")
				print("Reload model from model/output_model.h5 or your model!!!")
				sys.exit()
			try:
				prediction_folder = os.path.join(current, contents[s-1])
			except:
				print("Invalid option!!!\n")
				print("Reload model from model/output_model.h5 or your model!!!")
				sys.exit()

			predict_image(prediction_folder, self_loaded_model)		
			
	else:
		print("Invalid Option!!!")
		main_menu()
	
	

main_menu()




















