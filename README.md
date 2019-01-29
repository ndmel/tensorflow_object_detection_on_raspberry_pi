# Neural Netowrks on Raspberry Pi 3 B and Movidius

This readme is to describe process of launching pre-trained tensorflow models on Raspberry pi and using Movidius USB stick.

Needed harware:
* Raspberry Pi 3 B with installed Rasbian Stretch on a 20+ GB SD card
* Movidius USB stick
* USB extension cable
* Keyboard, mouse, web-cam

Software:
* Python3.5
* Tensorflow
* Movidius library

# Instructios to launch NN's on Raspberry Pi

To start things up, complete this tutorial https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi#2-install-tensorflow up to part 6 (detect objects) which will help to install tensorflow.

Go to 'object_detection' folder, then download desired mobilenet_v1 from here https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md. These are pre-trained models that can work even on pure Raspberry pi due to their lightweight architecture. 

```
cd /home/pi/tensorflow1/models/research/object_detection/
wget http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_192.tgz
mkdir mobilenet_v1_1.0_192
tar -xzvf mobilenet_v1_1.0_192.tgz -C mobilenet_v1_1.0_192
```

Download `Object_detection_mobilenet.py` from this repository, configure constants in script - MOD_F, MOD_RES, which are take from model name: `mobilenet_v1_MOD_F_MOD_RES (MOD_F='1.0', MOD_RES='192')`. Download labels file and put it in `data` directory. Make sure you have connected usb camera and launch the script.

```
python3 Object_detection_mobilenet.py --usbcam (should change script to process args)
```

# How to launch NN's using Movidius

Now that we have downloaded trained tensorflow models, we can convert them into Movidius format.

**NB** Only NN's with strict input size can be converted into Movidius format, for example [224, 224, 3], but no [?, 224, 3]. 

To convert models we will need:
* Neural network with strictly defined input size. Mobilenet is one of such networks, for this example input is 192x192 pixels.
* Frozen graph of a neural network. Mobilenet comes with compiled frozen graph, for your custom network follow instructions from official Movidius site.
* Input and Ouput nodes names. For mobilenet their are `input` and `MobilenetV1/Predictions/Reshape_1`. For custom neural network you can check their names using tensorboard tool to visiualize and search for nodes names. 
* Movidius tools, install them from here.

Now we can go to directory with mobilenet frozen graph and convert it to Movidius format.

```
cd mobilenet_v1_1.0_192/
mvNCCompile -s 12 mobilenet_v1_1.0_192_frozen.pb -in=input -on=MobilenetV1/Predictions/Reshape_1
cd ../
```

To launch object detection model on movidius, download `Object_detection_mobilenet_movidius.py` script, change the same parameters as before and run script.

```
python3 Object_detection_mobilenet_movidius.py --usbcam (should change script to process args)
```
