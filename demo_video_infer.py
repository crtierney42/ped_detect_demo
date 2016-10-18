#!/usr/bin/env python

import time
import os
import sys
import cv2
import caffe
import numpy as np

if (len(sys.argv) < 2):
    print "demo_video_infer.py <mov>"
    quit()


pedfn=sys.argv[1]
print "File: ",pedfn




# Configure Caffe to use the GPU for inference
caffe.set_mode_gpu()

# Set the model job directory from DIGITS here
PED_DIR='data/'
CARS_DIR='data_cars/'
# Set the data job directory from DIGITS here
PED_DIR='data/'
CARS_DIR='data_cars/'

last_iteration_ped='70800'
last_iteration_cars='49200'

width_ped=512
height_ped=1024
width_cars=384
height_cars=1248
width_out=512
height_out=1024
width_out=512
height_out=1024

# Load the dataset mean image file
#mean = np.load('data/mean.binaryproto')

b = caffe.proto.caffe_pb2.BlobProto()
data = open('data/mean.binaryproto','rb').read()
b.ParseFromString(data)
mean = np.array(caffe.io.blobproto_to_array(b))[0]

mean=np.swapaxes(mean,0,2)
mean=np.swapaxes(mean,0,1)


# Instantiate a Caffe model in GPU memory
# The model architecture is defined in the deploy.prototxt file
# The pretrained model weights are contained in the snapshot_iter_<number>.caffemodel file
classifier_ped = caffe.Net(os.path.join(PED_DIR,'deploy.prototxt'), 
                       os.path.join(PED_DIR,'snapshot_iter_' + last_iteration_ped + '.caffemodel'),
                       caffe.TEST)

# Instantiate a Caffe Transformer object that wil preprocess test images before inference
transformer_ped = caffe.io.Transformer({'data': classifier_ped.blobs['data'].data.shape})
transformer_ped.set_transpose('data', (2,0,1))
#transformer.set_mean('data',mean.mean(1).mean(1)/255)
transformer_ped.set_raw_scale('data', 255)
transformer_ped.set_channel_swap('data', (2,1,0))
BATCH_SIZE_PED, CHANNELS_PED, HEIGHT_PED, WIDTH_PED = classifier_ped.blobs['data'].data[...].shape

# Instantiate a Caffe model in GPU memory
# The model architecture is defined in the deploy.prototxt file
# The pretrained model weights are contained in the snapshot_iter_<number>.caffemodel file
classifier_cars = caffe.Net(os.path.join(CARS_DIR,'deploy.prototxt'), 
                       os.path.join(CARS_DIR,'snapshot_iter_' + last_iteration_cars + '.caffemodel'),
                       caffe.TEST)

# Instantiate a Caffe Transformer object that wil preprocess test images before inference
transformer_cars = caffe.io.Transformer({'data': classifier_cars.blobs['data'].data.shape})
transformer_cars.set_transpose('data', (2,0,1))
#transformer.set_mean('data',mean.mean(1).mean(1)/255)
transformer_cars.set_raw_scale('data', 255)
transformer_cars.set_channel_swap('data', (2,1,0))
BATCH_SIZE_CARS, CHANNELS_CARS, HEIGHT_CARS, WIDTH_CARS = classifier_cars.blobs['data'].data[...].shape


print 'The input size for the PEDESTRIAN network is: (' + \
        str(BATCH_SIZE_PED), str(CHANNELS_PED), str(HEIGHT_PED), str(WIDTH_PED) + \
         ') (batch size, channels, height, width)'

print 'The input size for the CARS network is: (' + \
        str(BATCH_SIZE_CARS), str(CHANNELS_CARS), str(HEIGHT_CARS), str(WIDTH_CARS) + \
         ') (batch size, channels, height, width)'


while 1:
    # Create opencv video object
    vid = cv2.VideoCapture(pedfn)

    # We will just use every n-th frame from the video
    every_nth = 1
    frame_num = 0

    itxt=""

    label_rate = 6
    cval=1
    pause=0
    DetectObject=1
    DetectPed=1
    DetectCars=1
    while(vid.isOpened()):
        if pause != 1:
            ret, frame = vid.read()
            frame_num += 1
    
        if frame_num%every_nth == 0:
            frame = cv2.resize(frame, (height_out, width_out), 0, 0)
    
            if DetectObject == 1:             
                # Measure inference time for the feed-forward operation
                start = time.time()

                if DetectPed:
                    ### Pedestrian Object Detect
                     # Use the Caffe transformer to preprocess the frame
                    data_ped = transformer_ped.preprocess('data', frame.astype('float16')/255)
                    # Set the preprocessed frame to be the Caffe model's data layer
                    classifier_ped.blobs['data'].data[...] = data_ped
                    # Measure inference time for the feed-forward operation
                    # The output of DetectNet is an array of bounding box predictions
                    bounding_boxes_ped = classifier_ped.forward()['bbox-list'][0]

                if DetectCars:
     		    ### Car Object Detect
                    # Use the Caffe transformer to preprocess the frame
                    data_cars = transformer_cars.preprocess('data', frame.astype('float16')/255)
                    # Set the preprocessed frame to be the Caffe model's data layer
                    classifier_cars.blobs['data'].data[...] = data_cars
                    # The output of DetectNet is an array of bounding box predictions
                    bounding_boxes_cars = classifier_cars.forward()['bbox-list'][0]

                end = (time.time() - start)*1000

    
                if cval:
                    # Convert the image from OpenCV BGR format to matplotlib RGB format for display
                    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
               
                    # Create a copy of the image for drawing bounding boxes
                    overlay = frame.copy()
               
                    # Loop over the bounding box predictions and draw a rectangle for each bounding box
                    if DetectPed:
                        for bbox in bounding_boxes_ped:
                            if  bbox.sum() > 0:
        			    # Scale the bboxes to the output size
                                b0=np.int(bbox[0]/height_ped*height_out)
                                b1=np.int(bbox[1]/width_ped*width_out)
                                b2=np.int(bbox[2]/height_ped*height_out)
                                b3=np.int(bbox[3]/width_ped*width_out)
                                cv2.rectangle(overlay, (b0,b1), (b2,b3), (255, 0, 0), -1)
    
                    if DetectCars:
                        for bbox in bounding_boxes_cars:
                            if  bbox.sum() > 0:
    			    # Scale the bboxes to the output size
                                b0=np.int(bbox[0]/height_cars*height_out)
                                b1=np.int(bbox[1]/width_cars*width_out)
                                b2=np.int(bbox[2]/height_cars*height_out)
                                b3=np.int(bbox[3]/width_cars*width_out)
                                cv2.rectangle(overlay, (b0,b1), (b2,b3), (0, 255, 0), -1)
                        
                    # Overlay the bounding box image on the original image
                    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                else:
                    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    # scale up to make it more white
                    gray=127+np.rint(gray/2)
                    
                    gframe=np.copy(frame)
                    gframe[:,:,0]=gray
                    gframe[:,:,1]=gray
                    gframe[:,:,2]=gray

                    if DetectPed:
                        # Loop over the bounding boxes and copy the colored section into the overlay
                        for bbox in bounding_boxes_ped:
                            if  bbox.sum() > 0:
	    		    # Scale the bboxes to the output size
                                b0=np.int(bbox[0]/height_ped*height_out)
                                b1=np.int(bbox[1]/width_ped*width_out)
                                b2=np.int(bbox[2]/height_ped*height_out)
                                b3=np.int(bbox[3]/width_ped*width_out)
                                gframe[b1:b3,b0:b2,:]=frame[b1:b3,b0:b2,:]


                    if DetectCars:
                        for bbox in bounding_boxes_cars:
                            if  bbox.sum() > 0:
    			    # Scale the bboxes to the output size
                                b0=np.int(bbox[0]/height_cars*height_out)
                                b1=np.int(bbox[1]/width_cars*width_out)
                                b2=np.int(bbox[2]/height_cars*height_out)
                                b3=np.int(bbox[3]/width_cars*width_out)
                                gframe[b1:b3,b0:b2,:]=frame[b1:b3,b0:b2,:]

                    frame=gframe
      
                if (frame_num+1) % label_rate == 0:  
                    itxt="Inference time: %dms per frame" % end 
                
                # Display the inference time per frame
                cv2.putText(frame,itxt,
                            (10,500), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    
            cv2.imshow('frame',frame)
#            fn="outdata/outfile.%05d.jpg" % frame_num;
#            cv2.imwrite(fn,frame)

            # Now check the keypresses to do something different
            v=cv2.waitKey(1)
            if (v & 0xFF) == ord('q'):
                break
            if (v & 0xFF) == ord('g'):
                cval=abs(cval-1)
            if (v & 0xFF) == ord('d'):
                DetectObject=abs(DetectObject-1)
            if (v & 0xFF) == ord('p'):
                DetectPed=abs(DetectPed-1)
		print "New ped value: ",DetectPed
            if (v & 0xFF) == ord('c'):
                DetectCars=abs(DetectCars-1)	
		print "New car value: ",DetectCars
            if (v & 0xFF) == ord('s'):
                pause=abs(pause-1)
            if frame_num == vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
		quit()
                frame_num = 0
                vid = cv2.VideoCapture(pedfn)

    vid.release()


cv2.destroyAllWindows()


