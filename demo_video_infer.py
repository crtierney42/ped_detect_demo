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
MODEL_JOB_DIR='/home/ubuntu/digits/digits/jobs/20160905-143028-2f08'
MODEL_JOB_DIR='data/'
# Set the data job directory from DIGITS here
DATA_JOB_DIR='/home/ubuntu/digits/digits/jobs/20160905-135347-01d5'
DATA_JOB_DIR='data/'

last_iteration='70800'
#last_iteration='19140'

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
classifier = caffe.Net(os.path.join(MODEL_JOB_DIR,'deploy.prototxt'), 
                       os.path.join(MODEL_JOB_DIR,'snapshot_iter_' + last_iteration + '.caffemodel'),
                       caffe.TEST)

# Instantiate a Caffe Transformer object that wil preprocess test images before inference
transformer = caffe.io.Transformer({'data': classifier.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
#transformer.set_mean('data',mean.mean(1).mean(1)/255)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

BATCH_SIZE, CHANNELS, HEIGHT, WIDTH = classifier.blobs['data'].data[...].shape

print 'The input size for the network is: (' + \
        str(BATCH_SIZE), str(CHANNELS), str(HEIGHT), str(WIDTH) + \
         ') (batch size, channels, height, width)'

# Create opencv video object
vid = cv2.VideoCapture(pedfn)

# We will just use every n-th frame from the video
every_nth = 1
frame_num = 0
DetectObject = 1

cval=1
pause=0
while(vid.isOpened()):
    if pause != 1:
        ret, frame = vid.read()
        frame_num += 1

    if frame_num%every_nth == 0:
        frame = cv2.resize(frame, (1024, 512), 0, 0)

        if DetectObject == 1:             
             # Use the Caffe transformer to preprocess the frame
            data = transformer.preprocess('data', frame.astype('float16')/255)
        
            # Set the preprocessed frame to be the Caffe model's data layer
            classifier.blobs['data'].data[...] = data
        
            # Measure inference time for the feed-forward operation
            start = time.time()
            # The output of DetectNet is an array of bounding box predictions
            bounding_boxes = classifier.forward()['bbox-list'][0]
            end = (time.time() - start)*1000

            if cval:
                # Convert the image from OpenCV BGR format to matplotlib RGB format for display
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           
                # Create a copy of the image for drawing bounding boxes
                overlay = frame.copy()
            
                # Loop over the bounding box predictions and draw a rectangle for each bounding box
                for bbox in bounding_boxes:
                    if  bbox.sum() > 0:
                        cv2.rectangle(overlay, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (255, 0, 0), -1)
                    
                # Overlay the bounding box image on the original image
                frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            else:
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                
                gframe=np.copy(frame)
                gframe[:,:,0]=gray
                gframe[:,:,1]=gray
                gframe[:,:,2]=gray

                # Loop over the bounding boxes and copy the colored section into the overlay
                for bbox in bounding_boxes:
                    if  bbox.sum() > 0:
			b0=int(bbox[0])
			b1=int(bbox[1])
			b2=int(bbox[2])
			b3=int(bbox[3])
                        gframe[b1:b3,b0:b2,:]=frame[b1:b3,b0:b2,:]
                frame=gframe
       
            itxt="Inference time: %dms per frame" % end 
            # Display the inference time per frame
            cv2.putText(frame,itxt,
                        (10,500), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        cv2.imshow('frame',frame)
        fn="outdata/outfile.%05d.jpg" % frame_num;
        cv2.imwrite(fn,frame)

        # Now check the keypresses to do something different
        v=cv2.waitKey(1)
        if (v & 0xFF) == ord('q'):
            break
        if (v & 0xFF) == ord('c'):
            cval=abs(cval-1)
        if (v & 0xFF) == ord('d'):
            DetectObject=~DetectObject
        if (v & 0xFF) == ord('p'):
            pause=abs(pause-1)

vid.release()
cv2.destroyAllWindows()


