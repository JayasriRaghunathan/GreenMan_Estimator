import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import sys

from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/traffic_Novena3.avi',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0
    
    #initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        list_file = open('detection.txt', 'w')
        frame_index = -1 
    
    fps = 0.0
    count = 0
    frame_count=0
    dict = {}
    previous_dict = {}
    speed_dict = {}
    tracked_objects = []
    speed_dict={}
    first_frame_hgt = {}
    while True:
        _, img = vid.read()
        print("The length of a tracked objects "+str(len(tracked_objects)))
        frame_count = frame_count+1
        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        #print("Before the length of the classes is "+str(len(classes)))
        classes = classes[0]
        #print("the class name is "+str(classes[0]))
        #print("after the length of the classes is "+str(len(classes)))
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)    
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]
        
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        print("printing the detected class name "+str(classes[0]))
        detections = [detections[i] for i in indices if classes[i] == 'person'] 
        

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        previous_dict = dict
        dict={} 
        pt_distance=None
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            if class_name != 'person':
                continue 
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            #cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            dict[(class_name+str(track.track_id))]= (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            
            if not (class_name+str(track.track_id)) in tracked_objects:
                tracked_objects.append(class_name+str(track.track_id))
                speed_dict.update({(class_name+str(track.track_id)) : []})
                first_frame_hgt[class_name+str(track.track_id)]= int((bbox[3]-bbox[1]))
                print("the height calculated is "+str(int((bbox[3]-bbox[1]))))
                #class_name+str(track.track_id)+str("_speed") =[]
                
            if (class_name+str(track.track_id)) in dict.keys():
                if (class_name+str(track.track_id)) in previous_dict.keys():
                    a,b,c,d = dict[(class_name+str(track.track_id))]
                    current_fr_ctr = a+1/2*(c-a),b+1/2*(d-b)
                    a1,b1,c1,d1 = previous_dict[(class_name+str(track.track_id))]
                    prv_fr_ctr = a1+1/2*(c1-a1),b1+1/2*(d1-b1)
                    #pt_distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(dict[(class_name+str(track.track_id))], previous_dict[(class_name+str(track.track_id))])]))
                    
                    # calulating the distance between the bounding box centers of 2 adjacent frames
                    pt_distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(current_fr_ctr, prv_fr_ctr)]))
                    frame_rate = 12
                    image_aspect_ratio=18
                    current_frame_height = int((bbox[3]-bbox[1]))
                    
                    #calculating the speed, adjusting for the patron's distance from camera and metre vs pixel aspect ratio
                    cur_fr_spd  = pt_distance*frame_rate*(first_frame_hgt[class_name+str(track.track_id)])/current_frame_height/image_aspect_ratio
                    print(pt_distance)
                    print("the speed calculated is "+str(cur_fr_spd))
                    speed_dict[(class_name+str(track.track_id))].append(pt_distance)
                    
                    #np.sqrt((a-a1)**2+(b-b1)**2+(c-c1)+(d-d1)
                    #(class_name+str(track.track_id)+str("_speed")).append((a-a1)+(b-b1)+(c-c1)+(d-d1))
            print(class_name+str(track.track_id))
            print(dict[(class_name+str(track.track_id))])
            print("speed dict length is "+ str(len(speed_dict[(class_name+str(track.track_id))])))
            if pt_distance is not None:
                cv2.putText(img, class_name + "-" + str(track.track_id) + "-"+'%.2f' %cur_fr_spd,(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            else:
                cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
        print("the dict size is "+" for the frame "+ str(frame_count)+" "+str(len(dict)))
        print("the dict size is "+" for the previous  frame "+ str(frame_count)+" "+str(len(previous_dict)))  

        
        ### UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN
        #for det in detections:  
        #    bbox = det.to_tlbr() 
        #    cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
        
        # print fps on screen 
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.imshow('output', img)
        if FLAGS.output:
            out.write(img)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(converted_boxes) != 0:
                for i in range(0,len(converted_boxes)):
                    list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
            list_file.write('\n')

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break
    vid.release()
    if FLAGS.output:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
