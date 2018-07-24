//
//  MultiClassObjectDetector.cpp
//  pr2_perception
//
//  Created by Xun Wang on 12/05/16.
//  Copyright (c) 2016 Xun Wang. All rights reserved.
//

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/bind.hpp>
#include <boost/timer.hpp>
#include <boost/format.hpp>
#include <boost/ref.hpp>
#include <boost/foreach.hpp>
#include <boost/thread/thread.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <time.h>
#include <sstream>
#include <fstream>

#include "MultiClassObjectDetector.h"

#include "dn_object_detect/DetectedObjects.h"
#include "dn_object_detect/AnnotatedView.h"

namespace uts_perp {

using namespace std;
using namespace cv;

static const int kPublishFreq = 10; // darknet can work reasonably around 5FPS
static const string kDefaultDevice = "/wide_stereo/right/image_rect_color";
static const string kYOLOModel = "data/yolo.weights";
static const string kYOLOConfig = "data/yolo.cfg";
static const string kClassNamesConfig = "data/voc.names";

static const char * VoClassNames[] = { "aeroplane", "bicycle", "bird",
                              "boat", "bottle", "bus", "car",
                              "cat", "chair", "cow", "diningtable",
                              "dog", "horse", "motorbike",
                              "person", "pottedplant", "sheep",
                              "sofa", "train", "tvmonitor"
                            };

static int NofVoClasses = sizeof( VoClassNames ) / sizeof( VoClassNames[0] );

static inline long timediff_usec( timespec start, timespec end )
{
  if ((end.tv_nsec - start.tv_nsec) < 0) {
    return (end.tv_sec - start.tv_sec - 1) * 1E6 + (1E9 + end.tv_nsec - start.tv_nsec) / 1E3;
  }
  else {
    return (end.tv_sec - start.tv_sec) * 1E6 + (end.tv_nsec - start.tv_nsec) / 1E3;
  }
}

MultiClassObjectDetector::MultiClassObjectDetector() :
  imgTrans_( priImgNode_ ),
  initialised_( false ),
  doDetection_( false ),
  debugRequests_( 0 ),
  avRequests_( 0 ),
  srvRequests_( 0 ),
  procThread_( NULL ),
  object_detect_thread_( NULL )
{
  priImgNode_.setCallbackQueue( &imgQueue_ );
}

MultiClassObjectDetector::~MultiClassObjectDetector()
{
}

void MultiClassObjectDetector::init()
{
  NodeHandle priNh( "~" );
  std::string yoloModelFile;
  std::string yoloConfigFile;
  std::string classNamesFile;

  priNh.param<std::string>( "camera", cameraDevice_, kDefaultDevice );
  priNh.param<std::string>( "yolo_model", yoloModelFile, kYOLOModel );
  priNh.param<std::string>( "yolo_config", yoloConfigFile, kYOLOConfig );
  priNh.param<std::string>( "class_names", classNamesFile, kClassNamesConfig );
  priNh.param( "threshold", threshold_, 0.2f );

  const boost::filesystem::path modelFilePath = yoloModelFile;
  const boost::filesystem::path configFilePath = yoloConfigFile;

  if (boost::filesystem::exists( modelFilePath ) && boost::filesystem::exists( configFilePath )) {
    darkNet_ = load_network( (char*)yoloConfigFile.c_str(), (char*)yoloModelFile.c_str(), 0 );
    detectLayer_ = darkNet_->layers[darkNet_->n-1];
    printf( "detect layer (layer %d) w = %d h = %d n = %d\n", darkNet_->n, detectLayer_.w, detectLayer_.h, detectLayer_.n );
    set_batch_network( darkNet_, 1 );
    srand(2222222);
  }
  else {
    ROS_ERROR( "Unable to find YOLO darknet configuration or model files." );
    return;
  }

  if (!(detectLayer_.type == DETECTION || detectLayer_.type == REGION || detectLayer_.type == YOLO)) {
    ROS_ERROR( "Invalid YOLO darknet configuration." );
    return;
  }

  this->initClassLabels( classNamesFile );

  ROS_INFO( "Loaded detection model data." );

  procThread_ = new AsyncSpinner( 1, &imgQueue_ );
  procThread_->start();

  initialised_ = true;

  dtcPub_ = priImgNode_.advertise<dn_object_detect::DetectedObjects>( "/dn_object_detect/detected_objects", 1,
      boost::bind( &MultiClassObjectDetector::startDetection, this ),
      boost::bind( &MultiClassObjectDetector::stopDetection, this) );

  avPub_ = priImgNode_.advertise<dn_object_detect::AnnotatedView>( "/dn_object_detect/annotated_view", 1,
      boost::bind( &MultiClassObjectDetector::startAnnotatedView, this ),
      boost::bind( &MultiClassObjectDetector::stopAnnotatedView, this) );

  imgPub_ = imgTrans_.advertise( "/dn_object_detect/debug_view", 1,
      boost::bind( &MultiClassObjectDetector::startDebugView, this ),
      boost::bind( &MultiClassObjectDetector::stopDebugView, this) );
}

void MultiClassObjectDetector::fini()
{
  srvRequests_ = 1; // reset requests
  this->stopDetection();

  if (procThread_) {
    delete procThread_;
    procThread_ = NULL;
  }
  free_network( darkNet_ );
  initialised_ = false;
}

void MultiClassObjectDetector::continueProcessing()
{
  ros::spin();
}

void MultiClassObjectDetector::doObjectDetection()
{
  //ros::Rate publish_rate( kPublishFreq );
  ros::Time ts;

  float nms = 0.4;

  timespec time1, time2;
  DetectedList detectObjs;
  detectObjs.reserve( 30 ); // silly hardcode

  long interval = long( 1.0 / double( kPublishFreq ) * 1E6);
  long proctime = 0;

  float * orig_input = darkNet_->input;
  float * orig_input_gpu = darkNet_->input_gpu;

  while (doDetection_) {
    clock_gettime( CLOCK_MONOTONIC, &time1 );
    {
      boost::mutex::scoped_lock lock( mutex_ );
      if (imgMsgPtr_.get() == NULL) {
        imageCon_.wait( lock );
        continue;
      }
      try {
        cv_ptr_ = cv_bridge::toCvCopy( imgMsgPtr_, sensor_msgs::image_encodings::RGB8 );
        ts = imgMsgPtr_->header.stamp;
      }
      catch (cv_bridge::Exception & e) {
        ROS_ERROR( "Unable to convert image message to mat." );
        imgMsgPtr_.reset();
        continue;
      }
      imgMsgPtr_.reset();
    }

    if (cv_ptr_.get()) {
      IplImage img = cv_ptr_->image;
      image im = ipl_to_image( &img );
      image sized = resize_image( im, darkNet_->w, darkNet_->h );
      float *X = sized.data;
      float *predictions = network_predict( darkNet_, X );
      //printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
      //convert_yolo_detections( predictions, detectLayer_.classes, detectLayer_.n, detectLayer_.sqrt,
          //detectLayer_.side, 1, 1, threshold_, probs, boxes, 0);
      int nboxes = 0;
      detection * dets = get_network_boxes( darkNet_, im.w, im.h, threshold_, 0.5, 0, 1, &nboxes);

      if (nms) {
        do_nms_sort( dets, nboxes, detectLayer_.classes, nms );
      }

      this->consolidateDetectedObjects( &im, dets, nboxes, detectObjs );
      //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, 0, 20);
      free_detections( dets, nboxes );
      free_image(im);
      free_image(sized);

      darkNet_->input = orig_input;
      darkNet_->input_gpu = orig_input_gpu;

      this->publishDetectedObjects( detectObjs );
      if (debugRequests_ > 0)
        this->drawDebug( detectObjs );
      if (avRequests_ > 0)
        this->publishAnnotatedView( detectObjs );
    }
    cv_ptr_.reset();

    clock_gettime( CLOCK_MONOTONIC, &time2 );
    proctime = timediff_usec( time1, time2 );
    //printf( "detect process time %li usec\n",  proctime);
    if (interval > proctime)
      usleep( interval - proctime );
  }
}

void MultiClassObjectDetector::processingRawImages( const sensor_msgs::ImageConstPtr& msg )
{
  // assume we cannot control the framerate (i.e. default 30FPS)
  boost::mutex::scoped_lock lock( mutex_, boost::try_to_lock );

  if (lock) {
    imgMsgPtr_ = msg;
    imageCon_.notify_one();
  }
}

void MultiClassObjectDetector::startDebugView()
{
  if (debugRequests_ == 0)
    this->startDetection();

  debugRequests_++;
}

void MultiClassObjectDetector::stopDebugView()
{
  debugRequests_--;
  if (debugRequests_ <= 0)
    this->stopDetection();

}

void MultiClassObjectDetector::startAnnotatedView()
{
  if (avRequests_ == 0)
    this->startDetection();

  avRequests_++;
}

void MultiClassObjectDetector::stopAnnotatedView()
{
  avRequests_--;
  if (avRequests_ <= 0)
    this->stopDetection();

}

void MultiClassObjectDetector::startDetection()
{
  if (!initialised_) {
    ROS_ERROR( "Detector is not initialised correctly!\n" );
    return;
  }
  srvRequests_ ++;
  if (srvRequests_ >= 1)
    return;

  doDetection_ = true;
  cv_ptr_.reset();
  imgMsgPtr_.reset();

  image_transport::TransportHints hints( "compressed" );
  imgSub_ = imgTrans_.subscribe( cameraDevice_, 1,
                                  &MultiClassObjectDetector::processingRawImages, this, hints );

  object_detect_thread_ = new boost::thread( &MultiClassObjectDetector::doObjectDetection, this );

  ROS_INFO( "Starting multi-class object detection service." );
}

void MultiClassObjectDetector::stopDetection()
{
  srvRequests_--;
  if (srvRequests_ > 0)
    return;

  doDetection_ = false;

  if (object_detect_thread_) {
    object_detect_thread_->join();
    delete object_detect_thread_;
    object_detect_thread_ = NULL;

    imgSub_.shutdown();

    ROS_INFO( "Stopping multi-class object detection service." );
  }
}

void MultiClassObjectDetector::publishDetectedObjects( const DetectedList & objs )
{
  dn_object_detect::DetectedObjects tObjMsg;
  tObjMsg.header = cv_ptr_->header;

  tObjMsg.objects.resize( objs.size() );

  for (size_t i = 0; i < objs.size(); i++) {
    tObjMsg.objects[i] = objs[i];
  }

  dtcPub_.publish( tObjMsg );
}

void MultiClassObjectDetector::publishAnnotatedView( const DetectedList & objs )
{
  dn_object_detect::AnnotatedView avMsg;
  avMsg.image = *(cv_ptr_->toImageMsg());
  avMsg.objects.resize( objs.size() );

  for (size_t i = 0; i < objs.size(); i++) {
    avMsg.objects[i] = objs[i];
  }

  avPub_.publish( avMsg );
}

void MultiClassObjectDetector::drawDebug( const DetectedList & objs )
{
  cv::Scalar boundColour( 255, 0, 255 );
  cv::Scalar connColour( 209, 47, 27 );

  for (size_t i = 0; i < objs.size(); i++) {
    dn_object_detect::ObjectInfo obj = objs[i];
    cv::rectangle( cv_ptr_->image, cv::Rect(obj.tl_x, obj.tl_y, obj.width, obj.height),
        boundColour, 2 );

    // only write text on the head or body if no head is detected.
    std::string box_text = format( "%s prob=%.2f", obj.type.c_str(), obj.prob );
    // Calculate the position for annotated text (make sure we don't
    // put illegal values in there):
    cv::Point2i txpos( std::max(obj.tl_x - 10, 0),
                      std::max(obj.tl_y - 10, 0) );
    // And now put it into the image:
    putText( cv_ptr_->image, box_text, txpos, FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
  }
  imgPub_.publish( cv_ptr_->toImageMsg() );
}

void MultiClassObjectDetector::consolidateDetectedObjects( const image * im, detection * dets,
     int numofDetects, DetectedList & objList )
{
  //printf( "max_nofb %d, NofVoClasses %d\n", max_nofb, NofVoClasses );
  int objclass = 0;
  float prob = 0.0;

  objList.clear();

  for(int i = 0; i < numofDetects; ++i){
    int objclass = -1;
    float objmaxprob = 0.0;
    std::string objlabels;
    for(int j = 0; j < nofClasses_; ++j){
      if (dets[i].prob[j] > threshold_){
        if (dets[i].prob[j] > objmaxprob) {
          objmaxprob = dets[i].prob[j];
        }
        if (objclass < 0) {
          objlabels = classLabels_[j];
          objclass = j;
        }
        else {
          objlabels += "," + classLabels_[j];
        }
      }
    }
    if(objclass >= 0){
      dn_object_detect::ObjectInfo newObj;
      newObj.type = objlabels;
      newObj.prob = objmaxprob;

      box b = dets[i].bbox;
      //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

      int left  = (b.x - b.w/2.) * im->w;
      int right = (b.x + b.w/2.) * im->w;
      int top   = (b.y - b.h/2.) * im->h;
      int bot   = (b.y + b.h/2.) * im->h;

      if (right > im->w-1)  right = im->w-1;
      if (bot > im->h-1)    bot = im->h-1;

      newObj.tl_x = left < 0 ? 0 : left;
      newObj.tl_y = top < 0 ? 0 : top;
      newObj.width = right - newObj.tl_x;
      newObj.height = bot - newObj.tl_y;
      objList.push_back( newObj );
    }
  }
}

void MultiClassObjectDetector::initClassLabels( const std::string & filename )
{
  const boost::filesystem::path labelFilePath = filename;
  if (boost::filesystem::exists( labelFilePath )) {
    ifstream ifs( filename.c_str(), std::ifstream::in );
    stringstream sstr;
    string label;
    sstr << ifs.rdbuf();
    ifs.close();
    while(std::getline( sstr, label ) ) {
      classLabels_.push_back( label );
      //printf( "class label %s\n", label.c_str() );
    }
    ROS_INFO( "Loaded class names from %s.\n", filename.c_str() );
  }
  if (classLabels_.size() == 0) {
    std::vector<std::string> data( VoClassNames, VoClassNames + NofVoClasses );
    classLabels_ = data;
    ROS_INFO( "Loaded default VOC class name list." );
  }
  nofClasses_ = (int)classLabels_.size();
}

} // namespace uts_perp
