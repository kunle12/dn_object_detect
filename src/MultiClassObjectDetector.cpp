//
//  MultiClassObjectDetector.cpp
//  pr2_perception
//
//  Created by Xun Wang on 12/05/16.
//  Copyright (c) 2016 Xun Wang. All rights reserved.
//

#include <sensor_msgs/image_encodings.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/bind.hpp>
#include <boost/timer.hpp>
#include <boost/format.hpp>
#include <boost/ref.hpp>
#include <boost/gil/gil_all.hpp>
#include <boost/foreach.hpp>
#include <boost/thread/thread.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include "MultiClassObjectDetector.h"

#include "dn_object_detect/DetectedObjects.h"

namespace uts_perp {

using namespace std;
using namespace cv;
  
static const int kPublishFreq = 10;
static const string kDefaultDevice = "/wide_stereo/right/image_rect_color";
static const string kYOLOModel = "data/yolo.model";
static const string kYOLOConfig = "data/yolo_config.ini";

MultiClassObjectDetector::MultiClassObjectDetector() :
  imgTrans_( priImgNode_ ),
  doDetection_( false ),
  stoppingBDThread_( false ),
  showDebug_( false ),
  srvRequests_( 0 ),
  procThread_( NULL )
{
  dtcPub_ = priImgNode_.advertise<dn_object_detect::DetectedObjects>( "/dn_object_detect/detected_objects", 1,
      boost::bind( &MultiClassObjectDetector::startDetection, this ),
      boost::bind( &MultiClassObjectDetector::stopDetection, this) );

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
  
  priNh.param<std::string>( "camera", cameraDevice_, kDefaultDevice );
  priNh.param<std::string>( "yolo_model", yoloModelFile, kYOLOModel );
  priNh.param<std::string>( "yolo_config", yoloConfigFile, kYOLOConfig );
  
  const boost::filesystem::path modelFilePath = yoloModelFile;
  const boost::filesystem::path configFilepath = yoloConfigFile;
  
  if (boost::filesystem::exists( modelFilePath ) && boost::filesystem::exists( yoloConfigFile )) {
    darkNet_ = parse_net_work_cfg( yoloConfigFile.c_str() );
    load_weights( &darkNet_, yoloModelFile.c_str() );
    detectLayer_ = darkNet_.layers[darkNet_.n-1];
    set_batch_darkNet_work( &darkNet_, 1 );
    srand(2222222);
  }
  else {
    ROS_ERROR( "Unable to find YOLO darknet configuration or model files." );
  }

  ROS_INFO( "Loaded detection model data." );
  
  procThread_ = new AsyncSpinner( 1, &imgQueue_ );
  procThread_->start();

  if (showDebug_) {
    imgPub_ = imgTrans_.advertise( "/dn_object_detect/debug_view", 1 );
  }
}

void MultiClassObjectDetector::fini()
{
  this->stopDetection();
  imgSub_.shutdown();

  if (procThread_) {
    delete procThread_;
    procThread_ = NULL;
  }
}

void MultiClassObjectDetector::continueProcessing()
{
  ros::spin();
}
  
void MultiClassObjectDetector::doObjectDetection()
{
  ros::Rate publish_rate( kPublishFreq );
  ros::Time ts;

  float nms=.5;

  box * boxes = calloc( detectLayer_.side * detectLayer_.side * detectLayer_.n, sizeof( box ) );
  float **probs = calloc( detectLayer_.side * detectLayer_.side * detectLayer_.n, sizeof(float *));
  for(int j = 0; j < detectLayer_.side * detectLayer_.side * detectLayer_.n; ++j) {
    probs[j] = calloc( detectLayer_.classes, sizeof(float *) );
  }

  while (1) {
    stoppingBDThread_ = !doDetection_;
    
    if (!stoppingScanThread_ || !stoppingBDThread_) { // make sure we don't go into wait state if scan thread has already quitted.
      data_preprocess_barrier_->wait();
    }
    if (!stoppingScanThread_ || !stoppingBDThread_) {
      boost::mutex::scoped_lock lock( mutex_ );
      if (imgMsgPtr_.get() == NULL) {
        publish_rate.sleep();
        stoppingBDThread_ = !doDetection_;
        data_postprocess_barrier_->wait();
        continue;
      }
      try {
        cv_ptr_ = cv_bridge::toCvCopy( imgMsgPtr_, sensor_msgs::image_encodings::BGR8 );
        ts = imgMsgPtr_->header.stamp;
      }
      catch (cv_bridge::Exception & e) {
        ROS_ERROR( "Unable to convert image message to mat." );
        imgMsgPtr_.reset();
        publish_rate.sleep();
        stoppingBDThread_ = !doDetection_;
        data_postprocess_barrier_->wait();
        continue;
      }
      imgMsgPtr_.reset();
      stoppingBDThread_ = !doDetection_;
      data_postprocess_barrier_->wait();
    }

    if (!stoppingFDThread_ || !stoppingBDThread_) {
      preprocess_barrier_->wait();
    }

    image im = load_image_color( input, 0, 0 );
    image sized = resize_image( im, darkNet_.w, darkNet_.h );
    float *X = sized.data;
    float *predictions = darkNet_work_predict( darkNet_, X );
    //printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
    convert_yolo_detections( predictions, detectLayer_.classes, detectLayer_.n, detectLayer_.sqrt,
        detectLayer_.side, 1, 1, thresh, probs, boxes, 0);
    if (nms) {
      do_nms_sort( boxes, probs, detectLayer_.side * detectLayer_.side * detectLayer_.n,
          detectLayer_.classes, nms );
    }
    //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
    //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, 0, 20);
    free_image(im);
    free_image(sized);

    if ((!stoppingFDThread_ || !stoppingBDThread_) && cv_ptr_.get()) {
      vector<ObjectWindow> results;
      Mat gray;
      cv::cvtColor( cv_ptr_->image, gray, CV_BGR2GRAY );
      //cv::blur( gray, gray, cv::Size(3, 3) );
      IplImage img = gray;
      detection( &img, results );
      for (size_t i = 0; i < results.size(); i++) {
        hmBodies_.push_back( cv::Rect( results.at(i).x0, results.at(i).y0,
           results.at(i).width, results.at(i).height ) );
      }
    }
    stoppingBDThread_ = !doDetection_;
    postprocess_barrier_->wait();

    cv_ptr_.reset();

    publish_rate.sleep();
  }

  // clean up
  for(int j = 0; j < detectLayer_.side * detectLayer_.side * detectLayer_.n; ++j) {
    free( probs[j] );
  }
  free( probs );
  free( boxes );
}

void MultiClassObjectDetector::doFaceDetection()
{
  cv::Mat faces_downloaded, tmpData;
  cv::gpu::GpuMat imgData, grayData, facesbuf;
  int detections = 0;

  while (1) {
    stoppingFDThread_ = !doDetection_;
    preprocess_barrier_->wait();
    
    if (stoppingBDThread_ && stoppingFDThread_)
      break;
    
    hmFaces_.clear();

    //cv::blur( cv_ptr_->image, tmpData, cv::Size( 4, 4 ) );
    imgData.upload( cv_ptr_->image );
    cv::gpu::cvtColor( imgData, grayData, CV_BGR2GRAY );
    cv::gpu::equalizeHist( grayData, grayData );
    detections = faceDetector_.detectMultiScale( grayData, facesbuf, 1.2, 4, 
       cv::Size(30, 30) );
    facesbuf.colRange(0, detections).download( faces_downloaded );
    for (int i = 0; i < detections; ++i) {
      hmFaces_.push_back( faces_downloaded.ptr<cv::Rect>()[i] );
    }
    stoppingFDThread_ = !doDetection_;
    postprocess_barrier_->wait();
    if (stoppingBDThread_ && stoppingFDThread_)
      break;
  }
}
  
void MultiClassObjectDetector::processingRawImages( const sensor_msgs::ImageConstPtr& msg )
{
  // assume we cannot control the framerate (i.e. default 30FPS)
  boost::mutex::scoped_lock lock( mutex_ );

  imgMsgPtr_ = msg;
}
  
void MultiClassObjectDetector::startDetection()
{
  srvRequests_ ++;
  if (srvRequests_ == 1) {
    ROS_INFO( "Start human detection and tracking service..." );
  }
  else {
    return;
  }

  doDetection_ = true;
  stoppingBDThread_ = stoppingFDThread_ = stoppingScanThread_ = false;
  cv_ptr_.reset();
  imgMsgPtr_.reset();

  imgSub_ = imgTrans_.subscribe( cameraDevice_, 1,
                                  &MultiClassObjectDetector::processingRawImages, this );


  preprocess_barrier_ = new boost::barrier( 2 );
  postprocess_barrier_ = new boost::barrier( 2 );
  data_preprocess_barrier_ = new boost::barrier( 2 );
  data_postprocess_barrier_ = new boost::barrier( 2 );
  
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
  }

  delete preprocess_barrier_;
  delete postprocess_barrier_;
  delete data_preprocess_barrier_;
  delete data_postprocess_barrier_;

  imgSub_.shutdown();

  ROS_INFO( "Stopping multi-class object detection service." );
}

void MultiClassObjectDetector::onIdentifiedObject( const TrackingObject & obj )
{
  dn_object_detect::TrackedObjectStatusChange tObjMsg;
  tObjMsg.header = cv_ptr_->header;
  tObjMsg.objtype = obj.objectType;
  tObjMsg.trackid = obj.trackID;
  tObjMsg.nameid = obj.recognisedID;
  tObjMsg.status = REC_OBJ;
  
  dtcPub_.publish( tObjMsg );
}

void MultiClassObjectDetector::drawDebug( std::vector<TrackingObject> & tObjs )
{
  cv::Scalar boundColour( 255, 0, 255 );
  cv::Scalar connColour( 209, 47, 27 );

  for (size_t i = 0; i < tObjs.size(); i++) {
    TrackingObject & obj = tObjs.at( i );
    cv::rectangle( cv_ptr_->image, obj.objectBound, boundColour, 2 );
    if (obj.dependent) {
      cv::line( cv_ptr_->image, obj.centre(), obj.dependent->centre(), connColour, 2 );
    }

    // only write text on the head or body if no head is detected.
    std::string box_text = format("ID = %d RecID = %d pos = (%.2f,%.2f,%.2f)", obj.trackID, obj.recognisedID,
                                  obj.est3DCoord.x, obj.est3DCoord.y, obj.est3DCoord.z );
    // Calculate the position for annotated text (make sure we don't
    // put illegal values in there):
    cv::Point2i txpos( std::max(obj.objectBound.x - 10, 0),
                      std::max(obj.objectBound.y - 10, 0) );
    // And now put it into the image:
    putText( cv_ptr_->image, box_text, txpos, FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
  }
  imgPub_.publish( cv_ptr_->toImageMsg() );
}

} // namespace uts_perp
