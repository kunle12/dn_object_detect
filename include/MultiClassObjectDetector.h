//
//  MultiClassObjectDetector.h
//  pr2_perception
//
//  Created by Xun Wang on 12/05/16.
//  Copyright (c) 2016 Xun Wang. All rights reserved.
//

#ifndef __pr2_perception__MultiClassObjectDetector__
#define __pr2_perception__MultiClassObjectDetector__

#include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/barrier.hpp>

#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <darknet/darknet.h>

#include "dn_object_detect/ObjectInfo.h"

using namespace std;
using namespace ros;

namespace uts_perp {

typedef std::vector<dn_object_detect::ObjectInfo> DetectedList;

class MultiClassObjectDetector
{
public:
  MultiClassObjectDetector();
  virtual ~MultiClassObjectDetector();

  void init();
  void fini();

  void continueProcessing();

private:
  NodeHandle priImgNode_;
  image_transport::ImageTransport imgTrans_;
  image_transport::Publisher imgPub_;
  image_transport::Subscriber imgSub_;

  Publisher dtcPub_;
  Publisher avPub_;

  bool doDetection_;
  bool initialised_;
  int debugRequests_;
  int srvRequests_;
  int avRequests_;

  float threshold_;

  boost::mutex mutex_;
  boost::condition_variable imageCon_;

  boost::thread * object_detect_thread_;

  sensor_msgs::ImageConstPtr imgMsgPtr_;

  std::string cameraDevice_;

  CallbackQueue imgQueue_;

  AsyncSpinner * procThread_;

  cv_bridge::CvImagePtr cv_ptr_;

  std::vector<std::string> classLabels_;
  int nofClasses_;

  network * darkNet_;
  layer detectLayer_;

  void processingRawImages( const sensor_msgs::ImageConstPtr& msg );

  void startDetection();
  void stopDetection();

  void startDebugView();
  void stopDebugView();

  void startAnnotatedView();
  void stopAnnotatedView();

  void doObjectDetection();

  void consolidateDetectedObjects( const cv::Mat & im, detection * dets,
       int numofDetects, DetectedList & objList );
  void publishDetectedObjects( const DetectedList & objs );
  void publishAnnotatedView( const DetectedList & objs );
  void drawDebug( const DetectedList & objs );
  void initClassLabels( const std::string & filename );
};

} // namespace uts_perp

#endif /* defined(__pr2_perception__MultiClassObjectDetector__) */
