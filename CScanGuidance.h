//
//  CScanGuidance.hpp
//  SfM_desktop
//
//  Created by Marco on 2017/05/29.
//  Copyright © 2017 IBM. All rights reserved.
//

#ifndef CScanGuidance_hpp
#define CScanGuidance_hpp

#include <stdio.h>
#include "CVUtils/GeometryUtils.hpp"
#include <opencv2/opencv.hpp>

#endif /* CScanGuidance_hpp */

using namespace std;
using namespace cv;

class CScanGuidance {
    
public:
    CScanGuidance();
    ~CScanGuidance();
    
    void updateCentroid(const vector<Matx31d> &pts3d);
    void calculateMask(const Mat &frame, const vector<Matx31d> &pts3d, const Matx33d &K, const Matx34d &P);
    
    Matx31d &getCentroid();
    RotatedRect &getBoundingBox();
    
private:
    
    void findMinMaxPts(const vector<Matx31d> &pts3d, Matx31d &ptMin, Matx31d &ptMax);
    
    
    //centroid
    Matx31d _centroid;
    
    //histogram
    Mat _hsvFrame;
    Mat _frame_float;
    MatND _hist;
    const int _channels[2] = {0,1};
    float _hrange[2] = {0,360};
    float _srange[2] = {0,1};
    const float* _ranges[2] = {_hrange,_srange};
    int _histSize[2];
    Mat _bProj;
    double _threshold;
    double _alpha;
    int _frameCount;
    
    //segmentation
    Mat _mask;
    vector<Point2i> _chull;
    RotatedRect _bbox;
    
    //scaling factor for processing wrt incoming frame size
    double _scale;
};
