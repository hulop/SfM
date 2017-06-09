//
//  CScanGuidance.cpp
//  SfM_desktop
//
//  Created by Marco on 2017/05/29.
//  Copyright © 2017 IBM. All rights reserved.
//

#include "CScanGuidance.h"

CScanGuidance::CScanGuidance() {

    _scale = 0.25;
    _histSize[0] = 60;
    _histSize[1] = 50;
    _threshold = 0.01;
    _alpha = 0.9;
    _frameCount = 0;
    _bbox = RotatedRect(Point2f(0,0), Size(0,0), 0);

}

CScanGuidance::~CScanGuidance() {
    
    
}

void CScanGuidance::updateCentroid(const vector<Matx31d> &pts3d) {
    //update centroid with latest point cloud
    _centroid.val[0] = 0;
    _centroid.val[1] = 0;
    _centroid.val[2] = 0;
    for (int i = 0; i < pts3d.size(); i++) {
        _centroid += pts3d[i];
    }
    _centroid *= (1.0/double(pts3d.size()));
}

void CScanGuidance::calculateMask(const Mat &frame, const vector<Matx31d> &pts3d, const Matx33d &K, const Matx34d &P) {
    
    //update centroid
    updateCentroid(pts3d);
    
    //project points in 2d in current frame
    vector<Point2i> pts2d;
    pts2d.reserve(pts3d.size());
    GeometryUtils::projectPoints(P, K, pts3d, pts2d, frame.size());
    Mat pts2dM = Mat(pts2d,false);
    
    //resize image frame
    Mat frame_small;
    resize(frame,frame_small,Size(frame.cols*_scale,frame.rows*_scale));
    
    //reset mask for current frame
    if (_mask.empty()) {
        _mask = Mat(frame_small.size(),CV_8UC1);
        _hsvFrame = Mat(frame_small.size(), CV_32FC3);
        _frame_float = Mat(frame_small.size(), CV_32FC3);
        _chull.reserve(frame_small.rows*frame_small.cols*_scale*_scale);
        _bProj = Mat(frame_small.size(),CV_32FC1);
    }
    _mask.setTo(0);
    _bProj.setTo(0);
    

    //find convex hull of projected points (if object is convex it is guaranteed to contain only the object)
    pts2dM *= _scale;
    convexHull(pts2dM, _chull);
    fillConvexPoly(_mask, _chull, Scalar(1));
    double area = contourArea(_chull);

    //H-S histogram
    MatND hist;
    frame_small.convertTo(_frame_float, CV_32FC3);
    cvtColor(_frame_float*1.0/255.0,_hsvFrame,CV_BGR2HSV);
    calcHist(&_hsvFrame, 1, _channels, _mask, hist, 2, _histSize, _ranges);
    //combine histograms
    if (_frameCount == 0)
        _hist = hist;
    else {
        _hist = _alpha*hist + (1-_alpha)*_hist;
        
    }
    
    calcBackProject(&_hsvFrame, 1, _channels, _hist, _bProj, _ranges);
    
    //threshold backprojection, store points in new convex hull
    _chull.clear();
    for (int i = 0; i < _bProj.rows; i++) {
        float *ptr = _bProj.ptr<float>(i);
        for (int j = 0; j < _bProj.cols; j++) {
            if (ptr[j]/area > _threshold) {
                _chull.push_back(Point2i(j,i));
            }
        }
    }
    
    //find oriented rectangle containing the object
    _bbox = minAreaRect(_chull);
    
    //undo the scaling so that bounding box is at full scale
    _bbox.size.height *= 1.0/_scale;
    _bbox.size.width *= 1.0/_scale;
    _bbox.center *= 1.0/_scale;
}

Matx31d &CScanGuidance::getCentroid() {
    return _centroid;
}


RotatedRect &CScanGuidance::getBoundingBox() {
    return _bbox;
}

void CScanGuidance::findMinMaxPts(const vector<Matx31d> &pts3d, Matx31d &ptMin, Matx31d &ptMax) {
    ptMin = Matx31d(DBL_MAX, DBL_MAX, DBL_MAX);
    ptMax = Matx31d(-DBL_MAX,-DBL_MAX,-DBL_MAX);
    for (int i = 0; i < pts3d.size(); i++) {
        ptMax.val[0] = (pts3d[i].val[0] > ptMax.val[0]) ? pts3d[i].val[0] : ptMax.val[0];
        ptMax.val[1] = (pts3d[i].val[1] > ptMax.val[1]) ? pts3d[i].val[1] : ptMax.val[1];
        ptMax.val[2] = (pts3d[i].val[2] > ptMax.val[2]) ? pts3d[i].val[2] : ptMax.val[2];
        ptMin.val[0] = (pts3d[i].val[0] < ptMin.val[0]) ? pts3d[i].val[0] : ptMin.val[0];
        ptMin.val[1] = (pts3d[i].val[1] < ptMin.val[1]) ? pts3d[i].val[1] : ptMin.val[1];
        ptMin.val[2] = (pts3d[i].val[2] < ptMin.val[2]) ? pts3d[i].val[2] : ptMin.val[2];
    }
    
}
