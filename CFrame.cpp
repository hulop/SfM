/*******************************************************************************
 * Copyright (c) 2017  IBM Corporation, Carnegie Mellon University and others
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *******************************************************************************/

#include "CFrame.h"

CFrame::CFrame() {
 
    //default flann tree index parameters
  //  _params = flann::LinearIndexParams();
    _nMatched = 0;
}

CFrame::CFrame(Mat frameIn) {
    //convert to grayscale and save
    _frame = frameIn;
    cvtColor(frameIn, _frameGrey, CV_RGB2GRAY);
    _nMatched = 0;
}

CFrame::CFrame(const CFrame &frame) {
    //frames
    _frame = frame._frame.clone();
    _frameGrey = frame._frameGrey.clone();
    
    //features
    _keypts = frame._keypts;
    _pts = frame._pts;
    _descriptors = frame._descriptors.clone();
    
    //3d pts
    _status = frame._status;
    _pts3DIdx = frame._pts3DIdx;
    
    //pose info
    _R = frame._R;
    _t = frame._t;
    _P = frame._P;
    _K = frame._K;
    
    //number matched
    _nMatched = frame._nMatched;
    
    //frame number
    _frameNo = frame._frameNo;
}

CFrame::~CFrame() {
}

CFrame& CFrame::operator=(const CFrame &frame) {
    //frames
    _frame = frame._frame.clone();
    _frameGrey = frame._frameGrey.clone();
    
    //features
    _keypts = frame._keypts;
    _pts = frame._pts;
    _descriptors = frame._descriptors.clone();

    //3d pts
    _status = frame._status;
    _pts3DIdx = frame._pts3DIdx;
    
    //pose info
    _R = frame._R;
    _t = frame._t;
    _P = frame._P;
    _K = frame._K;
    
    //number matched
    _nMatched = frame._nMatched;
    
    //frame number
    _frameNo = frame._frameNo;
    
    return *this;
}

void CFrame::setIntrinsic(Matx33d K) {
    _K = K;
}

void CFrame::setFrame(Mat frameIn, int frameNo, Matx33d K) {
    _frame = frameIn.clone();
    cvtColor(_frame, _frameGrey, CV_RGB2GRAY);
    _keypts.clear();
    _pts.clear();
    _descriptors = NULL;
    _frameNo = frameNo;
    _K = K;
}


void CFrame::setFrame(Mat frameIn, int frameNo) {
    _frame = frameIn.clone();
    cvtColor(_frame, _frameGrey, CV_RGB2GRAY);
    _keypts.clear();
    _pts.clear();
    _descriptors = NULL;
    _frameNo = frameNo;
}

void CFrame::setPoints(const vector<Point2d> &p) {
    _pts = p;
    //convert to keypoints
    _keypts.reserve(p.size());
    assert(_keypts.size() == 0);
    for (int i = 0; i < _pts.size(); i++) {
        _keypts.push_back(KeyPoint(_pts[i].x, _pts[i].y, 0));
    }

    //reset status vector
    _status.assign(p.size(), 0);
    _pts3DIdx.assign(p.size(), 0);
}

void CFrame::setPoints(const vector<Point2f> &p) {
    _pts.reserve(p.size());
    _keypts.reserve(p.size());

    //convert to keypoints
    assert(_keypts.size() == 0);
    for (int i = 0; i < p.size(); i++) {
        _keypts.push_back(KeyPoint(p[i].x, p[i].y, 0));
        _pts.push_back(Point2d(p[i].x,p[i].y));
    }
    
    //reset status vector
    _status.assign(p.size(), 0);
    _pts3DIdx.assign(p.size(), 0);
}

void CFrame::setKeyPoints(const vector<KeyPoint> &kp, const Mat desc) {
    _descriptors = desc;
    setKeyPoints(kp,desc);
}

void CFrame::setKeyPoints(const vector<cv::KeyPoint> &kp) {
    _keypts = kp;
    //convert to just points coordinates
    assert(_pts.size() == 0);
    for (int i = 0; i < _keypts.size(); i++) {
        _pts.push_back(Point2d(_keypts[i].pt.x,_keypts[i].pt.y));
    }
    
    //reset status vector
    _status.assign(_pts.size(), 0);
    //reset 3D index vector
    _pts3DIdx.assign(_pts.size(),0);
}

void CFrame::setPose() {
    //set pose as origin
    _R = Matx33f::eye();
    _t[0] = 0, _t[1] = 0, _t[2] = 0;
    calculateProjectionMatrix();
}

void CFrame::setPose(const Matx33d &R, const Vec3d &t) {
    _R = Mat(R).clone();
    _t = Mat(t).clone();
    calculateProjectionMatrix();
}

void CFrame::calculateProjectionMatrix() {
    //calculate projection matrix based on internal R and t
    _P << _R(0,0), _R(0,1), _R(0,2), _t(0), _R(1,0), _R(1,1), _R(1,2), _t(1), _R(2,0), _R(2,1), _R(2,2), _t(2);
}

void CFrame::updatePoints(const vector<Point2d> &pts2D, const vector<int> &pts2DIdx, const vector<int> &pts3DIdx) {
    
    for (int i = 0; i < pts2DIdx.size(); i++) {
        int idx = pts2DIdx[i];
        _pts[idx] = pts2D[i];
        _status[idx] = 1;
        _pts3DIdx[idx] = pts3DIdx[i];
    }
    
    _nMatched = sum(_status)[0];
    
}

void CFrame::updatePoints(const vector<Point2f> &pts2D, const vector<int> &pts2DIdx, const vector<int> &pts3DIdx) {
    
    for (int i = 0; i < pts2DIdx.size(); i++) {
        int idx = pts2DIdx[i];
        _pts[idx] = Point2d(pts2D[i]);
        _status[idx] = 1;
        _pts3DIdx[idx] = pts3DIdx[i];
    }
    
    _nMatched = sum(_status)[0];
    
}

void CFrame::updatePoints(const vector<int> &pts2DIdx, const vector<int> &pts3DIdx) {
    for (int i = 0; i < pts2DIdx.size(); i++) {
        int idx = pts2DIdx[i];
        _status[idx] = 1;
        _pts3DIdx[idx] = pts3DIdx[i];
    }
    
    _nMatched = sum(_status)[0];
    
}

void CFrame::getMatchedPoints(vector<Point2d> &pts2D, vector<int> &pts3DIdx, vector<int> &pts2DIdx) {
    

    //preallocate memory for speed
    pts2D.reserve(_nMatched);
    pts3DIdx.reserve(_nMatched);
    pts2DIdx.reserve(_nMatched);
    
    for (int i = 0; i < _status.size(); i++) {
        if (_status[i] == 1) {
            pts2D.push_back(_pts[i]);
            pts3DIdx.push_back(_pts3DIdx[i]);
            pts2DIdx.push_back(i);
        }
    }
    
}

void CFrame::getMatchedPoints(vector<Point2d> &pts2D, vector<int> &pts3DIdx) {

    //preallocate memory for speed
    pts2D.reserve(_nMatched);
    pts3DIdx.reserve(_nMatched);
    
    for (int i = 0; i < _status.size(); i++) {
        if (_status[i] == 1) {
            pts2D.push_back(_pts[i]);
            pts3DIdx.push_back(_pts3DIdx[i]);
        }
    }
    
}

void CFrame::getPoints(vector<Point2f> &pts) {
    
    
}

int CFrame::getNMatchedPoints() {
    return _nMatched;
}

Mat CFrame::getRotationRodrigues() {
    Mat rvec;
    Rodrigues(_R, rvec);
    return rvec;
}

int CFrame::findClosestPointIndex(Point2f pt) {
    int idx = -1;
    double minDist = DBL_MAX;
    
    for (int i = 0; i < _pts.size(); i++) {
        double d = (_pts[i].x - pt.x)*(_pts[i].x - pt.x) + (_pts[i].y - pt.y)*(_pts[i].y - pt.y);
        if (d < minDist) {
            minDist = d;
            idx = i;
        }
    }
    
    return idx;
}
