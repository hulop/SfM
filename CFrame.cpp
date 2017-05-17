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

CFrame::CFrame(const Matx33d &K, const vector<double> &d, const Size &imSize) {

    _nMatched = 0;
    
    //set intrinsics
    _d = d;
    _K = K;
    _imSize = imSize;
    _Kopt = getOptimalNewCameraMatrix(_K, _d, _imSize, 0);
    
    //set statistics
    _meanErr = 0;
    _maxErr = 0;
    
}

CFrame::CFrame(const Mat &frameIn, const Matx33d &K, const vector<double> &d, const Size &imSize) {
    //convert to grayscale and save
    _frame = frameIn;
    cvtColor(frameIn, _frameGrey, CV_RGB2GRAY);
    _nMatched = 0;
    
    //set intrinsics
    _d = d;
    _K = K;
    _imSize = imSize;
    _Kopt = getOptimalNewCameraMatrix(_K, _d, _imSize, 0);
}

CFrame::CFrame(const CFrame &frame) {
    //frames
    _frame = frame._frame.clone();
    _frameGrey = frame._frameGrey.clone();
    
    //features
    _keypts = frame._keypts;
    _pts = frame._pts;
    _pts_dist = frame._pts_dist;
    _descriptors = frame._descriptors.clone();
    
    //3d pts
    _status = frame._status;
    _pts3DIdx = frame._pts3DIdx;
    
    //pose info
    _R = frame._R;
    _t = frame._t;
    _P = frame._P;
    _rot = frame._rot;
    
    //number matched
    _nMatched = frame._nMatched;
    
    //frame number
    _frameNo = frame._frameNo;
    
    //intrinsics
    _Kopt = frame._Kopt;
    _d = frame._d;
    _K = frame._K;
    _imSize = frame._imSize;
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
    _pts_dist = frame._pts_dist;
    _descriptors = frame._descriptors.clone();

    //3d pts
    _status = frame._status;
    _pts3DIdx = frame._pts3DIdx;
    
    //pose info
    _R = frame._R;
    _t = frame._t;
    _P = frame._P;
    _rot = frame._rot;
    
    //number matched
    _nMatched = frame._nMatched;
    
    //frame number
    _frameNo = frame._frameNo;
    
    //intrinsics
    _Kopt = frame._Kopt;
    _d = frame._d;
    _K = frame._K;
    _imSize = frame._imSize;
    
    return *this;
}

void CFrame::resetInternals() {
    _keypts.clear();
    _pts_dist.clear();
    _pts.clear();
    _status.clear(); //check if point has been matched already
    _statusDist.clear();
    _nMatched = 0;
    _frameNo = -1;
    _pts3DIdx.clear();
    setPose();
}

void CFrame::setFrame(const Mat &frameIn, int frameNo, const Matx33d &K, const vector<double> &d, const Size &imSize) {
    
    //reset internal vectors
    resetInternals();
    
    _frame = frameIn.clone();
    cvtColor(_frame, _frameGrey, CV_RGB2GRAY);
    _frameNo = frameNo;
    
    //set new intrinsics
    _K = K;
    _d = d;
    _imSize = imSize;
    _Kopt = getOptimalNewCameraMatrix(_K, _d, _imSize, 0);
}


void CFrame::setFrame(const Mat &frameIn, int frameNo) {
    
    //reset internal vectors
    resetInternals();
    
    _frame = frameIn.clone();
    cvtColor(_frame, _frameGrey, CV_RGB2GRAY);
    _frameNo = frameNo;
}

void CFrame::setPoints(const vector<Point2d> &p) {
    _pts_dist = p;
    //undistort points
    undistortPoints(_pts_dist, _pts, _K, _d, _Kopt);
    
    //convert to keypoints
    _keypts.reserve(p.size());
    
    assert(_keypts.size() == 0);
    for (int i = 0; i < _pts.size(); i++) {
        _keypts.push_back(KeyPoint(_pts[i].x, _pts[i].y, 0));
    }

    //reset status vector
    _status.assign(p.size(), 0);
    _pts3DIdx.assign(p.size(), -1);
}

void CFrame::setPoints(const vector<Point2f> &p) {
    _pts_dist.reserve(p.size());
    for (int i = 0; i < p.size(); i++)
        _pts_dist.push_back(p[i]);
    
    undistortPoints(_pts_dist, _pts, _K, _d,_Kopt);
    _keypts.reserve(p.size());

    //convert to keypoints
    assert(_keypts.size() == 0);
    for (int i = 0; i < p.size(); i++) {
        _keypts.push_back(KeyPoint(_pts[i].x, _pts[i].y, 0));
    }
    
    //reset status vector
    _status.assign(p.size(), 0);
    _pts3DIdx.assign(p.size(), -1);
}

void CFrame::setKeyPoints(const vector<KeyPoint> &kp, const Mat &desc) {
    _descriptors = desc;
    setKeyPoints(kp);
}

void CFrame::setKeyPoints(const vector<cv::KeyPoint> &kp) {
    //convert to just points coordinates
    assert(_pts.size() == 0);
    for (int i = 0; i < kp.size(); i++) {
        _pts_dist.push_back(Point2d(kp[i].pt.x,kp[i].pt.y));
    }
    
    //undistort
    undistortPoints(_pts_dist, _pts, _K, _d, _Kopt);
    
    //reconvert to keypoints
    for (int i = 0; i < kp.size(); i++) {
        _keypts.push_back(KeyPoint(_pts[i].x,_pts[i].y,0));
    }
    
    //reset status vector
    _status.assign(_pts.size(), 0);
    //reset 3D index vector
    _pts3DIdx.assign(_pts.size(),-1);
}

void CFrame::setPose() {
    //set pose as origin
    _R = Matx33f::eye();
    Rodrigues(_R,_rot);
    _t.val[0] = 0, _t.val[1] = 0, _t.val[2] = 0;
    calculateProjectionMatrix();
}

void CFrame::setPose(const Matx33d &R, const Matx31d &t) {
    _R = Mat(R).clone();
    Rodrigues(_R, _rot);
    _t = t;
    calculateProjectionMatrix();
}

void CFrame::calculateProjectionMatrix() {
    //recalculate R from rodrigues in case it has been manipulated
    Rodrigues(_rot,_R);
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
        _pts[idx] = Point2d(pts2D[i].x,pts2D[i].y);
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

void CFrame::getMatchedPoints(vector<int> &pts2DIdx, vector<int> &pts3DIdx) {
    //preallocate memory for speed
    
    pts2DIdx.reserve(_nMatched);
    pts3DIdx.reserve(_nMatched);
    
    for (int i = 0; i < _status.size(); i++) {
        if (_status[i] == 1) {
            pts2DIdx.push_back(i);
            pts3DIdx.push_back(_pts3DIdx[i]);
        }
    }
}

void CFrame::getMatchedPoints(vector<int> &pts3DIdx) {
    pts3DIdx.reserve(_nMatched);
    
    for (int i = 0; i < _status.size(); i++) {
        if (_status[i] == 1)
            pts3DIdx.push_back(_pts3DIdx[i]);
    }
    
}

void CFrame::getMatchedIndices(vector<int> &pts2DIdx) {
    pts2DIdx.reserve(_nMatched);
    
    for (int i = 0; i < _status.size(); i++) {
        if (_status[i] == 1) {
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

void CFrame::getUnmatchedPoints(vector<Point2d> &pts2D, vector<int> &pts2DIdx) {
    
    int nPoints = _status.size() - _nMatched;
    pts2D.reserve(nPoints);
    pts2DIdx.reserve(nPoints);
    
    for (int i =0; i < _status.size(); i++) {
        if (_status[i] == 0) {
            pts2D.push_back(_pts[i]);
            pts2DIdx.push_back(i);
        }
    }
    
}

void CFrame::getUnmatchedPoints(vector<Point2d> &pts2D, Mat &desc, vector<int> &pts2DIdx) {
    
    int nPoints = _status.size() - _nMatched;
    pts2D.reserve(nPoints);
    pts2DIdx.reserve(nPoints);
    
    for (int i =0; i < _status.size(); i++) {
        if (_status[i] == 0) {
            pts2D.push_back(_pts[i]);
            pts2DIdx.push_back(i);
            desc.push_back(_descriptors.row(i));
        }
    }
    
}

void CFrame::getPointsAt(const vector<int> &pts2DIdx, vector<Point2d> &pts2D) {
        pts2D.reserve(pts2D.size()+pts2DIdx.size());
    for (int i = 0; i < pts2DIdx.size(); i++) {
        pts2D.push_back(_pts[pts2DIdx[i]]);
    }
}

void CFrame::getPointsAt(const vector<int> &pts2DIdx, vector<Point2f> &pts2D) {
    
    pts2D.reserve(pts2D.size()+pts2DIdx.size());
    for (int i = 0; i < pts2DIdx.size(); i++) {
        int idx = pts2DIdx[i];
        pts2D.push_back(Point2f(_pts[idx].x,_pts[idx].y));
    }
}

void CFrame::getPointsDistortedAt(const vector<int> &pts2DIdx, vector<Point2d> &pts2D) {
    pts2D.reserve(pts2D.size()+pts2DIdx.size());
    for (int i =0; i < pts2DIdx.size(); i++) {
        pts2D.push_back(_pts_dist[pts2DIdx[i]]);
    }
    
}

void CFrame::getPoints3DIdxAt(const vector<int> &pts2DIdx, vector<int> &pts3DIdx) {
    pts3DIdx.reserve(pts2DIdx.size());
    for (int i = 0; i < pts2DIdx.size(); i++) {
        int idx = pts2DIdx[i];
        pts3DIdx.push_back(_pts3DIdx[idx]);
    }
    
}

int CFrame::getNMatchedPoints() {
    return _nMatched;
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

int CFrame::findClosestPointIndexDistorted(Point2f pt) {
    int idx = -1;
    double minDist = DBL_MAX;
    
    for (int i = 0; i < _pts_dist.size(); i++) {
        double d = (_pts_dist[i].x - pt.x)*(_pts_dist[i].x - pt.x) + (_pts_dist[i].y - pt.y)*(_pts_dist[i].y - pt.y);
        if (d < minDist) {
            minDist = d;
            idx = i;
        }
    }
    
    return idx;
}


void CFrame::getDescriptorsAt(const vector<int> &pts2DIdx, cv::Mat &descriptors) {
    
    for (int i = 0; i < pts2DIdx.size(); i++) {
        descriptors.push_back(_descriptors.row(pts2DIdx[i]));
    }
    
}

void CFrame::cullPoints(const vector<int> &pts3DIdx) {
    for (int i = 0; i < _pts3DIdx.size(); i++) {
        int idx = _pts3DIdx[i];
        if (idx != -1) {
            //check the new index
            if (pts3DIdx[idx] == -1)
                _status[i] = 0;
        }
    }
    _nMatched = sum(_status)[0];
}

void CFrame::updateFrameErrorStatistics(const double meanErr, const double maxErr) {
    _meanErr = meanErr;
    _maxErr = maxErr;
}
