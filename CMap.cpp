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

#include "CMap.h"

CMap::CMap() {
    _lastPtNo = 0;
    
}

CMap::~CMap() {
    
}


void CMap::addNewPoints(const vector<Point3d> &pts3D, const vector<vector<int> > &pts2DIdx, const vector<int> &frameNo, vector<int> &pts3DIdx) {
    
    const int oldCount = _lastPtNo;
    
    for (int i = 0; i < pts3D.size(); i++) {
        _pts3D.push_back(pts3D[i]);
        _pts3DIdx.push_back(_lastPtNo);
        _frameNo.push_back(vector<int> ());
        _pts2DIdx.push_back(vector<int> ());
        for (int j = 0; j < pts2DIdx.size(); j++) {
            _frameNo[oldCount + i].push_back(frameNo[j]);
            _pts2DIdx[oldCount + i].push_back(pts2DIdx[j][i]);
        }
        _lastPtNo++;
    }
    
    vector<int>::const_iterator first = _pts3DIdx.begin() + oldCount;
    vector<int>::const_iterator last = _pts3DIdx.end();
    pts3DIdx = vector<int>(first,last);
}

void CMap::getPointsAtIdx(const vector<int> &ptIdx, vector<Point3d> &pts3D) {
    
    //preallocate memory for speed
    pts3D.reserve(ptIdx.size());
    
    for (int i = 0; i < ptIdx.size(); i++) {
        pts3D.push_back(_pts3D[ptIdx[i]]);
    }
}

void CMap::getPointsInFrame(vector<Point3d> &pts3D, vector<int> &pts2DIdx, const int frameNo) {
    vector<int>::iterator idxIter;
    for (int i = 0; i < _pts3D.size(); i++) {
        idxIter = find(_frameNo[i].begin(), _frameNo[i].end(), frameNo);
        auto pos = idxIter - _frameNo[i].begin();
        //push point in result vector
        pts3D.push_back(_pts3D[i]);
        pts2DIdx.push_back(_pts2DIdx[i][pos]);
    }
}

void CMap::getPointsInFrame(vector<Point3d> &pts3D, vector<int> &pts3DIdx, vector<int> &pts2DIdx, const int frameNo) {
    vector<int>::iterator idxIter;
    for (int i = 0; i < _pts3D.size(); i++) {
        idxIter = find(_frameNo[i].begin(), _frameNo[i].end(), frameNo);
        auto pos = idxIter - _frameNo[i].begin();
        //push point in result vector
        pts3D.push_back(_pts3D[i]);
        pts3DIdx.push_back(_pts3DIdx[i]);
        pts2DIdx.push_back(_pts2DIdx[i][pos]);
    }
}

void CMap::addPointMatches(const vector<int> &pts3DIdx, const vector<int> &pts2DIdx, const int frameNo) {
    
    for (int i = 0; i < pts3DIdx.size(); i++) {
        int idx = pts3DIdx[i];
        _pts2DIdx[idx].push_back(pts2DIdx[i]);
        _frameNo[idx].push_back(frameNo);
    }
}

void CMap::getPoints(vector<Point3d> &pts3D) {
    pts3D = _pts3D;
}


Point3d CMap::getCentroid() {
    updateCentroid();
    return _centroid;
    
}

void CMap::updateCentroid() {
    vector<float> x, y, z;
    x.reserve(_pts3D.size());
    y.reserve(_pts3D.size());
    z.reserve(_pts3D.size());
    
    for (int i = 0; i < _pts3D.size(); i++) {
        x.push_back(_pts3D[i].x);
        y.push_back(_pts3D[i].y);
        z.push_back(_pts3D[i].z);
    }
    
    sort(x.begin(),x.end());
    sort(y.begin(),y.end());
    sort(z.begin(),z.end());
    int idx = round(_pts3D.size()/2.0);
    _centroid = {x[idx],y[idx],z[idx]};
}
