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


void CMap::addNewPoints(const vector<Matx31d> &pts3D, const vector<vector<int> > &pts2DIdx, const vector<int> &frameNo, vector<int> &pts3DIdx) {
    
    const int oldCount = _lastPtNo;
    
    for (int i = 0; i < pts3D.size(); i++) {
        _pts3D.push_back(pts3D[i]);
        _pts3DIdx.push_back(_lastPtNo);
        _frameNo.push_back(vector<int> ());
        _pts2DIdx.push_back(vector<int> ());
        //prepare descriptor
        _descriptor.push_back(Mat());
        for (int j = 0; j < pts2DIdx.size(); j++) {
            _frameNo[_lastPtNo].push_back(frameNo[j]);
            _pts2DIdx[_lastPtNo].push_back(pts2DIdx[j][i]);
            //push frame index in point visibility graph
            _pointInFrameIdx.emplace(_lastPtNo,frameNo[j]);
            _frameViewsPointIdx.emplace(frameNo[j],_lastPtNo);
        }
        _lastPtNo++;
    }
    
    //points have been observed in all frames
    for (int i = 0; i < frameNo.size(); i++) {
        for (int j = i+1; j < frameNo.size(); j++) {
            addCovisiblePoint(frameNo[i], frameNo[j],pts3D.size());
        }
    }
    
    vector<int>::const_iterator first = _pts3DIdx.begin() + oldCount;
    vector<int>::const_iterator last = _pts3DIdx.end();
    pts3DIdx = vector<int>(first,last);
}

void CMap::addCovisiblePoint(int idx0, int idx1, int increment) {
    tuple<int,int> key0(idx0,idx1);
    tuple<int,int> key1(idx1,idx0);
    int isFound = _covisibilityGraph.count(key0);
    if (isFound == 0) {
        _covisibilityGraph[key0] = increment;
        _covisibilityGraph[key1] = increment;
        _covisibilityFrameIdx.emplace(idx0,idx1);
        _covisibilityFrameIdx.emplace(idx1,idx0);
    } else {
        _covisibilityGraph[key0]+= increment;
        _covisibilityGraph[key1]+= increment;
        if (_covisibilityGraph[key0] == 0) {
            //remove link (happens when increment is negative)
            auto range = _covisibilityFrameIdx.equal_range(idx0);
            for (auto it = range.first; it != range.second; ++it) {
                if (it->second == idx1) {
                    _covisibilityFrameIdx.erase(it);
                    break;
                }
            }
            range = _covisibilityFrameIdx.equal_range(idx1);
            for (auto it = range.first; it != range.second; ++it) {
                if (it->second == idx0) {
                    _covisibilityFrameIdx.erase(it);
                    break;
                }
            }
            _covisibilityGraph.erase(key0);
            _covisibilityGraph.erase(key1);
        }
    }
}

int CMap::getNPoints() {
    return _pts3D.size();
}

void CMap::addPointMatches(const vector<int> &pts3DIdx, const vector<int> &pts2DIdx, const int frameNo) {
    
    for (int i = 0; i < pts3DIdx.size(); i++) {
        int idx = pts3DIdx[i];
        _pts2DIdx[idx].push_back(pts2DIdx[i]);
        _frameNo[idx].push_back(frameNo);
        //update lookup map
        _pointInFrameIdx.emplace(idx,frameNo);
        _frameViewsPointIdx.emplace(frameNo,idx);
        //update covisibility
        for (int j = 0; j < _frameNo[idx].size()-1; j++)
            addCovisiblePoint(frameNo, _frameNo[idx][j], 1);
    }
}


int CMap::countMatchesBetweenFrames(int f0, int f1) {
    //count number of 3d points observed in frames f0 and f1
    int commonPts = 0;
    for (int i = 0; i < _pts3D.size(); i++) {
        bool found0 = false, found1 = false;
        for (int j = 0; j < _frameNo[i].size(); i++) {
            if (_frameNo[i][j] == f0)
                found0 = true;
            else if (_frameNo[i][j] == f1)
                found1 = true;
        
            if (found0 && found1) {
                commonPts++;
                break;
            }
        }
    }
    return commonPts;
}

void CMap::getPointsAtIdx(const vector<int> &pts3DIdx, vector<Matx31d> &pts3D) {
    
    //preallocate memory for speed
    pts3D.reserve(pts3DIdx.size());
    
    for (int i = 0; i < pts3DIdx.size(); i++) {
        int idx = pts3DIdx[i];
        pts3D.push_back(_pts3D[idx]);
    }
}

void CMap::getPointsInFrame(vector<Matx31d> &pts3D, vector<int> &pts2DIdx, const int frameNo) {
    
    auto range = _frameViewsPointIdx.equal_range(frameNo);
    if ((range.first != _frameViewsPointIdx.end()) && (range.second != range.first)) {
        for (auto it = range.first; it != range.second; ++it) {
            int ptIdx = it->second;
            pts3D.push_back(_pts3D[ptIdx]);
            for (int i = 0; i < _frameNo[ptIdx].size(); i++) {
                if (_frameNo[ptIdx][i] == frameNo) {
                    pts2DIdx.push_back(_pts2DIdx[ptIdx][i]);
                }
            }
        }
    }
}

void CMap::getPointsInFrame(vector<Matx31d> &pts3D, vector<int> &pts3DIdx, vector<int> &pts2DIdx, const int frameNo) {
    
    auto range = _frameViewsPointIdx.equal_range(frameNo);
    if ((range.first != _frameViewsPointIdx.end()) && (range.second != range.first)) {
        for (auto it = range.first; it != range.second; ++it) {
            int ptIdx = it->second;
            pts3D.push_back(_pts3D[ptIdx]);
            pts3DIdx.push_back(ptIdx);
            for (int i = 0; i < _frameNo[ptIdx].size(); i++) {
                if (_frameNo[ptIdx][i] == frameNo) {
                    pts2DIdx.push_back(_pts2DIdx[ptIdx][i]);
                }
            }
        }
    }
}

void CMap::getPointsInFrame_Mutable(vector<double *> &pts3D, vector<int> &pts3DIdx, vector<int> &pts2DIdx, const int frameNo) {
    
    auto range = _frameViewsPointIdx.equal_range(frameNo);
    if ((range.first != _frameViewsPointIdx.end()) && (range.second != range.first)) {
        for (auto it = range.first; it != range.second; ++it) {
            int ptIdx = it->second;
            pts3D.push_back(_pts3D[ptIdx].val);
            pts3DIdx.push_back(ptIdx);
            for (int i = 0; i < _frameNo[ptIdx].size(); i++) {
                if (_frameNo[ptIdx][i] == frameNo) {
                    pts2DIdx.push_back(_pts2DIdx[ptIdx][i]);
                }
            }
        }
    }
}

void CMap::getPointsInFrame(vector<int> &pts3DIdx, vector<int> &pts2DIdx, const int frameNo) {
    auto range = _frameViewsPointIdx.equal_range(frameNo);
    if ((range.first != _frameViewsPointIdx.end()) && (range.second != range.first)) {
        for (auto it = range.first; it != range.second; ++it) {
            int ptIdx = it->second;
            pts3DIdx.push_back(ptIdx);
            for (int i = 0; i < _frameNo[ptIdx].size(); i++) {
                if (_frameNo[ptIdx][i] == frameNo) {
                    pts2DIdx.push_back(_pts2DIdx[ptIdx][i]);
                }
            }
        }
    }
}

void CMap::getPointsInFrames(vector<Matx31d> &pts3D, vector<int> &pts3DIdx, const vector<int> &frameNo) {
    
    for (int i = 0; i < frameNo.size(); i++) {
        //find points for each frame
        auto range = _frameViewsPointIdx.equal_range(frameNo[i]);
        if ((range.first != _frameViewsPointIdx.end()) && (range.second != range.first)) {
            for (auto it = range.first; it != range.second; ++it)
                pts3DIdx.push_back(it->second);
        }
    }
    
    //make sure vectors contain unique values
    sort(pts3DIdx.begin(),pts3DIdx.end());
    vector<int>::iterator endIt = unique(pts3DIdx.begin(),pts3DIdx.end());
    pts3DIdx.resize(distance(pts3DIdx.begin(),endIt));
    
    //copy 3d points
    pts3D.reserve(pts3DIdx.size());
    for (int i = 0; i < pts3DIdx.size(); i++) {
        pts3D.push_back(pts3D[i]);
    }
}

void CMap::getPointsInFrames(vector<int> &pts3DIdx, const vector<int> &frameNo) {
    
    for (int i = 0; i < frameNo.size(); i++) {
        //find points for each frame
        auto range = _frameViewsPointIdx.equal_range(frameNo[i]);
        if ((range.first != _frameViewsPointIdx.end()) && (range.second != range.first)) {
            for (auto it = range.first; it != range.second; ++it)
                pts3DIdx.push_back(it->second);
        }
    }
    
    //make sure vectors contain unique values
    sort(pts3DIdx.begin(),pts3DIdx.end());
    vector<int>::iterator endIt = unique(pts3DIdx.begin(),pts3DIdx.end());
    pts3DIdx.resize(distance(pts3DIdx.begin(),endIt));
}


void CMap::getPoints(vector<Matx31d> &pts3D) {
    pts3D = _pts3D;
}

void CMap::getPoints_Mutable(vector<double*> &pts3D) {
    for (int i = 0; i < _pts3D.size(); i++)
        pts3D.push_back(_pts3D[i].val);
}


Point3d CMap::getCentroid() {
    updateCentroid();
    return _centroid;
    
}

void CMap::addDescriptors(const vector<int> &pts3DIdx, const cv::Mat &descriptors) {
    
    for (int i = 0; i < pts3DIdx.size(); i++) {
        int idx = pts3DIdx[i];
        _descriptor[idx].push_back(descriptors.row(i));
    }
    
}

void CMap::updateCentroid() {
    vector<float> x, y, z;
    x.reserve(_pts3D.size());
    y.reserve(_pts3D.size());
    z.reserve(_pts3D.size());
    
    for (int i = 0; i < _pts3D.size(); i++) {
        x.push_back(_pts3D[i].val[0]);
        y.push_back(_pts3D[i].val[1]);
        z.push_back(_pts3D[i].val[2]);
    }
    
    sort(x.begin(),x.end());
    sort(y.begin(),y.end());
    sort(z.begin(),z.end());
    int idx = round(_pts3D.size()/2.0);
    _centroid = {x[idx],y[idx],z[idx]};
}

//find all frames with a number of mutually visible features higher than threshold
void CMap::getFramesConnectedToFrame(int frameNo, vector<int> &covisibleFrames, int threshold) {

    //covisibleFrames.push_back(frameNo);
    
    //find frames that share link
    auto range = _covisibilityFrameIdx.equal_range(frameNo);
    for (auto it = range.first; it!= range.second; ++it) {
        tuple<int,int> key(frameNo,it->second);
        int linkStrength = _covisibilityGraph[key];
        if (linkStrength > threshold)
            covisibleFrames.push_back(it->second);
    }
}

//find nearest neighbours of a list of 3d points, returns indices and euclidean distance from the original points
void CMap::findNearestNeighbours(const vector<Matx31d> &pts3D, vector<int> &orgPtsIdx, vector<double> &dist) {
    
    
    
}


void CMap::getPointsInFrame(vector<int> &pts3DIdx, const int frameNo) {
    auto range = _frameViewsPointIdx.equal_range(frameNo);
    if ((range.first != _frameViewsPointIdx.end()) && (range.second != range.first)) {
        for (auto it = range.first; it != range.second; ++it)
            pts3DIdx.push_back(it->second);
    }    
}

//find descriptor that minimises hamming distance with all the existing descriptors.
//TODO: create own descriptor that minimises hamming distance (majority vote for each bit in the binary string)
void CMap::getRepresentativeDescriptors(const vector<int> &pts3DIdx, Mat &descriptors) {
    

    for (int i = 0; i < pts3DIdx.size(); i++) {
        int idx = pts3DIdx[i];
    
        Mat dist(_descriptor[idx].rows,_descriptor[idx].rows,CV_32F);
        dist.setTo(0);
        for (int j = 0; j < _descriptor[idx].rows; j++) {
            for (int k = j+1; k < _descriptor[idx].rows; k++) {
                //calculate hamming distance
                float d = norm(_descriptor[idx].row(j),_descriptor[idx].row(k),NORM_HAMMING);
                dist.ptr<float>(j)[k] = d;
                dist.ptr<float>(k)[j] = d;
            }
        }
    
        //sum distances across rows
        Mat sDist;
        reduce(dist,sDist,0,REDUCE_SUM);
    
        //find minimum distance
        int minIdx = -1;
        float minDist = FLT_MAX;
        for (int i = 0; i < sDist.cols; i++) {
            float d = sDist.ptr<float>(0)[i];
            if (d < minDist) {
                minDist = d;
                minIdx = i;
            }
        }
    
        //put descriptor in output
        descriptors.push_back(_descriptor[idx].row(minIdx));
    }
}


void CMap::removePoints(int threshold, vector<int> &ptsCullIdx, vector<int> &newPtsIdx) {
    //find points seen by less than threshold frames
    for (int i = 0; i < _frameNo.size(); i++) {
        if (_frameNo[i].size() < threshold) {
            ptsCullIdx.push_back(i);
        }
    }
    sort(ptsCullIdx.begin(),ptsCullIdx.end());
    removePoints(ptsCullIdx, newPtsIdx);
}

void CMap::removePoints(const vector<int> &ptsCullIdx, vector<int> &newPtsIdx) {
    
    size_t newSize = _pts3D.size() - ptsCullIdx.size();
    vector<Matx31d> pts3D_t(newSize);
    vector<Mat> descriptor_t(newSize);
    vector<vector<int>> frameNo_t(newSize);
    vector<vector<int>> pts2DIdx_t(newSize);
    
    size_t head = 0;
    size_t tail;
    size_t copied = 0;
    for (int i = 0; i < ptsCullIdx.size(); i++) {
        //find iterator of point to cull in point index array
        vector<int>::iterator it = find(_pts3DIdx.begin() + head, _pts3DIdx.end(), ptsCullIdx[i]);
        tail = it - _pts3DIdx.begin();
        
        if (tail != head) {
            //copy all elements up to that point to the temporary vectors
            swap_ranges(_pts3D.begin()+head,_pts3D.begin()+tail,pts3D_t.begin()+copied);
            swap_ranges(_descriptor.begin()+head, _descriptor.begin()+tail, descriptor_t.begin()+copied);
            swap_ranges(_frameNo.begin()+head, _frameNo.begin()+tail, frameNo_t.begin()+copied);
            swap_ranges(_pts2DIdx.begin()+head, _pts2DIdx.begin()+tail, pts2DIdx_t.begin()+copied);
        }
        
        int count = 0;
        for (size_t j = tail; j < _pts3DIdx.size(); j++) {
            int cullPt = ptsCullIdx[i + count];
            int orgPt = _pts3DIdx[j];
            if (cullPt != orgPt) {
                break;
            }
            count++;
        }
        
        head = tail + count;
        copied = pts3D_t.size();
        i += (count-1);
    }
    int pippo = 1;

//    //find vector of indices to keep
//    vector<int> ptsKeepIdx(_pts3DIdx.size() - ptsCullIdx.size());
//    vector<int>::iterator ptsIdxIt = set_difference(_pts3DIdx.begin(), _pts3DIdx.end(), ptsCullIdx.begin(), ptsCullIdx.end(), ptsKeepIdx.begin());
//    
//    //create vector mapping old indices to new ones
//    int count = 0;
//    for (int i = 0; i < _pts3DIdx.size(); i++) {
//        if(binary_search(ptsCullIdx.begin(), ptsCullIdx.end(), _pts3DIdx[i])) {
//            newPtsIdx.push_back(-1);
//        } else {
//            newPtsIdx.push_back(count);
//            count++;
//        }
//    }
//    
//    //create temporary copies of the vectors
//    vector<Matx31d> pts3D_t; pts3D_t.reserve(ptsKeepIdx.size());
//    vector<Mat> descriptor_t; descriptor_t.reserve(ptsKeepIdx.size());
//    vector<vector<int>> frameNo_t; frameNo_t.reserve(ptsKeepIdx.size());
//    vector<vector<int>> pts2DIdx_t; pts2DIdx_t.reserve(ptsKeepIdx.size());
//    vector<int> pts3DIdx_t;
//    _lastPtNo = 0;
//    for (int i = 0; i < ptsKeepIdx.size(); i++) {
//        int idx = ptsKeepIdx[i];
//        swap(pts3D_t[i],push_back(_pts3D[idx]));
//        descriptor_t.push_back(_descriptor[idx]);
//        pts3DIdx_t.push_back(i);
//        
//        swap(frameNo_t[i],_frameNo[idx]);
//        swap(pts2DIdx_t[i],_pts2DIdx[idx]);
//        _lastPtNo++;
//    }
//
//    //swap vectors
//    swap(_pts3D,pts3D_t);
//    swap(_descriptor, descriptor_t);
//    swap(_frameNo, frameNo_t);
//    swap(_pts2DIdx, pts2DIdx_t);
//    swap(_pts3DIdx, pts3DIdx_t);
//    
//    //rebuild covisibility graph
//    _frameViewsPointIdx.clear();
//    _pointInFrameIdx.clear();
//    _covisibilityGraph.clear();
//    _covisibilityFrameIdx.clear();
//    for (int i = 0; i < _pts3D.size(); i++) {
//        for (int j = 0; j < _frameNo[i].size(); j++) {
//            _pointInFrameIdx.emplace(_pts3DIdx[i],_frameNo[i][j]);
//            _frameViewsPointIdx.emplace(_frameNo[i][j],_pts3DIdx[i]);
//            for (int k = j+1; k < _frameNo[i].size(); k++) {
//                addCovisiblePoint(_frameNo[i][j], _frameNo[i][k]);
//            }
//        }
//    }
}
