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
    //preallocate for speed
    int newSize = _pts3D.size() + pts3D.size();
    pts3DIdx.reserve(pts3D.size());
    _pts3D.reserve(newSize);
    _pts3DIdx.reserve(newSize);
    _frameNo.reserve(newSize);
    _pts2DIdx.reserve(newSize);
    _descriptor.reserve(newSize);
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
        pts3DIdx.push_back(_lastPtNo);
        _lastPtNo++;
    }
    
    //points have been observed in all frames
    for (int i = 0; i < frameNo.size(); i++) {
        for (int j = i+1; j < frameNo.size(); j++) {
            addCovisiblePoint(frameNo[i], frameNo[j],pts3D.size());
        }
    }
    
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
    return _pts3DIdx.size();
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

void CMap::getPointsAtIdx(const vector<int> &pts3DIdx, vector<Matx31d> &pts3D) {
    
    //preallocate memory for speed
    pts3D.reserve(pts3DIdx.size());
    
    for (int i = 0; i < pts3DIdx.size(); i++) {
        int idx = pts3DIdx[i];
        pts3D.push_back(_pts3D[idx]);
    }
}

void CMap::getPointsInFrame(vector<Matx31d> &pts3D, vector<int> &pts2DIdx, const int frameNo) {
    
    int nPoints = _frameViewsPointIdx.count(frameNo);
    auto range = _frameViewsPointIdx.equal_range(frameNo);
    pts3D.reserve(pts3D.size()+nPoints);
    pts2DIdx.reserve(pts2DIdx.size()+nPoints);
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
    
    int nPoints = _frameViewsPointIdx.count(frameNo);
    auto range = _frameViewsPointIdx.equal_range(frameNo);
    pts3D.reserve(pts3D.size()+nPoints);
    pts3DIdx.reserve(pts3DIdx.size()+nPoints);
    pts2DIdx.reserve(pts2DIdx.size()+nPoints);
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
    
    int nPoints = _frameViewsPointIdx.count(frameNo);
    auto range = _frameViewsPointIdx.equal_range(frameNo);
    pts3D.reserve(pts3D.size()+nPoints);
    pts3DIdx.reserve(pts3DIdx.size()+nPoints);
    pts2DIdx.reserve(pts2DIdx.size()+nPoints);
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

void CMap::getPointsInFrame_Mutable(vector<double *> &pts3D, vector<int> &pts2DIdx, const int frameNo) {
    
    int nPoints = _frameViewsPointIdx.count(frameNo);
    auto range = _frameViewsPointIdx.equal_range(frameNo);
    pts3D.reserve(pts3D.size()+nPoints);
    pts2DIdx.reserve(pts2DIdx.size()+nPoints);
    if ((range.first != _frameViewsPointIdx.end()) && (range.second != range.first)) {
        for (auto it = range.first; it != range.second; ++it) {
            int ptIdx = it->second;
            pts3D.push_back(_pts3D[ptIdx].val);
            for (int i = 0; i < _frameNo[ptIdx].size(); i++) {
                if (_frameNo[ptIdx][i] == frameNo) {
                    pts2DIdx.push_back(_pts2DIdx[ptIdx][i]);
                }
            }
        }
    }
}

void CMap::getPointsInFrame(vector<int> &pts3DIdx, vector<int> &pts2DIdx, const int frameNo) {
    
    int nPoints = _frameViewsPointIdx.count(frameNo);
    auto range = _frameViewsPointIdx.equal_range(frameNo);
    pts3DIdx.reserve(pts3DIdx.size()+nPoints);
    pts2DIdx.reserve(pts2DIdx.size()+nPoints);
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
        int count = _frameViewsPointIdx.count(frameNo[i]);
        pts3DIdx.reserve(pts3DIdx.size()+count);
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
        int idx = pts3DIdx[i];
        pts3D.push_back(pts3D[idx]);
    }
}

void CMap::getPointsInFrames(vector<int> &pts3DIdx, const vector<int> &frameNo) {
    
    for (int i = 0; i < frameNo.size(); i++) {
        //find points for each frame
        auto range = _frameViewsPointIdx.equal_range(frameNo[i]);
        int count = _frameViewsPointIdx.count(frameNo[i]);
        pts3DIdx.reserve(pts3DIdx.size()+count);
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
    pts3D.reserve(_pts3DIdx.size());
    for (int i = 0; i < _pts3DIdx.size(); i++) {
        int idx = _pts3DIdx[i];
        pts3D.push_back(_pts3D[idx]);
    }
   
}

void CMap::getPoints_Mutable(vector<double*> &pts3D) {
    pts3D.reserve(_pts3DIdx.size());
    for (int i = 0; i < _pts3DIdx.size(); i++) {
        int idx = _pts3DIdx[i];
        pts3D.push_back(_pts3D[idx].val);
    }
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
    x.reserve(_pts3DIdx.size());
    y.reserve(_pts3DIdx.size());
    z.reserve(_pts3DIdx.size());
    
    for (int i = 0; i < _pts3DIdx.size(); i++) {
        int idx = _pts3DIdx[i];
        x.push_back(_pts3D[idx].val[0]);
        y.push_back(_pts3D[idx].val[1]);
        z.push_back(_pts3D[idx].val[2]);
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


void CMap::getPointsInFrame(vector<int> &pts3DIdx, const int frameNo) {
    auto range = _frameViewsPointIdx.equal_range(frameNo);
    int count = _frameViewsPointIdx.count(frameNo);
    pts3DIdx.reserve(pts3DIdx.size()+count);
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
    for (int i = 0; i < _pts3DIdx.size(); i++) {
        int idx = _pts3DIdx[i];
        if (_frameNo[idx].size() < threshold) {
            ptsCullIdx.push_back(idx);
        }
    }
    sort(ptsCullIdx.begin(),ptsCullIdx.end());
    removePoints(ptsCullIdx, newPtsIdx);
}

void CMap::removePoints(const vector<int> &ptsCullIdx, vector<int> &ptsStatusFlag) {
    
    //get points to keep
    vector<int> ptsToKeep(_pts3DIdx.size());
    vector<int>::iterator it = set_difference(_pts3DIdx.begin(), _pts3DIdx.end(), ptsCullIdx.begin(), ptsCullIdx.end(), ptsToKeep.begin());
    ptsToKeep.resize(it - ptsToKeep.begin());
    swap(_pts3DIdx, ptsToKeep);
    
    //remove from graphs
    for (int i = 0; i < ptsCullIdx.size(); i++) {
        pair<unordered_multimap<int,int>::iterator,unordered_multimap<int,int>::iterator> affectedFrames = _pointInFrameIdx.equal_range(ptsCullIdx[i]);
        unordered_multimap<int,int>::iterator begFrame = affectedFrames.first;
        unordered_multimap<int,int>::iterator endFrame = affectedFrames.second;
        if ((begFrame != _pointInFrameIdx.end()) && (endFrame != begFrame)) {
            
            for (unordered_multimap<int,int>::iterator it_out = begFrame; it_out != endFrame; ++it_out) {
                int idx0 = it_out->second;
                //remove point from frame views
                auto range  = _frameViewsPointIdx.equal_range(idx0);
                for (auto it = range.first; it != range.second; ++it) {
                    if (it->second == ptsCullIdx[i]) {
                        _frameViewsPointIdx.erase(it);
                        break;
                    }
                }
                
                unordered_multimap<int,int>::iterator it_copy = it_out;
                for (unordered_multimap<int,int>::iterator it_in = ++it_copy; it_in != endFrame; ++it_in) {
                    
                    //update covisibility graph
                    int idx1 = it_in->second;
                    tuple<int,int> key0(idx0,idx1);
                    tuple<int,int> key1(idx1,idx0);
                    _covisibilityGraph[key0]+= -1;
                    _covisibilityGraph[key1]+= -1;
                    
                    //if link strength goes to zero, remove link
                    if (_covisibilityGraph[key0] == 0) {
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
            
            //remove point
            _pointInFrameIdx.erase(ptsCullIdx[i]);
        }
    }
    
    //prepare map flagging points to remove
    ptsStatusFlag.resize(_pts3D.size());
    for (int i = 0; i < ptsCullIdx.size(); i++) {
        ptsStatusFlag[ptsCullIdx[i]] = -1;
    }
}

int CMap::getPointFrameVisibility(const int pt3DIdx) {
    int nFrameVisible = 0;
    nFrameVisible = _pointInFrameIdx.count(pt3DIdx);
    
    return nFrameVisible;
}

void CMap::removeFrame(const int frameNo, const vector<int> &framePtsIdx) {
    
    //for all points, remove from vectors
    for (int i = 0; i < framePtsIdx.size(); i++) {
        //get position
        int ptIdx = framePtsIdx[i];
        
        //it is not actually needed to remove information from vectors as long as it is removed from lookup multimaps
        vector<int>::iterator vectIt = lower_bound(_frameNo[ptIdx].begin(), _frameNo[ptIdx].end(), frameNo);
        int pos = vectIt - _frameNo[ptIdx].begin();
        
        //remove from vectors
        _frameNo[ptIdx].erase(vectIt);
        _pts2DIdx[ptIdx].erase(_pts2DIdx[ptIdx].begin() + pos);
        Mat newDesc(_descriptor[ptIdx].rows -1 , _descriptor[ptIdx].cols, _descriptor[ptIdx].type());
        Size dSize = newDesc.size();
        if (pos > 0) {
            Rect roi(0,0,dSize.width,pos);
            _descriptor[ptIdx](roi).copyTo(newDesc(roi));
        }
        if (pos < (dSize.height - 1)) {
            Rect roiFrom(0,pos+1,dSize.width,dSize.height-pos-1);
            Rect roiTo(0, pos, dSize.width, dSize.height-pos-1);
            _descriptor[ptIdx](roiFrom).copyTo(newDesc(roiTo));
        }
        _descriptor[ptIdx] = newDesc;
        
        //remove frame from list associated to point
        pair<unordered_multimap<int,int>::iterator, unordered_multimap<int,int>::iterator> frameRange = _pointInFrameIdx.equal_range(ptIdx);
        for (auto itFrame = frameRange.first; itFrame != frameRange.second; ++itFrame) {
            int fIdx = itFrame->second;
            if (fIdx == frameNo) {
                _pointInFrameIdx.erase(itFrame);
                break;
            }
        }
    }
    
    //remove from graphs
    _frameViewsPointIdx.erase(frameNo);
    pair<unordered_multimap<int,int>::iterator,unordered_multimap<int,int>::iterator> affectedFrames = _covisibilityFrameIdx.equal_range(frameNo);
    for (auto it = affectedFrames.first; it != affectedFrames.second; ++it) {
        pair<unordered_multimap<int,int>::iterator,unordered_multimap<int,int>::iterator> singleFrame = _covisibilityFrameIdx.equal_range(it->second);
        for (auto it2 = singleFrame.first; it2 != singleFrame.second; ++it2) {
            if (it2->second == frameNo) {
                _covisibilityFrameIdx.erase(it2);
                break;
            }
        }
        tuple<int,int> key0(frameNo,it->second);
        tuple<int,int> key1(it->second,frameNo);
        _covisibilityGraph.erase(key0);
        _covisibilityGraph.erase(key1);
    }
    _covisibilityFrameIdx.erase(frameNo);
}


bool CMap::arePointsSeenByAtLeast(const vector<int> &pts3DIdx, const int nFramesThreshold, const double ratioPtsThreshold) {
    
    int ptsThreshold = ceil(ratioPtsThreshold*pts3DIdx.size());
    int countTrue = 0;
    for (int i = 0; i < pts3DIdx.size(); i++) {
        int idx = pts3DIdx[i];
        if (_frameNo[idx].size() > nFramesThreshold) {
            countTrue++;
            if (countTrue > ptsThreshold)
                return true;
        }
    }
    
    return false;
}
