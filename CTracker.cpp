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

#include "CTracker.h"

CTracker::CTracker(const Matx33d &K, const vector<double> &d, const Size &imSize) : _currFrame(K,d,imSize), _prevFrame(K,d,imSize) {
    
    _ratioTest = 0.8;
    
    //other parameters
    _maxMatchDistance = 40; //maximum distance between matching features (small motion prior)
    _minMatchDistance = 1.5; //minimum distance between matching features (parallax constraint
    _minFeatures = 5; //minimum number of matches
    _maxOrgFeatDist = 1; //minimum distance between detected features and computed flow

    
    _maxHammingDistance = 90;
    
    //square thresholds for efficiency
    _minMatchDistanceSq = _minMatchDistance*_minMatchDistance;
    _maxMatchDistanceSq = _maxMatchDistance*_maxMatchDistance;
    
    //initialise detector and descriptor
    _detector = new brisk::BriskFeatureDetector(60,6,true);
    bool rotInvariant = true, scaleInvariant = true;
    _descriptor = new brisk::BriskDescriptorExtractor(rotInvariant,scaleInvariant,brisk::BriskDescriptorExtractor::Version::briskV2, 1.0);
    
}

CTracker::~CTracker() {
    
    
}

void CTracker::matchFeaturesRadius(const vector<Point2d> &pts0, const Mat &desc0, const vector<Point2d> &pts1, const Mat &desc1, vector<int> &matchIdx0, vector<int> &matchIdx1) {
    
    vector<vector<DMatch>> matches;
    vector<double> matchDistance(pts1.size(), -1);
    vector<int> matchedIdx(pts1.size(), -1);
    
    int matchCount = 0;
    
    for (int i = 0; i < pts0.size(); i++) {
        Point2d prevPt = pts0[i];
        
        //for all points detected in the current frame, find all candidate matches
        vector<int> candMatchIdx;
        vector<double> candMatchDist;
        for (int j = 0; j < pts1.size(); j++) {
            Point2d currPt = pts1[j];
            double dSq = (prevPt.x - currPt.x)*(prevPt.x - currPt.x) + (prevPt.y - currPt.y)*(prevPt.y - currPt.y);
            //check that candidate match is within allowed motion window
            if ((dSq < _maxMatchDistanceSq) && (dSq > _minMatchDistanceSq)) {
                //calculate hamming distance and check that is below the threshold and better than any other existing matches
                double ham = norm(desc0.row(i),desc1.row(j),NORM_HAMMING);
                if ((ham < _maxHammingDistance) && ((ham < matchDistance[j]) || (matchDistance[j] == -1))) {
                    candMatchIdx.push_back(j);
                    candMatchDist.push_back(ham);
                }
            }
        }
        
        if (candMatchIdx.size() != 0) {
            int idx = -1;
            if (candMatchIdx.size() > 1) {
                //sort distances and do ratio test
                vector<size_t> orgIdx = VectorUtils::sort_indexes(candMatchDist);
                size_t topMatchIdx = orgIdx[0];
                size_t secondMatchIdx = orgIdx[1];
                bool ratioTest = ((candMatchDist[topMatchIdx]/candMatchDist[secondMatchIdx]) < _ratioTest);
                if (ratioTest)
                    idx = candMatchIdx[topMatchIdx];
            } else
                idx = candMatchIdx[0];
            
            //find best candidate and save
            if (idx != -1) {
                if (matchDistance[idx] == -1) {
                    matchIdx0.push_back(i);
                    matchIdx1.push_back(idx);
                    matchedIdx[idx] = matchCount;
                    matchCount++;
                } else {
                    int oldIdx = matchedIdx[idx];
                    matchIdx0[oldIdx] = i;
                }
                matchDistance[idx] = candMatchDist[0];
            }
        }
    }
}

void CTracker::matchFeatures(const vector<Point2d> &pts0, const Mat &desc0, const vector<Point2d> &pts1, const Mat &desc1, vector<int> &matchIdx0, vector<int> &matchIdx1) {
    vector<vector<DMatch>> matches;
    
    _matcher.knnMatch(desc0, desc1, matches, 2);
    vector<double> matchDistance(pts1.size(), -1);
    vector<int> matchedIdx(pts1.size(), -1);
    
    int matchCount = 0;
    for (int i = 0; i < matches.size(); i++) {
        int prevIdx = matches[i][0].queryIdx;
        int currIdx = matches[i][0].trainIdx;
        double ratio = matches[i][0].distance/matches[i][1].distance;
        double d = (pts0[prevIdx].x - pts1[currIdx].x)*(pts0[prevIdx].x - pts1[currIdx].x) + (pts0[prevIdx].y - pts1[currIdx].y)*(pts0[prevIdx].y - pts1[currIdx].y);
        
        bool minDist = (d > _minMatchDistanceSq);
        bool maxDist = (d < _maxMatchDistanceSq);
        bool crossRatio = (ratio < _ratioTest);
        bool newMatch = (matchDistance[currIdx] == -1);
        bool betterMatch = (matches[i][0].distance < matchDistance[currIdx]);
        
        if (minDist && maxDist && crossRatio && (newMatch || betterMatch)) {
            //matches are not 1-1
            if (newMatch) {
                matchIdx0.push_back(prevIdx);
                matchIdx1.push_back(currIdx);
                matchedIdx[currIdx] = matchCount;
                matchCount++;
            }
            else {
                int oldIdx = matchedIdx[currIdx];
                matchIdx0[oldIdx] = prevIdx;
            }
            matchDistance[currIdx] = matches[i][0].distance;
        }
    }
}

void CTracker::matchFeaturesRadius(const vector<Point2d> &pts0, const Mat &desc0, const vector<Point2d> &pts1, const Mat &desc1, vector<int> &matchIdx0, vector<int> &matchIdx1, double minDistance, double maxDistance) {
    
    vector<vector<DMatch>> matches;
    vector<double> matchDistance(pts1.size(), -1);
    vector<int> matchedIdx(pts1.size(), -1);
   
    double minDistanceSq = minDistance*minDistance;
    double maxDistanceSq = maxDistance*maxDistance;
    int matchCount = 0;
    
    for (int i = 0; i < pts0.size(); i++) {
        Point2d prevPt = pts0[i];
        
        //for all points detected in the current frame, find all candidate matches
        vector<int> candMatchIdx;
        vector<double> candMatchDist;
        for (int j = 0; j < pts1.size(); j++) {
            Point2d currPt = pts1[j];
            double dSq = (prevPt.x - currPt.x)*(prevPt.x - currPt.x) + (prevPt.y - currPt.y)*(prevPt.y - currPt.y);
            //check that candidate match is within allowed motion window
            if ((dSq < maxDistanceSq) && (dSq > minDistanceSq)) {
                //calculate hamming distance and check that is below the threshold and better than any other existing matches
                double ham = norm(desc0.row(i),desc1.row(j),NORM_HAMMING);
                if ((ham < _maxHammingDistance) && ((ham < matchDistance[j]) || (matchDistance[j] == -1))) {
                    candMatchIdx.push_back(j);
                    candMatchDist.push_back(ham);
                }
            }
        }
        
        if (candMatchIdx.size() != 0) {
            int idx = -1;
            if (candMatchIdx.size() > 1) {
                //sort distances and do ratio test
                vector<size_t> orgIdx = VectorUtils::sort_indexes(candMatchDist);
                size_t topMatchIdx = orgIdx[0];
                size_t secondMatchIdx = orgIdx[1];
                bool ratioTest = ((candMatchDist[topMatchIdx]/candMatchDist[secondMatchIdx]) < _ratioTest);
                if (ratioTest)
                    idx = candMatchIdx[topMatchIdx];
            } else
                idx = candMatchIdx[0];
            
            //find best candidate and save
            if (idx != -1) {
                if (matchDistance[idx] == -1) {
                    matchIdx0.push_back(i);
                    matchIdx1.push_back(idx);
                    matchedIdx[idx] = matchCount;
                    matchCount++;
                } else {
                    int oldIdx = matchedIdx[idx];
                    matchIdx0[oldIdx] = i;
                }
                matchDistance[idx] = candMatchDist[0];
            }
        }
    }
}

void CTracker::matchFeatures(const vector<Point2d> &pts0, const Mat &desc0, const vector<Point2d> &pts1, const Mat &desc1, vector<int> &matchIdx0, vector<int> &matchIdx1, double minDistance, double maxDistance) {
    vector<vector<DMatch>> matches;
    
    _matcher.knnMatch(desc0, desc1, matches, 2);
    vector<double> matchDistance(pts1.size(), -1);
    vector<int> matchedIdx(pts1.size(), -1);
    
    double minDistanceSq = minDistance*minDistance;
    double maxDistanceSq = maxDistance*maxDistance;
    int matchCount = 0;
    for (int i = 0; i < matches.size(); i++) {
        int prevIdx = matches[i][0].queryIdx;
        int currIdx = matches[i][0].trainIdx;
        double ratio = matches[i][0].distance/matches[i][1].distance;
        double d = (pts0[prevIdx].x - pts1[currIdx].x)*(pts0[prevIdx].x - pts1[currIdx].x) + (pts0[prevIdx].y - pts1[currIdx].y)*(pts0[prevIdx].y - pts1[currIdx].y);
        
        
        bool minDist = (d > minDistanceSq);
        bool maxDist = (d < maxDistanceSq);
        bool crossRatio = (ratio < _ratioTest);
        bool newMatch = (matchDistance[currIdx] == -1);
        bool betterMatch = (matches[i][0].distance < matchDistance[currIdx]);
        
        
        if (minDist && maxDist && crossRatio && (newMatch || betterMatch)) {
            //matches are not 1-1
            if (newMatch) {
                matchIdx0.push_back(prevIdx);
                matchIdx1.push_back(currIdx);
                matchedIdx[currIdx] = matchCount;
                matchCount++;
            }
            else {
                int oldIdx = matchedIdx[currIdx];
                matchIdx0[oldIdx] = prevIdx;
            }
            matchDistance[currIdx] = matches[i][0].distance;
        }
    }
}

bool CTracker::detectFeaturesOpticalFlow() {
    int maxFeats = 500;
    double qualityLvl = 0.05;
    double minDistance = 10;
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size subPixWinSize(5,5);
    vector<Point2f> pts;
    
    //detect corners
    goodFeaturesToTrack(_currFrame.getFrameGrey(), pts, maxFeats, qualityLvl, minDistance);
    
    //subpixel refinement
    cornerSubPix(_currFrame.getFrameGrey(), pts, subPixWinSize, Size(-1,-1), termcrit);
    
    if (pts.size() < _minFeatures)
        return false;
    
    _currFrame.setPoints(pts);
    return true;
    
}

//detect BRISK features. Detector and descriptor are initialised in the constructor
bool CTracker::detectFeatures() {
    vector<KeyPoint> kp;
    Mat desc;
    
    _detector->detect(_currFrame.getFrameGrey(), kp);
    
    if (kp.size() < _minFeatures)
        return false;
    
    _descriptor->compute(_currFrame.getFrameGrey(), kp, desc);
    _currFrame.setKeyPoints(kp,desc);
    return true;
}

bool CTracker::matchFeaturesRadius() {
    //reset structures
    _currMatch.clear();
    _prevMatch.clear();
    _currIdx.clear();
    _prevIdx.clear();
    vector<vector<DMatch>> matches;
    
    //get keypoints
    vector<Point2d> prevPts, currPts;
    Mat prevDesc, currDesc;
    Mat(_prevFrame.getPointsDistorted()).copyTo(prevPts);
    Mat(_currFrame.getPointsDistorted()).copyTo(currPts);
    prevDesc = _prevFrame.getDescriptors();
    currDesc = _currFrame.getDescriptors();
    
    //match candidates within radius
    int matchCount = 0;
    vector<double> matchDistance(currPts.size(), -1);
    vector<int> matchedIdx(currPts.size(),-1);
    for (int i = 0; i < prevPts.size(); i++) {
        Point2d prevPt = prevPts[i];
        
        //for all points detected in the current frame, find all candidate matches
        vector<int> candMatchIdx;
        vector<double> candMatchDist;
        for (int j = 0; j < currPts.size(); j++) {
            Point2d currPt = currPts[j];
            double dSq = (prevPt.x - currPt.x)*(prevPt.x - currPt.x) + (prevPt.y - currPt.y)*(prevPt.y - currPt.y);
            //check that candidate match is within allowed motion window
            if ((dSq < _maxMatchDistanceSq) && (dSq > _minMatchDistanceSq)) {
                //calculate hamming distance and check that is below the threshold and better than any other existing matches
                double ham = norm(prevDesc.row(i),currDesc.row(j),NORM_HAMMING);
                if ((ham < _maxHammingDistance) && ((ham < matchDistance[j]) || (matchDistance[j] == -1))) {
                    candMatchIdx.push_back(j);
                    candMatchDist.push_back(ham);
                }
            }
        }
        
        if (candMatchIdx.size() != 0) {
            int idx = -1;
            if (candMatchIdx.size() > 1) {
                //sort distances and do ratio test
                vector<size_t> orgIdx = VectorUtils::sort_indexes(candMatchDist);
                size_t topMatchIdx = orgIdx[0];
                size_t secondMatchIdx = orgIdx[1];
                bool ratioTest = ((candMatchDist[topMatchIdx]/candMatchDist[secondMatchIdx]) < _ratioTest);
                if (ratioTest)
                    idx = candMatchIdx[topMatchIdx];
            } else
                idx = candMatchIdx[0];
            
            //find best candidate and save
            if (idx != -1) {
                if (matchDistance[idx] == -1) {
                    _prevIdx.push_back(i);
                    _currIdx.push_back(idx);
                    matchedIdx[idx] = matchCount;
                    matchCount++;
                } else {
                    int oldIdx = matchedIdx[idx];
                    _prevIdx[oldIdx] = i;
                }
                matchDistance[idx] = candMatchDist[0];
            }
        }
    }
    
    //save undistorted matches from indices
    _prevFrame.getPointsAt(_prevIdx, _prevMatch);
    _currFrame.getPointsAt(_currIdx, _currMatch);
    
    if (matchCount >= _minFeatures)
        return true;
    
    return false;
}

void CTracker::matchFeatures(const vector<int> &prevFrameIdx, const vector<int> &currFrameIdx, vector<int> &prevMatchIdx, vector<int> &currMatchIdx) {
    
    vector<vector<DMatch>> matches;
    
    //get keypoints
    vector<Point2d> prevPts, currPts;
    _prevFrame.getPointsAt(prevFrameIdx, prevPts);
    _currFrame.getPointsAt(currFrameIdx, currPts);
    
    //match closest 2 candidates
    Mat prevDesc, currDesc;
    _prevFrame.getDescriptorsAt(prevFrameIdx, prevDesc);
    _currFrame.getDescriptorsAt(currFrameIdx, currDesc);
    _matcher.knnMatch(prevDesc, currDesc, matches, 2);
    vector<double> matchDistance(currPts.size(), -1);
    vector<int> matchedIdx(currPts.size(),-1);
    
    
    //TODO: worth doing Fwd/Bwd test?
    prevMatchIdx.reserve(matches.size());
    currMatchIdx.reserve(matches.size());
    int matchCount = 0;
    for (int i = 0; i < matches.size(); i++) {
        int prevIdx = matches[i][0].queryIdx;
        int currIdx = matches[i][0].trainIdx;
        double ratio = matches[i][0].distance/matches[i][1].distance;
        double d = (prevPts[prevIdx].x - currPts[currIdx].x)*(prevPts[prevIdx].x - currPts[currIdx].x) + (prevPts[prevIdx].y - currPts[currIdx].y)*(prevPts[prevIdx].y - currPts[currIdx].y);
        
        bool minDist = (d > _minMatchDistanceSq);
        bool maxDist = (d < _maxMatchDistanceSq);
        bool crossRatio = (ratio < _ratioTest);
        bool newMatch = (matchDistance[currIdx] == -1);
        bool betterMatch = (matches[i][0].distance < matchDistance[currIdx]);
        
        if ( minDist && maxDist && crossRatio && ( newMatch || betterMatch)) {
            //matches are not 1-1
            if (newMatch) {
                prevMatchIdx.push_back(prevFrameIdx[prevIdx]);
                currMatchIdx.push_back(currFrameIdx[currIdx]);
                matchedIdx[currIdx] = matchCount;
                matchCount++;
            }
            else {
                int oldIdx = matchedIdx[currIdx];
                prevMatchIdx[oldIdx] = prevFrameIdx[prevIdx];
            }
            matchDistance[currIdx] = matches[i][0].distance;
        }
    }
}

bool CTracker::matchFeatures() {
    //reset structures
    _currMatch.clear();
    _prevMatch.clear();
    _currIdx.clear();
    _prevIdx.clear();
    vector<vector<DMatch>> matches;
    
    //get keypoints
    vector<Point2d> prevPts, currPts;
    prevPts = _prevFrame.getPointsDistorted();
    currPts = _currFrame.getPointsDistorted();
    
    //match closest 2 candidates
    _matcher.knnMatch(_prevFrame.getDescriptors(), _currFrame.getDescriptors(), matches, 2);
    vector<double> matchDistance(currPts.size(), -1);
    vector<int> matchedIdx(currPts.size(),-1);
    
    //TODO: worth doing Fwd/Bwd test?
    _prevIdx.reserve(matches.size());
    _currIdx.reserve(matches.size());
    int matchCount = 0;
    for (int i = 0; i < matches.size(); i++) {
        int prevIdx = matches[i][0].queryIdx;
        int currIdx = matches[i][0].trainIdx;
        double ratio = matches[i][0].distance/matches[i][1].distance;
        double d = (prevPts[prevIdx].x - currPts[currIdx].x)*(prevPts[prevIdx].x - currPts[currIdx].x) + (prevPts[prevIdx].y - currPts[currIdx].y)*(prevPts[prevIdx].y - currPts[currIdx].y);
        
        bool minDist = (d > _minMatchDistanceSq);
        bool maxDist = (d < _maxMatchDistanceSq);
        bool crossRatio = (ratio < _ratioTest);
        bool newMatch = (matchDistance[currIdx] == -1);
        bool betterMatch = (matches[i][0].distance < matchDistance[currIdx]);
        
        if ( minDist && maxDist && crossRatio && ( newMatch || betterMatch)) {
            //matches are not 1-1
            if (newMatch) {
                _prevIdx.push_back(prevIdx);
                _currIdx.push_back(currIdx);
                matchedIdx[currIdx] = matchCount;
                matchCount++;
            }
            else {
                int oldIdx = matchedIdx[currIdx];
                _prevIdx[oldIdx] = prevIdx;
            }
            matchDistance[currIdx] = matches[i][0].distance;
        }
    }
    
    //save undistorted matches from indices
    _prevFrame.getPointsAt(_prevIdx, _prevMatch);
    _currFrame.getPointsAt(_currIdx, _currMatch);
    
    if (matchCount >= _minFeatures)
        return true;
    
    return false;
}

//compute optical flow from previous keyframe
bool CTracker::computeOpticalFlow() {
    
    //parameters
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size winSize(21,21), subPixWinSize(8,8);
    vector<uchar> status;
    vector<float> err;
    int maxLevel = 3;
    
    //reset structures
    _currMatch.clear();
    _prevMatch.clear();
    _currIdx.clear();
    _prevIdx.clear();

    //structures
    vector<Point2f> currPts, prevPts;
    Mat(_prevFrame.getPointsDistorted()).copyTo(prevPts);
    vector<Point2f> currMatch, currDetectedPoints;
    Mat(_currFrame.getPointsDistorted()).copyTo(currDetectedPoints);
    vector<double> matchDistance(currDetectedPoints.size(), -1);
    vector<bool> matchStatus(currDetectedPoints.size());
    vector<int> matchedIdx(currDetectedPoints.size(), -1);
    
    //predict motion of points
    //predictFlow(currPts);
    
    //compute LK optical flow
    int matchCount = 0;
    double maxDistSq = _maxMatchDistance*_maxMatchDistance;
    double maxFeatDistSq = _maxOrgFeatDist*_maxOrgFeatDist;
    double minDistSq = _minMatchDistance*_minMatchDistance;
    //   if (_prevFrame.getNPoints() > 0) {
    calcOpticalFlowPyrLK(_prevFrame.getFrameGrey(), _currFrame.getFrameGrey(), prevPts, currPts, status, err, winSize, maxLevel, termcrit, 0, 0.001 );
    //check that PyrLK does not play tricks on us
    assert(prevPts.size() == currPts.size());
    
    //check matches
    for (int i = 0; i < status.size(); i++) {
        if (status[i]) {
            //look for actual detected point in other image
            int idx = _currFrame.findClosestPointIndexDistorted(currPts[i]);
            float e = (currPts[i].x - currDetectedPoints[idx].x)*(currPts[i].x - currDetectedPoints[idx].x) + (currPts[i].y - currDetectedPoints[idx].y)*(currPts[i].y - currDetectedPoints[idx].y);
            float d = (prevPts[i].x - currPts[i].x)*(prevPts[i].x - currPts[i].x) + (prevPts[i].y - currPts[i].y)*(prevPts[i].y - currPts[i].y);
            
            if ((d < maxDistSq) && (e < maxFeatDistSq) && (d > minDistSq) && ((matchDistance[idx] >= e) || (matchDistance[idx] == -1)) ) {
                
                //check if point was already matched
                if (matchStatus[idx] == true) {
                    //overwrite previous match since the one found is better
                    const int oldIdx = matchedIdx[idx];
                    _prevIdx[oldIdx] = i;
                } else {
                    //save indices in CFrame arrays
                    _prevIdx.push_back(i);
                    _currIdx.push_back(idx);
                    
                    matchStatus[idx] = true;
                    matchedIdx[idx] = matchCount;
                    matchCount++;
                }
                
                matchDistance[idx] = e;
            }
        }
    }
    
    //save undistorted matches from indices
    _prevFrame.getPointsAt(_prevIdx, _prevMatch);
    _currFrame.getPointsAt(_currIdx, _currMatch);
    
    
#ifdef DEBUGINFO
    cout << "# Matches: " << sum(_matchStatus)[0] << endl;
#endif
    
    //   }
    
    if (matchCount >= _minFeatures)
        return true;
    
    return false;
}

//--------------------------------------//
//                                      //
//                                      //
//          BUNDLE ADJUSTMENT           //
//                                      //
//                                      //
//--------------------------------------//
void CTracker::initialiseBAOptions() {
    
    //create solution options
    _opts.linear_solver_type = ceres::DENSE_SCHUR;
    _opts.minimizer_progress_to_stdout = false;
    _opts.logging_type = ceres::LoggingType::SILENT;
}

CTracker::BAStructAndPoseFunctor::BAStructAndPoseFunctor(double pt2d_x, double pt2d_y, const double *k) {
    
    _pt2d_x = pt2d_x;
    _pt2d_y = pt2d_y;
    _k = k;
}
template <typename T> bool CTracker::BAStructAndPoseFunctor::operator()(const T *const R, const T *const t, const T *const point, T *residuals) const {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(R, point, p);
        // camera[3,4,5] are the translation.
        p[0] += t[0]; p[1] += t[1]; p[2] += t[2];
        
        //project
        T xp =  p[0] / p[2];
        T yp =  p[1] / p[2];
        
        //intrinsics
        T predicted_x = T(_k[0])*xp + T(_k[1])*yp + T(_k[2]);
        T predicted_y = T(_k[4])*yp + T(_k[5]);
        
        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(_pt2d_x);
        residuals[1] = predicted_y - T(_pt2d_y);
        return true;
}


CTracker::BAPoseFunctor::BAPoseFunctor(double pt2d_x, double pt2d_y, const double *pt3d, const double *k) {
    
    _pt2d_x = pt2d_x;
    _pt2d_y = pt2d_y;
    _pt3d = pt3d;
    _k = k;
}

template <typename T> bool CTracker::BAPoseFunctor::operator()(const T *const R, const T *const t, T *residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    T M[3];
    M[0] = T(_pt3d[0]); M[1] = T(_pt3d[1]); M[2] = T(_pt3d[2]);
    ceres::AngleAxisRotatePoint(R, M, p);
    // camera[3,4,5] are the translation.
    p[0] += t[0]; p[1] += t[1]; p[2] += t[2];
    
    //project
    T xp =  p[0] / p[2];
    T yp =  p[1] / p[2];
    
    //intrinsics
    T predicted_x = T(_k[0])*xp + T(_k[1])*yp + T(_k[2]);
    T predicted_y = T(_k[4])*yp + T(_k[5]);
    
    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(_pt2d_x);
    residuals[1] = predicted_y - T(_pt2d_y);
    return true;
}

CTracker::BAStructFunctor::BAStructFunctor(double pt2d_x, double pt2d_y, const double *R, const double *t, const double *k) {
    
    _pt2d_x = pt2d_x;
    _pt2d_y = pt2d_y;
    _R = R;
    _t = t;
    _k = k;
}

template <typename T> bool CTracker::BAStructFunctor::operator()(const T *const pt3d, T *residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    T rot[3];
    rot[0] = T(_R[0]); rot[1] = T(_R[1]); rot[2] = T(_R[2]);
    ceres::AngleAxisRotatePoint(rot, pt3d, p);
    // camera[3,4,5] are the translation.
    p[0] += T(_t[0]); p[1] += T(_t[1]); p[2] += T(_t[2]);
    
    //project
    T xp =  p[0] / p[2];
    T yp =  p[1] / p[2];
    
    //intrinsics
    T predicted_x = T(_k[0])*xp + T(_k[1])*yp + T(_k[2]);
    T predicted_y = T(_k[4])*yp + T(_k[5]);
    
    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(_pt2d_x);
    residuals[1] = predicted_y - T(_pt2d_y);
    return true;
}

void CTracker::bundleAdjustmentStructAndPose(const vector<Point2d> &observations, const vector<int> &camIdx, const vector<Matx33d> &K, vector<double*> &R, vector<double*> &t, vector<double*> &pts3D, int isStructOrPose) {
    //create ceres problem
    ceres::Problem prob;
    
    //populate problem
    ceres::CostFunction* cost_function;
    for (int i = 0; i < observations.size(); i++) {
        int camNo = camIdx[i];
        
        switch (isStructOrPose) {
            case 0:
                cost_function = CTracker::BAStructFunctor::Create(observations[i].x, observations[i].y, R[camNo], t[camNo], K[camNo].val);
                prob.AddResidualBlock(cost_function, NULL, pts3D[i]);
                break;
            case 1:
                cost_function = CTracker::BAPoseFunctor::Create(observations[i].x, observations[i].y, pts3D[i], K[camNo].val);
                prob.AddResidualBlock(cost_function, NULL, R[camNo], t[camNo]);
                break;
            case 2:
                cost_function = CTracker::BAStructAndPoseFunctor::Create(observations[i].x, observations[i].y, K[camNo].val);
                prob.AddResidualBlock(cost_function, NULL, R[camNo], t[camNo], pts3D[i]);
                break;
            default:
                break;
        }
        
    }
    
    
    //solve BA
    ceres::Solver::Summary summ;
    ceres::Solve(_opts, &prob, &summ);
}
