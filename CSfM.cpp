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

#include "CSfM.h"

CSfM::CSfM(const Matx33d &K, const Size &imSize, const vector<double> &d) : _tracker(K,d,imSize)
{
    _state = NOT_INITIALIZED;
    
 
    _maxReprErr = 7; //maximum average reprojection/transfer error

    _lostCount = 0;
    _minCovisibilityStrength = 0; //how many points are covisible between frames for bundle adjustment to proceed
    _maxLost = 10; //how many frames can be processed before declaring loss of track
    _frameCount = 0; //frame counter
    _motionHistoryLength = 2; //how many frames should you remember the pose for depending on how complicated the motion model is

    
    //time threshold before adding new keyframe
    _newKFrameTimeLag = 10;
    
    //number of feature threshold to consider connected to frames with covisible points
    _covisibilityThreshold = 50;
    
    //minimum number of keyframes that a map point need to visible in before getting culled
    _minVisibilityFrameNo = 3;
    
    //signals
    _keyFrameAdded = false;
}

CSfM::~CSfM() {
}

void CSfM::addFrame(cv::Mat frameIn) {
#ifdef DEBUGINFO
    cout << "Number of points in map: " << _mapper.getNPoints() << endl;
#endif
    
    //set frame
    _tracker.setCurrentFrame(frameIn,_frameCount);
    //_currFrame.setFrame(frameIn,_frameCount);
    
    bool success;
    int nCulledPts;
    
    switch (_state) {
        case NOT_INITIALIZED:
            success = init();
            break;
        case RUNNING:
            success = tracking();
            if (_keyFrameAdded) {
                mapping();
            }
            break;
        case LOST:
            success = recovery();
            break;
        default:
            break;
    }

    //debug show points
    vector<Matx31d> showPts3D, allPts3D; vector<int> showPts3DIdx;
    _tracker._prevFrame.getMatchedPoints(showPts3DIdx);
    _mapper.getPointsAtIdx(showPts3DIdx, showPts3D);
    _mapper.getPoints(allPts3D);
    Mat dShow = Display2D::display3DProjections(_tracker._prevFrame.getFrameGrey(), _tracker._prevFrame.getIntrinsicUndistorted(), _tracker._prevFrame.getRotation(), _tracker._prevFrame.getTranslation(), allPts3D, 3, Scalar(0,0,255),1);
    dShow = Display2D::display3DProjections(dShow, _tracker._prevFrame.getIntrinsicUndistorted(), _tracker._prevFrame.getRotation(), _tracker._prevFrame.getTranslation(), showPts3D, 3, Scalar(0,255,0));
    //_vOut << dShow;
    imshow("Debug",dShow);
    waitKey(1);

    _frameCount++;
}

bool CSfM::mapping() {
    bool success = false;
    
    // ORBSLAM 6.A.
    //Edge graph is already updated automatically in tracking(). Calculate BoW
    
    // ORBSLAM 6.C.
    // Map point creation
    
    // match current keyframe against all other connected keyframes
    // get connected keyframes
    vector<int> connectedKFNo;
    int lastKFIdx = _kFrames.size() -1;
    int lastKFNo = _kFrames[lastKFIdx].getFrameNo();
    _mapper.getFramesConnectedToFrame(lastKFNo, connectedKFNo, _covisibilityThreshold);
    
    //matching against connected keyframes
    int count = 0;
    for (int i = 0; i < connectedKFNo.size(); i++) {
        //get unmatched points for the last keyframe added and the one connected. do it on each loop iteration since we are constantly matching some of the points
        int kfIdx = _FrameNoTokFrameIdx[connectedKFNo[i]];
        Mat currUnmatchedDesc, prevUnmatchedDesc;
        vector<int> currUnmatchedPtsIdx2D, prevUnmatchedPtsIdx2D;
        vector<Point2d> currUnmatchedPts2D, prevUnmatchedPts2D;
        Matx34d P0, P1;
        Matx33d K0, K1;
        _kFrames[lastKFIdx].getUnmatchedPoints(currUnmatchedPts2D, currUnmatchedDesc, currUnmatchedPtsIdx2D);
        _kFrames[kfIdx].getUnmatchedPoints(prevUnmatchedPts2D, prevUnmatchedDesc, prevUnmatchedPtsIdx2D);
        P0 = _kFrames[kfIdx].getProjectionMatrix(); K0 = _kFrames[kfIdx].getIntrinsicUndistorted();
        P1 = _kFrames[lastKFIdx].getProjectionMatrix(); K1 = _kFrames[lastKFIdx].getIntrinsicUndistorted();
        
        //match points
        vector<int> prevMatchIdx, currMatchIdx;
        vector<Point2d> prevMatchPts2D, currMatchPts2D;
        _tracker.matchFeatures(prevUnmatchedPts2D, prevUnmatchedDesc, currUnmatchedPts2D, currUnmatchedDesc, prevMatchIdx, currMatchIdx);
        prevMatchPts2D.reserve(prevMatchIdx.size());
        currMatchPts2D.reserve(currMatchIdx.size());
        for (int j = 0; j < prevMatchIdx.size(); j++) {
            prevMatchPts2D.push_back(prevUnmatchedPts2D[prevMatchIdx[j]]);
            currMatchPts2D.push_back(currUnmatchedPts2D[currMatchIdx[j]]);
        }
        
        //triangulation
        vector<Matx31d> matchPts3D;
        GeometryUtils::triangulatePoints(P0, P1, K0, K1, prevMatchPts2D, currMatchPts2D, matchPts3D);
        if (matchPts3D.size() > 0) {
            //filter matches based on epipolar line distance and positive depth
            Matx33d F;
            vector<uchar> status;
            vector<int> prevFilt2DIdx, currFilt2DIdx;
            vector<Point2d> prevFilt2DPts, currFilt2DPts;
            vector<Matx31d> filtPts3D;
            GeometryUtils::calculateFundamentalMatrix(K0, _kFrames[kfIdx].getRotation(), _kFrames[kfIdx].getTranslation(), K1, _kFrames[lastKFIdx].getRotation(), _kFrames[lastKFIdx].getTranslation(), F);
            GeometryUtils::filterMatches(F, prevMatchPts2D, currMatchPts2D, matchPts3D, status, _maxReprErr);
            for (int j = 0; j < status.size(); j++) {
                if (status[j] == 1) {
                    int prevIdx = prevMatchIdx[j];
                    int currIdx = currMatchIdx[j];
                    prevFilt2DIdx.push_back(prevUnmatchedPtsIdx2D[prevIdx]);
                    currFilt2DIdx.push_back(currUnmatchedPtsIdx2D[currIdx]);
                    prevFilt2DPts.push_back(prevUnmatchedPts2D[prevIdx]);
                    currFilt2DPts.push_back(currUnmatchedPts2D[currIdx]);
                    filtPts3D.push_back(matchPts3D[i]);
                    }
                }

            //update map and frames
            vector<int> filtPts3DIdx;
            Mat pDesc, cDesc;
            _mapper.addNewPoints(filtPts3D, {prevFilt2DIdx}, {connectedKFNo[i]}, filtPts3DIdx);
            _kFrames[kfIdx].updatePoints(prevFilt2DIdx, filtPts3DIdx);
            _kFrames[kfIdx].getDescriptorsAt(prevFilt2DIdx, pDesc);
            _kFrames[lastKFIdx].getDescriptorsAt(currFilt2DIdx, cDesc);
            _mapper.addDescriptors(filtPts3DIdx, pDesc);
            
        
            count += filtPts3D.size();
            
            //reproject in other keyframes and check for matches
            for (int j = 0; j < connectedKFNo.size(); j++) {
                if (i != j) {
                
                    int kfIdx_j = _FrameNoTokFrameIdx[connectedKFNo[j]];
                
                    //get unmatched points in current frame
                    vector<int> unmatchedPts2DIdx;
                    vector<Point2d> unmatchedPts2D;
                    Mat unmatchedDesc;
                    _kFrames[kfIdx_j].getUnmatchedPoints(unmatchedPts2D, unmatchedPts2DIdx);
                    _kFrames[kfIdx_j].getDescriptorsAt(unmatchedPts2DIdx, unmatchedDesc);
                
                    //match descriptors against closest keyframe
                    vector<Point2d> reprPts2D;
                    GeometryUtils::projectPoints(_kFrames[kfIdx_j].getProjectionMatrix(), _kFrames[kfIdx_j].getIntrinsicUndistorted(), filtPts3D, reprPts2D);
                
                    vector<int> matchIdx0, matchIdx1;
                    if (abs(connectedKFNo[j] - connectedKFNo[i]) < abs(connectedKFNo[j] - lastKFNo))
                        _tracker.matchFeatures(reprPts2D, pDesc, unmatchedPts2D, unmatchedDesc, matchIdx0, matchIdx1, 0, _maxReprErr);
                    else
                        _tracker.matchFeatures(reprPts2D, cDesc, unmatchedPts2D, unmatchedDesc, matchIdx0, matchIdx1, 0, _maxReprErr);
                
                    //get matched 3d points index and update frame
                    vector<int> newPts3DIdx, newPts2DIdx;
                    newPts3DIdx.reserve(matchIdx0.size());
                    newPts2DIdx.reserve(matchIdx0.size());
                    for (int k = 0; k < matchIdx0.size(); k++) {
                        newPts3DIdx.push_back(filtPts3DIdx[matchIdx0[k]]);
                        newPts2DIdx.push_back(unmatchedPts2DIdx[matchIdx1[k]]);
                    }
                    updateMapAndFrame(connectedKFNo[j], newPts3DIdx, newPts2DIdx);
                }
            }
            //update
            _mapper.addPointMatches(filtPts3DIdx, currFilt2DIdx, lastKFNo);
            _kFrames[lastKFIdx].updatePoints(currFilt2DIdx, filtPts3DIdx);
            _mapper.addDescriptors(filtPts3DIdx, cDesc);
         }
    }
    
#ifdef DEBUGINFO
    cout << "(MAPPER) Found matches for " << count << " new map points" << endl;
#endif
    
    
    // ORBSLAM 6.B.
    // Recent map points culling
    int cullMapCount = cullMapPoints();
    
#ifdef DEBUGINFO
    cout << "(MAPPER) Culled " << cullMapCount << " map points" << endl;
#endif
    
    // ORBSLAM 6.D.
    // Local BA
    connectedKFNo.push_back(lastKFNo);
    bundleAdjustment(connectedKFNo,CTracker::BA_TYPE::POSE_ONLY);
    
    // ORBSLAM 6.E.
    // Local KF culling
    int cullFrameCount = cullKeyFrames();
#ifdef DEBUGINFO
    cout << "(MAPPER) Culled " << cullFrameCount << " key frames" << endl;
#endif
    
    _mapper.incrementMapAge(0,1);
    _keyFrameAdded = false;
    
    return success;
}

void CSfM::bundleAdjustment(const vector<int> &frameIdx, int isStructAndPose) {
    vector<double*> R;
    vector<double*> t;
    vector<double*> pts3d;
    vector<Point2d> pts2d;
    vector<int> camIdx;
    vector<Matx33d> K;
    
    vector<int> org3dIdx;
    
    
    for (int i = 0; i < frameIdx.size(); i++) {
        vector<int> pts2dIdx;
        int currIdx = frameIdx[i];
        int kFrameIdx = _FrameNoTokFrameIdx[currIdx];
        //get intrinsic matrix
        K.push_back(_kFrames[kFrameIdx].getIntrinsicUndistorted());
        //get the pose vectors
        t.push_back(_kFrames[kFrameIdx].getTranslation_Mutable());
        R.push_back(_kFrames[kFrameIdx].getRotationRodrigues_Mutable());
        //get the 3d points
        _mapper.getPointsInFrame_Mutable(pts3d, pts2dIdx, currIdx);
        
        //get the corresponding 2d points
        _kFrames[kFrameIdx].getPointsAt(pts2dIdx, pts2d);
        //get camera indices for the points added
        int prevSize = camIdx.size();
        camIdx.reserve(pts3d.size());
        for (int j = 0; j < pts3d.size() - prevSize; j++) {
            camIdx.push_back(i);
        }
    }
    
    //run bundle adjustment
    _tracker.bundleAdjustmentStructAndPose(pts2d, camIdx, K, R, t, pts3d, isStructAndPose);
    
    //update projection matrices
    for (int i = 0; i < frameIdx.size(); i++) {
        int kFrameIdx = _FrameNoTokFrameIdx[frameIdx[i]];
        _kFrames[kFrameIdx].calculateProjectionMatrix();
    }
}

void CSfM::filterMatches(const vector<uchar> &status) {
    //filter matches according to the status mask from findhomography or findfundamental
    vector<Point2f> tempPrevMatch, tempCurrMatch;
    vector<int> tempPrevIdx, tempCurrIdx;
    
    tempPrevMatch.reserve(status.size());
    tempCurrMatch.reserve(status.size());
    tempPrevIdx.reserve(status.size());
    tempCurrIdx.reserve(status.size());
    for (int i = 0; i < status.size(); i++) {
        if (status[i] == 1) {
            tempPrevMatch.push_back(_tracker._prevMatch[i]);
            tempCurrMatch.push_back(_tracker._currMatch[i]);
            tempPrevIdx.push_back(_tracker._prevIdx[i]);
            tempCurrIdx.push_back(_tracker._currIdx[i]);
        }
    }
    //swap buffers
    swap(_tracker._prevMatch,tempPrevMatch);
    swap(_tracker._currMatch,tempCurrMatch);
    swap(_tracker._prevIdx,tempPrevIdx);
    swap(_tracker._currIdx,tempCurrIdx);
}

void CSfM::predictFlow(vector<Point2f> &predPts) {
    //right now do nothing, initialise prediction with current coordinates
}

float CSfM::calculateHomographyScore(const vector<Point2f> &pts0, const vector<Point2f> &pts1, const Matx33f &H, const vector<uchar> &status, const float Th, const float Gamma) {
    float s = 0;
    
    //compute matrix inverse
    Matx33f Hinv = H.inv();
    
    //get forward and backward transformed points
    vector<Point2f> fwd, bwd;
    perspectiveTransform(pts0, fwd, H);
    perspectiveTransform(pts1, bwd, Hinv);
    
    //score calculated inside loop
    float e01, e10, gamma0, gamma1;
    int count = 0;
    for (int i = 0; i < pts0.size(); i++) {
        if (status[i] == 1) {
            e01 = (pts1[i].x - fwd[i].x)*(pts1[i].x - fwd[i].x) + (pts1[i].y - fwd[i].y)*(pts1[i].y - fwd[i].y);
            e10 = (pts0[i].x - bwd[i].x)*(pts0[i].x - bwd[i].x) + (pts0[i].y - bwd[i].y)*(pts0[i].y - bwd[i].y);
            gamma0 = (e01 < Th) ? Gamma - e01 : 0;
            gamma1 = (e10 < Th) ? Gamma - e10 : 0;
            s += gamma0 + gamma1;
            count++;
        }
    }
    
    return s/count;
}

float CSfM::calculateFundamentalScore(const vector<Point2f> &pts0, const vector<Point2f> &pts1, const Matx33f &F, const vector<uchar> &status, const float Tf, const float Gamma) {
    float s = 0;
    
    //compute epipolar lines
    vector<Vec3f> epiLines0, epiLines1;
    computeCorrespondEpilines(pts0, 1, F, epiLines1);
    computeCorrespondEpilines(pts1, 2, F, epiLines0);
    
    float e01,e10,gamma0,gamma1;
    int count = 0;
    for (int i = 0; i < pts0.size(); i++) {
        if (status[i] == 1) {
            //compute distance from epilines
            e01 = GeometryUtils::distancePointLine2D(pts0[i], epiLines0[i]);
            e10 = GeometryUtils::distancePointLine2D(pts1[i], epiLines1[i]);
        
            //compute gamma
            gamma0 = (e01 < Tf) ? Gamma - e01 : 0;
            gamma1 = (e10 < Tf) ? Gamma - e10 : 0;
        
            s += gamma0 + gamma1;
            count++;
        }
    }
    
    return s/count;
}

void CSfM::updateMotionHistory(const Matx33d &R, const Matx31d &t) {
    //FIFO queue
    if (_R.size() >= _motionHistoryLength) {
        _R.pop_back();
        _t.pop_back();
    }
    _R.push_back(R);
    _t.push_back(t);
}

bool CSfM::addKeyFrame() {
    //at least 20 frames from last keyframe added
    bool a = (_tracker._currFrame.getFrameNo() >= (_kFrames[_kFrames.size()-1].getFrameNo() + _newKFrameTimeLag));
    //current frame tracks at least 50 points
    bool b = (_tracker._currFrame.getNMatchedPoints() >= 50);
    
    
    //current frame tracks less than 90% of points than last keyframe OR
    //there is the potential for many more matches
    int nkfPts = _kFrames[_kFrames.size()-1].getNMatchedPoints();
    int ncurrPts = _tracker._currFrame.getNMatchedPoints();

    bool c = (ncurrPts < 0.9*nkfPts); 
    bool d = false;//(_tracker._currMatch.size() - ncurrPts > 100);
    
    bool addKeyFrame = (a && b && (c || d)  )  ;
    return addKeyFrame;
}

bool CSfM::tracking() {

    //find map points tracked in the last frame
    bool success = false;
    
    //(ORBSLAM 5.A)
    //detect features
    _tracker.detectFeatures();
    
    //(ORBSLAM 5.B)
    //match only features that have been recorded already
    vector<int> prevMatch2DIdx, prev2DIdx, curr2DIdx, currMatch2DIdx;
    _tracker._prevFrame.getMatchedIndices(prev2DIdx);
    int nCurrPts = _tracker._currFrame.getNPoints();
    curr2DIdx.reserve(nCurrPts);
    for (int i = 0; i < nCurrPts; i++) {
        curr2DIdx.push_back(i);
    }
    _tracker.matchFeatures(prev2DIdx, curr2DIdx, prevMatch2DIdx, currMatch2DIdx);
    

#ifdef DEBUGINFO
    cout << "(TRACKING) Found map points: " << currMatch2DIdx.size() << " out of " << _tracker._currIdx.size() << " matches" << endl;
#endif
    
    //check if we found enough points
    if (currMatch2DIdx.size() < _tracker._minFeatures) {
        //increase lost frame count and check if we are actually lost
        _lostCount++;
        
        //do not swap buffers, chances are this is a blurry frame. Keep on matching against the previous one
        
        if (_lostCount > _maxLost) {
            //(ORBSLAM 5.C)
            _state = LOST;
            waitKey();
        }
#ifdef DEBUGINFO
        cout << "(TRACKING) Lost track!" << endl;
#endif
    }
    else {
        //reset lost count
        _lostCount = 0;
        
        //get point match coordinates
        vector<Matx31d> currMatch3D; vector<Point2d> currMatch2D; vector<int> currMatch3DIdx;
        _tracker._prevFrame.getPoints3DIdxAt(prevMatch2DIdx, currMatch3DIdx);
        _mapper.getPointsAtIdx(currMatch3DIdx, currMatch3D);
        _tracker._currFrame.getPointsAt(currMatch2DIdx, currMatch2D);
        
        //solve PnP
        int iter = 20;
        double confidence = 0.99;
        double reprErr = _maxReprErr;
        Mat rvec = Mat::zeros(3,1,CV_64FC1);
        Mat tvec = Mat::zeros(3,1,CV_64FC1);
        
        vector<int> inlierIdx;
        solvePnPRansac(currMatch3D, currMatch2D, _tracker._currFrame.getIntrinsicUndistorted(), Mat::zeros(4,1,CV_64FC1), rvec, tvec, false, iter, reprErr, confidence, inlierIdx, SOLVEPNP_ITERATIVE);
        
        //update pose of current frame
        Mat R;
        Rodrigues(rvec,R);
        _tracker._currFrame.setPose(R,tvec);
        
        //filter outliers
        vector<int> filt2DIdx, filt3DIdx;
        filt2DIdx.reserve(inlierIdx.size()); filt3DIdx.reserve(inlierIdx.size());
        for (int i = 0; i < inlierIdx.size(); i++) {
            int idx = inlierIdx[i];
            filt2DIdx.push_back(currMatch2DIdx[idx]);
            filt3DIdx.push_back(currMatch3DIdx[idx]);
        }
        _tracker._currFrame.updatePoints(filt2DIdx, filt3DIdx);
        _mapper.updatePointViews(filt3DIdx);
        
#ifdef DEBUGINFO
        cout << "(TRACKING) PnP: removed " << currMatch2D.size() - inlierIdx.size() << " outliers" << endl;
#endif
        
        //(ORBSLAM 5.D)
        //increase number of matched points
        findMapPointsInCurrentFrame();
        
        //bundle adjustment - pose only for speed
//        vector<int> covisibleFrameIdx;
//        _mapper.getFramesConnectedToFrame(lastFrameNo, covisibleFrameIdx);
//        covisibleFrameIdx.push_back(lastFrameNo);
//        bundleAdjustment(covisibleFrameIdx, CTracker::BA_TYPE::POSE_ONLY);

        //(ORBSLAM 5.E)
        //decide if keyframe
        if (addKeyFrame()) {
#ifdef DEBUGINFO
            cout << "---- Keyframe added! ----" << endl;
#endif
            //add to keyframes pool
            int lastFrameNo = _tracker._currFrame.getFrameNo();
            _kFrames.push_back(CKeyFrame(_tracker._currFrame));
            int lastKFrameIdx = _kFrames.size()-1;
            _kFrameIdxToFrameNo[lastKFrameIdx] = lastFrameNo;
            _FrameNoTokFrameIdx[_tracker._currFrame.getFrameNo()] = lastKFrameIdx;
            
            //update points
            vector<int> curr3DPtsIdx, curr2DPtsIdx;
            Mat currDesc;
            _kFrames[lastKFrameIdx].getMatchedPoints(curr2DPtsIdx, curr3DPtsIdx);
            _mapper.addPointMatches(curr3DPtsIdx, curr2DPtsIdx, lastFrameNo);
            _kFrames[lastKFrameIdx].getDescriptorsAt(curr2DPtsIdx, currDesc);
            _mapper.addDescriptors(curr3DPtsIdx, currDesc);
            
            //update motion model
            updateMotionHistory(_kFrames[lastKFrameIdx].getRotation(), _kFrames[lastKFrameIdx].getTranslation());
            
            //send signal to mapper
            _keyFrameAdded = true;
        }
        else {
            //update motion model based on pnp
            updateMotionHistory(R, tvec);
        }
        
        //swap buffers between current and previous frame
        swap(_tracker._prevFrame,_tracker._currFrame);
    }
    
    _mapper.incrementMapAge(1,0);
    return success;
}

void CSfM::findMapPointsInCurrentFrame() {
    Matx34d P = _tracker._currFrame.getProjectionMatrix();
    Matx33d K = _tracker._currFrame.getIntrinsicUndistorted();
    
    //find keyframes connected in covisibility graph
    int refKno = _kFrames[_kFrames.size()-1].getFrameNo();
    vector<int> covisibleFrameIdx;
    _mapper.getFramesConnectedToFrame(refKno, covisibleFrameIdx, _covisibilityThreshold);
    covisibleFrameIdx.push_back(refKno);
    
    //get all unmatched 3d points visible from covisible frames
    vector<int> covisiblePts3DIdx, existingPts3DIdx;
    _mapper.getPointsInFrames(covisiblePts3DIdx, covisibleFrameIdx);
    _tracker._currFrame.getMatchedPoints(existingPts3DIdx);
    sort(covisiblePts3DIdx.begin(), covisiblePts3DIdx.end());
    sort(existingPts3DIdx.begin(), existingPts3DIdx.end());
    
    vector<int> newPts3DIdx(covisiblePts3DIdx.size());
    vector<int>::iterator it = set_difference(covisiblePts3DIdx.begin(),covisiblePts3DIdx.end(), existingPts3DIdx.begin(), existingPts3DIdx.end(), newPts3DIdx.begin());
    newPts3DIdx.resize(it - newPts3DIdx.begin());
    
    //get unmatched points in current frame
    vector<int> unmatchedPts2DIdx;
    vector<Point2d> unmatchedPts2D;
    Mat unmatchedDesc;
    _tracker._currFrame.getUnmatchedPoints(unmatchedPts2D, unmatchedPts2DIdx);
    _tracker._currFrame.getDescriptorsAt(unmatchedPts2DIdx, unmatchedDesc);
    
    //match descriptors
    vector<Matx31d> newPts3D;
    Mat newPtsDesc;
    vector<Point2d> newPts2D;
    _mapper.getPointsAtIdx(newPts3DIdx, newPts3D);
    _mapper.getRepresentativeDescriptors(newPts3DIdx, newPtsDesc);
    GeometryUtils::projectPoints(P, K, newPts3D, newPts2D);
    
    vector<int> matchMapIdx, matchFrameIdx;
    _tracker.matchFeatures(newPts2D, newPtsDesc, unmatchedPts2D, unmatchedDesc, matchMapIdx, matchFrameIdx, 0, _maxReprErr);
    
    //get matched 3d points index and update frame
    vector<int> match3DIdx, match2DIdx;
    match3DIdx.reserve(matchMapIdx.size());
    for (int i = 0; i < matchMapIdx.size(); i++) {
        int idx = matchMapIdx[i];
        match3DIdx.push_back(newPts3DIdx[idx]);
        idx = matchFrameIdx[i];
        match2DIdx.push_back(unmatchedPts2DIdx[idx]);
    }
    _tracker._currFrame.updatePoints(match2DIdx, match3DIdx);
    _mapper.updatePointViews(match3DIdx);
    
#ifdef DEBUGINFO
    cout << "(TRACKING) Found additional " << matchMapIdx.size() << " points from map" << endl;
#endif
}

int CSfM::cullMapPoints() {
    //Cull map points to guarantee robust, repeatable features
    
    //1. Cull points observed by fewer than 3 keyframes
    vector<int> cullIdx, newPtsIdx;
    _mapper.removePointsThreshold(cullIdx, newPtsIdx);
    if (cullIdx.size() > 0) {
        for (int i = 0; i < _kFrames.size(); i++) {
            _kFrames[i].cullPoints(newPtsIdx);
        }
        _tracker._prevFrame.cullPoints(newPtsIdx);
        _tracker._currFrame.cullPoints(newPtsIdx);
    }
    return cullIdx.size();
}

int CSfM::cullKeyFrames(double thresholdRate) {
    
    //find frames that have at least a thresholdRate% overlap in matched features with _minVisibilityFrameNo other frames
    int culledFrameNo = 0;
    
    //greedy approach, start from the oldest keyframe
    int kfIdx = 0;
    while (kfIdx < _kFrames.size()-1) {
        //get all the points found in current keyframe
        vector<int> pts3DIdx;
        _kFrames[kfIdx].getMatchedPoints(pts3DIdx);
        int frameNo = _kFrames[kfIdx].getFrameNo();
        
        //check if at least thresholdRate points have been matched by at least _minVisibilityFrameNo other keyframes
        bool cull = _mapper.arePointsSeenByAtLeast(pts3DIdx, _minVisibilityFrameNo, thresholdRate);
        
        //if so, cull the frame
        if (cull) {
            //remove frame references from the map
            _mapper.removeFrame(frameNo, pts3DIdx);
            
            //remove from list of keyframes
            _kFrames.erase(_kFrames.begin() + kfIdx);
            
            //remove from lookup vector
            _kFrameIdxToFrameNo.erase(kfIdx);
            _FrameNoTokFrameIdx.erase(frameNo);
            
            culledFrameNo++;
        } else {
            //advance counter
            kfIdx++;
        }
    }
    
    
    return culledFrameNo;
}

void CSfM::updateMapAndFrame(int frameNo, const vector<int> &pts3DIdx, const vector<int> &pts2DIdx) {

    int kIdx = _FrameNoTokFrameIdx[frameNo];
    Mat desc;
    _mapper.addPointMatches(pts3DIdx,pts2DIdx,frameNo);
    _kFrames[kIdx].updatePoints(pts2DIdx, pts3DIdx);
    _kFrames[kIdx].getDescriptorsAt(pts2DIdx, desc);
    _mapper.addDescriptors(pts3DIdx,desc);
}

void CSfM::addNewPointsMapAndFrame(int frameNo, const vector<Matx31d> &pts3D, const vector<int> &pts2DIdx, vector<int> &pts3DIdx) {
    
    int kIdx = _FrameNoTokFrameIdx[frameNo];
    Mat desc;
    _mapper.addNewPoints(pts3D, vector<vector<int>>{pts2DIdx}, vector<int>{frameNo}, pts3DIdx);
    _kFrames[kIdx].updatePoints(pts2DIdx, pts3DIdx);
    _kFrames[kIdx].getDescriptorsAt(pts2DIdx, desc);
    _mapper.addDescriptors(pts3DIdx, desc);
}

bool CSfM::recovery() {
    bool success = false;
    
    return success;
}

bool CSfM::startVideoOutput(string fileOut, int fourcc, Size imSize) {
    double _dScale = 0.5;
    _vOut = VideoWriter(fileOut, fourcc, 25, Size(imSize.width*_dScale
                                                         ,imSize.height*_dScale));
    
    if(!_vOut.isOpened()) {
        cerr << "Couldn't open " << fileOut << endl;
        return false;
    }

    return true;
}

void CSfM::stopVideoOutput() {
    
    _vOut.release();
}

void CSfM::getReconstruction(vector<Matx31d> &volume, vector<Vec3i> &colour) {
    _mapper.getPoints(volume);
}

bool CSfM::init() {
    
    //bool success = detectFeaturesOpticalFlow();
    bool success = _tracker.detectFeatures();
    
    //if features are detected
    if (success) {
        //if this is the first frame
        if (_kFrames.size() == 0) {
            CKeyFrame firstFrame(_tracker._currFrame);
            _kFrames.push_back(firstFrame);
            _kFrames[0].setPose();
            _kFrameIdxToFrameNo[0] = _tracker._currFrame.getFrameNo();
            _FrameNoTokFrameIdx[_tracker._currFrame.getFrameNo()] = 0;
            
            //swap buffers
            swap(_tracker._prevFrame, _tracker._currFrame);
            return true;
        }
        else {
            
            bool enoughFeaturesFound = _tracker.matchFeatures();
            //bool enoughFeaturesFound = computeOpticalFlow();
            
            if (!enoughFeaturesFound) {
                //reset frames
                //_kFrames.pop_back();
                //CKeyFrame firstFrame(_currFrame);
                //_kFrames.push_back(firstFrame);
#ifdef DEBUGINFO
                cout << "Not enough features found" << endl;
#endif
            }
            else {
                
                //find homography and fundamental matrix
                //DO NOT USE RANSAC for the following reasons:
                //1. matching correctness is ensured by optical flow (filtering is at the detector/matcher stage, not at the tracking stage)
                //2. unless the points are uniformely distributed it is possible to select a set that lies on the same plane, which yields an incorrect result
                vector<uchar> Hmask, Fmask;
                Matx33f H = findHomography( _tracker._prevMatch, _tracker._currMatch, 0, 5.99, Hmask);
                Matx33f F = findFundamentalMat(_tracker._prevMatch, _tracker._currMatch, CV_FM_RANSAC, 3.84,0.99, Fmask);
                
                if (!Mat(H).empty() || !Mat(F).empty()) {
                    Matx33d R_init;
                    Vec3d t_init;
                    
                    float s_h = calculateHomographyScore(_tracker._prevMatch, _tracker._currMatch, H, Hmask, 5.99, 5.99);
                    float s_f = calculateFundamentalScore(_tracker._prevMatch, _tracker._currMatch, F, Fmask, 3.84, 5.99);
                    
                    //get intrinsic matrices
                    Matx33f K0 = _kFrames[0].getIntrinsicUndistorted();
                    Matx33f K1 = _tracker._currFrame.getIntrinsicUndistorted();
                    
                    //choose best model
                    double r_h = s_h / (s_h + s_f);
                    bool canInit = false;
                    if (r_h > 0.45) {
                        //choose homography
#ifdef DEBUGINFO
                        
                        cout << "Homography initialization" << endl;
#endif
                        
                        //filter outliers
                        filterMatches(Hmask);
                        
                        
                        //calculate reprojection error
                        double errHomo = GeometryUtils::calculateHomographyAvgError(_tracker._prevMatch, _tracker._currMatch, H);
                        
                        //find decomposition
                        bool canDecompose = GeometryUtils::RtFromHomographyMatrix(H, K0, K1, _tracker._prevMatch, _tracker._currMatch, R_init, t_init);
#ifdef DEBUGINFO
                        if (errHomo >= _maxReprErr)
                            cout << "Homography transfer error too large" << endl;
#endif
                        
                        if (canDecompose && (errHomo < _maxReprErr)) {
                            canInit = true;
                        }
                        
                    }
                    else {
                        //choose fundamental matrix
#ifdef DEBUGINFO
                        cout << "Fundamental matrix initialization" << endl;
#endif
                        //remove outliers
                        filterMatches(Fmask);
                        
                        //find reprojection error on epilines
                        double errFund = GeometryUtils::calculateFundamentalAvgError(_tracker._prevMatch, _tracker._currMatch, F);
                        
                        //essential matrix
                        Matx33f E = K0.t()*F*K1;
                        //decompose essential
                        bool canDecompose = GeometryUtils::RtFromEssentialMatrix(E, K0, K1, _tracker._prevMatch, _tracker._currMatch, R_init, t_init);
#ifdef DEBUGINFO
                        if (errFund >= _maxReprErr)
                            cout << "Symmetric transfer error too large" << endl;
#endif
                        if (canDecompose && (errFund < _maxReprErr)) {
                            
                            canInit = true;
                        }
                    }
                    
                    
                    if (canInit) {
                        
                        //set pose information
                        _tracker._currFrame.setPose(R_init,t_init);
                        
                        //triangulate points
                        vector<Matx31d> pts3D;
                        GeometryUtils::triangulatePoints(_kFrames[0].getProjectionMatrix(), _tracker._currFrame.getProjectionMatrix(), K0, K1, _tracker._prevMatch, _tracker._currMatch, pts3D);
                        
                        //remove outliers
                        vector<uchar> status;
                        int nRemoved = GeometryUtils::filterMatches(F, _tracker._prevMatch, _tracker._currMatch, pts3D, status, _maxReprErr);
#ifdef DEBUGINFO
                        cout << "Removed " << nRemoved << " outliers" << endl;
#endif
                        vector<int> tPrevIdx, tCurrIdx;
                        vector<Point2f> tPrevMatch, tCurrMatch;
                        vector<Matx31d> filt3D;
                        tPrevIdx.reserve(_tracker._prevIdx.size() - nRemoved);
                        tCurrIdx.reserve(_tracker._currIdx.size() - nRemoved);
                        tPrevMatch.reserve(_tracker._prevMatch.size() - nRemoved);
                        tCurrMatch.reserve(_tracker._currMatch.size() - nRemoved);
                        filt3D.reserve(_tracker._currMatch.size() - nRemoved);
                        for (int i = 0; i < status.size(); i++) {
                            if (status[i] == 1) {
                                tPrevIdx.push_back(_tracker._prevIdx[i]);
                                tCurrIdx.push_back(_tracker._currIdx[i]);
                                tPrevMatch.push_back(_tracker._prevMatch[i]);
                                tCurrMatch.push_back(_tracker._currMatch[i]);
                                filt3D.push_back(pts3D[i]);
                            }
                        }
                        swap(_tracker._prevIdx,tPrevIdx);
                        swap(_tracker._currIdx, tCurrIdx);
                        swap(_tracker._prevMatch, tPrevMatch);
                        swap(_tracker._currMatch, tCurrMatch);
                        
                        //update map and frames
                        vector<int> pts3DIdx;
                        _mapper.addNewPoints(filt3D,vector<vector<int>>{_tracker._prevIdx,_tracker._currIdx}, vector<int>{_kFrames[0].getFrameNo(),_tracker._currFrame.getFrameNo()},pts3DIdx);
                        
                        _kFrames[0].updatePoints(_tracker._prevMatch,_tracker._prevIdx,pts3DIdx);
                        _tracker._currFrame.updatePoints(_tracker._currMatch,_tracker._currIdx,pts3DIdx);
                        

                        //add descriptors
                        Mat prevDescriptors, currDescriptors;
                        _kFrames[0].getDescriptorsAt(_tracker._prevIdx, prevDescriptors);
                        _tracker._currFrame.getDescriptorsAt(_tracker._currIdx, currDescriptors);
                        _mapper.addDescriptors(pts3DIdx, prevDescriptors);
                        _mapper.addDescriptors(pts3DIdx, currDescriptors);
                        _mapper.incrementMapAge(0,2);
                    
                        //push new keyframe
                        _kFrames.push_back(CKeyFrame(_tracker._currFrame));
                        _kFrameIdxToFrameNo[1] = _tracker._currFrame.getFrameNo();
                        _FrameNoTokFrameIdx[_tracker._currFrame.getFrameNo()] = 1;
                        
                        //bundle adjustment of both structure and pose
                        bundleAdjustment(vector<int>{_kFrames[0].getFrameNo(),_kFrames[1].getFrameNo()},CTracker::BA_TYPE::STRUCT_AND_POSE);

                        
#ifdef DEBUGINFO
                        cout << "# Features triangulated in frame 0: " << _kFrames[0].getNMatchedPoints() << endl;
                        cout << "# Features triangulated in frame 1: " << _kFrames[1].getNMatchedPoints() << endl;
#endif
                        
                        //change state
                        _state = RUNNING;
                        _tracker._currFrame.setPose(_kFrames[1].getRotation(), _kFrames[1].getTranslation());
                        swap(_tracker._prevFrame,_tracker._currFrame);
                        return true;
                    }
                    else {
                        //pop frame and start again
                        //_kFrames.pop_back();
                        //_currFrame = _prevFrame;
                    }
                } //computed valid fundamental and homography matrix
                
            }//match found
            
        } //second frame processed
        
    } //not enough features detected
    
    //if enough time has passed, swap buffers to avoid remaining stuck with the first frame
    if (_tracker._currFrame.getFrameNo() >= _tracker._prevFrame.getFrameNo() + _newKFrameTimeLag)
       swap(_tracker._prevFrame,_tracker._currFrame);
    return false;
}
