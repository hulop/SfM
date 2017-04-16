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

CSfM::CSfM(double fx, double fy, double s, double xc, double yc, Size imSize, vector<double> &d) {
    _K << fx, s, xc, 0, fy, yc, 0, 0, 1;
    _state = NOT_INITIALIZED;
    _imSize = imSize;
    
    //set distortion and calculate mapping matrices
    setDistortionCoefficients(d);
    
    //create feature detector and set default parameters
    int nFeatures = 1000;
    double scaleFactor = 1.2;
    int nLevels = 8;
    int edgeThreshold = 31;
    int firstLevel = 0;
    int WTA_K = 4;
    int scoreType = ORB::HARRIS_SCORE;
    int patchSize = 31;
    int fastThreshold = 20;
    _maxMatchDistance = 40;
    
    _detector = ORB::create(nFeatures,scaleFactor,nLevels,edgeThreshold,firstLevel,WTA_K,scoreType,patchSize,fastThreshold);
    //_prevDesc = NULL;
    //_currDesc = NULL;
    
    //create feature matcher
    _matcher = DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE); //FLANN currently not working on iOS
    _ratioTest = 0.8;
    
    //other parameters
    _minFeatures = 5;
    _maxReprErr = 3;
    _maxOrgFeatDist = 1;
    _lostCount = 0;
    
    //debug info
#ifdef DEBUGINFO
    namedWindow("Debug");
    _dScale = 0.5;
#endif
    
}

CSfM::~CSfM() {
}

void CSfM::addFrame(cv::Mat frameIn) {
    cout << _mapper.getNPoints() << endl;
    //undistort incoming frame
    //TODO: to save ms, you can undistort it after greyscale conversion
    Mat frameU;
    remap(frameIn, frameU,_map1,_map2,INTER_NEAREST,BORDER_CONSTANT,0);
    
    //set frame
    _prevFrame = _currFrame;
    _currFrame.setFrame(frameU,_frameCount,_Kopt);

//debug info
#ifdef DEBUGINFO
    Mat dShow;
    resize(frameU, dShow, Size(_dScale*_imSize.width,_dScale*_imSize.height), 0,0, INTER_NEAREST);
 //   imshow("Debug",dShow);
#endif
    
    bool success;
    switch (_state) {
        case NOT_INITIALIZED:
            success = init();
            break;
        case RUNNING:
            success = tracking();
            break;
        case LOST:
            success = recovery();
            break;
        default:
            break;
    }
    
    _frameCount++;
}

bool CSfM::init() {
    
    bool success = detectFeaturesOpticalFlow();
    //bool success = detectFeatures();
    
    //if features are detected
    if (success) {
        //if this is the first frame
        if (_kFrames.size() == 0) {
            CKeyFrame firstFrame(_currFrame);
            _kFrames.push_back(firstFrame);
            return true;
        }
        else {
            //bool enoughFeaturesFound = matchFeatures();
            bool enoughFeaturesFound = computeOpticalFlow();
            
            if (!enoughFeaturesFound) {
                //reset frames
                _kFrames.pop_back();
                //CKeyFrame firstFrame(_currFrame);
                //_kFrames.push_back(firstFrame);
            }
            else {
                
                //find homography and fundamental matrix
                //DO NOT USE RANSAC for the following reasons:
                //1. matching correctness is ensured by optical flow (filtering is at the detector/matcher stage, not at the tracking stage)
                //2. unless the points are uniformely distributed it is possible to select a set that lies on the same plane, which yields an incorrect result
                vector<uchar> Hmask, Fmask;
                Matx33f H = findHomography( _prevMatch, _currMatch, 0, 5.99, Hmask);
                Matx33f F = findFundamentalMat(_prevMatch, _currMatch, CV_FM_8POINT, 3.84,0.99, Fmask);
                
                if (!Mat(H).empty() || !Mat(F).empty()) {
                    Matx33d R_init;
                    Vec3d t_init;
                    
                    float s_h = calculateHomographyScore(_prevMatch, _currMatch, H, Hmask, 5.99, 5.99);
                    float s_f = calculateFundamentalScore(_prevMatch, _currMatch, F, Fmask, 3.84, 5.99);
                    
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
                        float errHomo = CTracker::calculateHomographyAvgError(_prevMatch, _currMatch, H);
                        
                        //find decomposition
                        bool canDecompose = CTracker::RtFromHomographyMatrix(H, _Kopt, _prevMatch, _currMatch, R_init, t_init);
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
                        double errFund = CTracker::calculateFundamentalAvgError(_prevMatch, _currMatch, F);
                
                        //essential matrix
                        Matx33f E = Matx33f(_Kopt).t()*F*Matx33f(_Kopt);
                        //decompose essential
                        bool canDecompose = CTracker::RtFromEssentialMatrix(E, _Kopt, _prevMatch, _currMatch, R_init, t_init);
                        
                        if (canDecompose && (errFund < 5)) {
                            canInit = true;
                        }
                    }
                    
                    
                    if (canInit) {
                        
                        //set pose information
                        _kFrames[0].setPose();
                        _currFrame.setPose(R_init,t_init);
                        //add second frame to keyframe list
                        _kFrames.push_back(CKeyFrame(_currFrame));
                        
                        //triangulate points
                        vector<Point3d> pts3D;
                        CTracker::triangulatePoints(_kFrames[0].getProjectionMatrix(), _kFrames[1].getProjectionMatrix(), _Kopt, _Kopt, _prevMatch, _currMatch, pts3D);

                        for (int d = 0; d < pts3D.size(); d++) {
                            cout << pts3D[d].x << " " << pts3D[d].y << " " << pts3D[d].z << ";" << endl;
                        }
//                        vector<Point2f> reproj;
//                        Vec3f tvec = _kFrames[1].getTranslation();
//                        Mat rvec = _kFrames[1].getRotationRodrigues();
//                        rvec.convertTo(rvec, CV_32F);
//                        vector<Point3f> pts3Df;
//                        for (int j = 0; j < pts3D.size(); j++) {
//                            pts3Df.push_back(Point3f(pts3D[j]));
//                        }
//                        projectPoints(pts3Df, rvec, tvec, Matx33f(_Kopt), vector<float>(), reproj);
//                        
//                        //reproject manually
//                        Matx33f RR = _kFrames[1].getRotation();
//                        Mat dShow = (_kFrames[1].getFrame()).clone() ;
//                        for (int d = 0; d < _currMatch.size(); d++) {
//                                circle(dShow, Point2f(reproj[d]), 3, Scalar(0,255,0),-1,CV_AA);
//                        }
//                        Mat dShow_small;
//                        resize(dShow, dShow_small, Size(0.5*dShow.cols,0.5*dShow.rows));
//                                                imshow("Debug",dShow_small);
//                                                waitKey();
                        
                        //bundle adjustment
                        
            
                        //update map and frames
                        vector<int> pts3DIdx;
                        _mapper.addNewPoints(pts3D,vector<vector<int>>{_prevIdx,_currIdx}, vector<int>{0,1},pts3DIdx);
                        
                        _kFrames[0].updatePoints(_prevMatch,_prevIdx,pts3DIdx);
                        _kFrames[1].updatePoints(_currMatch,_currIdx,pts3DIdx);
                        
#ifdef DEBUGINFO
                        cout << "# Features triangulated in frame 0: " << _kFrames[0].getNMatchedPoints() << endl;
                        cout << "# Features triangulated in frame 1: " << _kFrames[1].getNMatchedPoints() << endl;
#endif
                        //change state
                        _currFrame = _kFrames[1];
                        _state = RUNNING;
                        return true;
                    }
                    else {
                        //pop frame and start again
                        _kFrames.pop_back();
                    }
                } //computed valid fundamental and homography matrix
            
            }//match found
        
        } //second frame processed
    
    } //not enough features detected
    return false;
}

void CSfM::filterMatches(const vector<uchar> &status) {
    //filter matches according to the status mask from findhomography or findfundamental
    vector<Point2f> tempPrevMatch, tempCurrMatch;
    vector<int> tempPrevIdx, tempCurrIdx;
    for (int i = 0; i < status.size(); i++) {
        if (status[i] == 1) {
            tempPrevMatch.push_back(_prevMatch[i]);
            tempCurrMatch.push_back(_currMatch[i]);
            tempPrevIdx.push_back(_prevIdx[i]);
            tempCurrIdx.push_back(_currIdx[i]);
        }
    }
    //swap buffers
    _prevMatch = tempPrevMatch;
    _currMatch = tempCurrMatch;
    _prevIdx = tempPrevIdx;
    _currIdx = tempCurrIdx;
}

bool CSfM::computeOpticalFlow() {
    
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
    Mat(_prevFrame.getPoints()).copyTo(prevPts);
    vector<Point2f> currMatch, currDetectedPoints;
    Mat(_currFrame.getPoints()).copyTo(currDetectedPoints);
    _matchDistance.assign(currDetectedPoints.size(), -1);
    _matchStatus.assign(currDetectedPoints.size(), 0);
    _matchedIdx.assign(currDetectedPoints.size(), -1);
    
    //compute LK optical flow
    int matchCount = 0;
    if (_prevFrame.getNPoints() > 0) {
        calcOpticalFlowPyrLK(_prevFrame.getFrameGrey(), _currFrame.getFrameGrey(), prevPts, currPts, status, err, winSize, maxLevel, termcrit, 0, 0.001 );
        //check that PyrLK does not play tricks on us
        assert(prevPts.size() == currPts.size());
        
        //check matches
        for (int i = 0; i < status.size(); i++) {
            if (status[i]) {
                //look for actual detected point in other image
                int idx = _currFrame.findClosestPointIndex(currPts[i]);
                float e = norm(currPts[i] - currDetectedPoints[idx]);
                float d = norm(prevPts[i] - currPts[i]);
                
                if ((d < _maxMatchDistance) && (e < _maxOrgFeatDist) && ((_matchDistance[idx] >= e) || (_matchDistance[idx] == -1)) ) {

                    //check if point was already matched
                    if ((_matchStatus[idx] == 1)) {
                        //overwrite previous match since the one found is better
                        const int oldIdx = _matchedIdx[idx];
                        _prevMatch[oldIdx] = prevPts[i];
                        _prevIdx[oldIdx] = i;
                    } else {
                        //save matches
                        _prevMatch.push_back(prevPts[i]);
                        _currMatch.push_back(currDetectedPoints[idx]);
                        
                        //save indices in CFrame arrays
                        _prevIdx.push_back(i);
                        _currIdx.push_back(idx);

                        
                        _matchStatus[idx] = 1;
                        _matchedIdx[idx] = matchCount;
                        matchCount++;
                    }
                    _matchDistance[idx] = e;
                }
            }
        }

#ifdef DEBUGINFO
        cout << "# Matches: " << sum(_matchStatus)[0] << endl;
#endif
    }
    
    
//    vector<DMatch> Dmatches;
//    for (int i = 0; i < matchCount; i++) {
//        Dmatches.push_back(DMatch(i, i, 0));
//    }
//
//    Mat matchImg, dShow;
//    vector<KeyPoint> match0, match1;
//    KeyPoint::convert(_prevMatch, match0);
//    KeyPoint::convert(_currMatch, match1);
//    drawMatches(_prevFrame.getFrameGrey() , match0, _currFrame.getFrameGrey(), match1, Dmatches, matchImg);
//    float _dScale = 0.5;
//    resize(matchImg, dShow, Size(2*_dScale*_imSize.width,_dScale*_imSize.height), 0,0,INTER_NEAREST);
//    imshow("Debug",dShow);
//    waitKey();
//  //  _vOut << dShow;

    
    if (matchCount >= _minFeatures)
        return true;
    
    return false;
}


bool CSfM::detectFeaturesOpticalFlow() {
    bool success = false;

    int maxFeats = 500;
    double qualityLvl = 0.05;
    double minDistance = 8;
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

bool CSfM::detectFeatures() {
    vector<KeyPoint> kp;
    Mat desc;
    _detector->detectAndCompute(_currFrame.getFrameGrey(), noArray(), kp, desc);
    
    //show detected features
#ifdef DEBUGINFO
    Mat detectedFeats, dShow;
    drawKeypoints(_currFrame.getFrameGrey(), kp, detectedFeats);
    resize(detectedFeats, dShow, Size(_dScale*_imSize.width,_dScale*_imSize.height), 0,0, INTER_NEAREST);
    imshow("Debug",dShow);
   // waitKey();
#endif
    
    if (kp.size() < _minFeatures)
        return false;
    
    _currFrame.setKeyPoints(kp,desc);
    return true;
}

bool CSfM::matchFeatures() {
    //default method matches between current and previous frame
    
    //clear previous matches
    _prevMatch.clear();
    _currMatch.clear();
    
    //declare structures
    vector<vector<DMatch>> matches01, matches10;
    Mat prevDesc = _prevFrame.getDescriptors();
    Mat currDesc = _currFrame.getDescriptors();
    vector<KeyPoint> prevKeypts = _prevFrame.getKeyPoints();
    vector<KeyPoint> currKeypts = _currFrame.getKeyPoints();
    
    //match in both directions
    _matcher->knnMatch(prevDesc, currDesc, matches01, 2);
    _matcher->knnMatch(currDesc, prevDesc, matches10, 2);
    
    //check matches
    int matchCount = 0;
    for (int i = 0; i < matches01.size(); i++) {
        //forward-backward test, ratio test and distance test
        DMatch fwd = matches01[i][0];
        DMatch bwd = matches10[fwd.trainIdx][0];
        bool fbTest = (bwd.trainIdx == fwd.queryIdx);
        bool rTest = (matches01[i][0].distance < _ratioTest*matches01[i][1].distance);
        bool dTest = (norm(prevKeypts[fwd.queryIdx].pt - currKeypts[fwd.trainIdx].pt) < _maxMatchDistance);
        
        if (rTest && fbTest && dTest)  {
                _prevMatch.push_back(prevKeypts[fwd.queryIdx].pt);
                _currMatch.push_back(currKeypts[fwd.trainIdx].pt);
             //   _matches.push_back(DMatch(matchCount, matchCount, fwd.distance));
                matchCount++;
        }
    }
    
#ifdef DEBUGINFO
    Mat matchImg, dShow;
    vector<KeyPoint> match0, match1;
    KeyPoint::convert(_prevMatch, match0);
    KeyPoint::convert(_currMatch, match1);
  //  drawMatches(_prevFrame.getFrameGrey() , match0, _currFrame.getFrameGrey(), match1, _matches, matchImg);
    resize(matchImg, dShow, Size(2*_dScale*_imSize.width,_dScale*_imSize.height), 0,0,INTER_NEAREST);
    imshow("Debug",dShow);
    _vOut << dShow;
    //waitKey();
#endif
    
    if (matchCount >= _minFeatures)
        return true;
    
    return false;
}

bool CSfM::matchFeatures(CFrame &f0, CFrame &f1) {
    

    return true;
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
    
#ifdef DEBUGINFO
    //draw features and reprojections
    Mat dShow;
    vector<KeyPoint> dCurrKpts, dPrevKpts;
    KeyPoint::convert(bwd,dCurrKpts);
    KeyPoint::convert(pts0,dPrevKpts);
    drawKeypoints(_prevFrame.getFrameGrey(),dPrevKpts,dShow,Scalar(255,0,0));
    drawKeypoints(_prevFrame.getFrameGrey(),dCurrKpts,dShow,Scalar(0,0,255),DrawMatchesFlags::DRAW_OVER_OUTIMG);
    imshow("Debug",dShow);
    //waitKey();
#endif
    
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
            e01 = CTracker::distancePointLine2D(pts0[i], epiLines0[i]);
            e10 = CTracker::distancePointLine2D(pts1[i], epiLines1[i]);
        
            //compute gamma
            gamma0 = (e01 < Tf) ? Gamma - e01 : 0;
            gamma1 = (e10 < Tf) ? Gamma - e10 : 0;
        
            s += gamma0 + gamma1;
            count++;
        }
    }

    
//    //draw epilines
//    int nFeats = 10;
//    Mat dShow0 = (_prevFrame.getFrameGrey()).clone();
//    Mat dShow1 = (_currFrame.getFrameGrey()).clone();
//    cvtColor(dShow0, dShow0, CV_GRAY2RGB);
//    cvtColor(dShow1, dShow1, CV_GRAY2RGB);
//    for (int i = 0; i < nFeats; i++) {
//        //draw feature
//        circle(dShow0, pts0[i], 3, Scalar(0,255,0), -1, CV_AA);
//        
//        //draw corresponding epiline
//        line(dShow1, Point2f(0,-epiLines1[i][2]/epiLines1[i][1]), Point2f(_imSize.width,-(epiLines1[i][0]*_imSize.width + epiLines1[i][2])/epiLines1[i][1]), Scalar(0,255,0),1,CV_AA);
//    }
//    Mat dShow;
//    hconcat(dShow0, dShow1, dShow);
//    float _dScale = 0.5;
//    resize(dShow, dShow, Size(_dScale*2*_imSize.width,_dScale*_imSize.height), 0,0, INTER_NEAREST);
//    imshow("Debug",dShow);
//    waitKey();
    
    return s/count;
}

bool CSfM::tracking() {
    bool success = false;
    
    //detect features
    detectFeaturesOpticalFlow();
    
    //match (only stores matches in local structures)
    computeOpticalFlow();
    
    //get 3D points visible in the previous frame
    vector<Point3d> prevMatched3D;
    vector<int> prevMatched3DIdx;
    vector<int> prevMatched2DIdx;
    _mapper.getPointsInFrame(prevMatched3D, prevMatched3DIdx, prevMatched2DIdx, _prevFrame.getFrameNo());
    
    vector<Point2d> currMatched2D, currUnmatched2D, prevUnmatched2D, prevMatched2D;
    vector<Point3d> currMatched3D;
    vector<int> currMatched3DIdx, currMatched2DIdx, currUnmatched2DIdx, prevUnmatched2DIdx;
    vector<int>::iterator idxIter;
    for (int i = 0; i < _prevIdx.size(); i++) {
        idxIter = find(prevMatched2DIdx.begin(),prevMatched2DIdx.end(), _prevIdx[i]);
        auto pos = idxIter - prevMatched2DIdx.begin();
        if (idxIter != prevMatched2DIdx.end()) {
            currMatched2D.push_back(_currMatch[i]);
            prevMatched2D.push_back(_prevMatch[i]);
            currMatched2DIdx.push_back(_currIdx[i]);
            currMatched3D.push_back(prevMatched3D[pos]);
            currMatched3DIdx.push_back(prevMatched3DIdx[pos]);
        } else {
            currUnmatched2D.push_back(_currMatch[i]);
            prevUnmatched2D.push_back(_prevMatch[i]);
            currUnmatched2DIdx.push_back(_currIdx[i]);
            prevUnmatched2DIdx.push_back(_prevIdx[i]);
        }
    }
    
    //if no points have been matched in existing frame, the frame is lost (or try with the next one within a fixed time window)
    if (currMatched2D.size() < _minFeatures) {
        cout << "Lost track!" << endl;
        _lostCount++;
        //swap buffers in order to ignore the current frame, next frame will be matched against last valid frame
        _currFrame = _prevFrame;
        if (_lostCount > _maxLost) {
            _state = LOST;
        }
    } else {
        //reset lostCount
        _lostCount = 0;
        
        //solve PnP
        int iter = 100;
        double confidence = 0.99;
        double reprErr = 8.0;
        Mat rvec = Mat::zeros(3,1,CV_64FC1);
        Mat tvec = Mat::zeros(3,1,CV_64FC1);
        //TODO: can provide initial guess of r and t based on motion model
        
        vector<int> inlierIdx;
        // solvePnP(prevMatched3DSubset, currMatched2D, _Kopt, Mat::zeros(4,1,CV_32FC1), rvec, tvec,false, SOLVEPNP_ITERATIVE);
        solvePnPRansac(currMatched3D, currMatched2D, _Kopt, Mat::zeros(4,1,CV_64FC1), rvec, tvec, false, iter, reprErr, confidence, inlierIdx, SOLVEPNP_EPNP );
        
        //update pose of current frame
        Mat R;
        Rodrigues(rvec,R);
        _currFrame.setPose(R,tvec);
       
        //triangulate remaining points
        vector<Point3d> new3DPts;
        CTracker::triangulatePoints(_prevFrame.getProjectionMatrix(), _currFrame.getProjectionMatrix(), _Kopt, _Kopt, prevUnmatched2D, currUnmatched2D, new3DPts);
        
        //update 3d map with points already matched
        _mapper.addPointMatches(currMatched3DIdx,currMatched2DIdx,_currFrame.getFrameNo());
        
        //update 3d map with new points
        vector<int> new3DPtsIdx;
          _mapper.addNewPoints(new3DPts, vector<vector<int>>{prevUnmatched2DIdx,currUnmatched2DIdx}, vector<int>{_prevFrame.getFrameNo(),_currFrame.getFrameNo()}, new3DPtsIdx);
        
        //update frames with indices of new triangulated points
        _kFrames[_kFrames.size() - 1].updatePoints(prevUnmatched2DIdx, new3DPtsIdx);
        _currFrame.updatePoints(currMatched2DIdx, currMatched3DIdx);
        _currFrame.updatePoints(currUnmatched2DIdx, new3DPtsIdx);
        
        //bundle adjustment

        //add to keyframes list
        _kFrames.push_back(CKeyFrame(_currFrame));
        
        //just for debug
//        Mat dShow, outdShow;
//        vector<KeyPoint> detectedKP;
//        vector<Point2f> dPoints;
//        vector<int> dIdx;
//        _currFrame.getMatchedPoints(dPoints, dIdx);
//        KeyPoint::convert(dPoints, detectedKP);
//        float _dScale = 0.5;
//        drawKeypoints(_currFrame.getFrame(), detectedKP, dShow,Scalar(0,0,255));
//        resize(dShow, outdShow, Size(_dScale*_imSize.width,_dScale*_imSize.height), 0,0,INTER_NEAREST);
        
        Point3f c = _mapper.getCentroid();
        cout << tvec << endl;
        Mat dShow = _currFrame.getFrame();
        Matx31d u = _Kopt*_currFrame.getProjectionMatrix()*Matx41d(c.x,c.y,c.z,1);
        u *= 1.0/u(2);
        circle(dShow, Point(u(0),u(1)), 10, Scalar(0,0,255), -1, CV_AA);
        Mat outdShow;
        float _dScale = 0.5;
        resize(dShow, outdShow, Size(_dScale*_imSize.width,_dScale*_imSize.height), 0,0,INTER_NEAREST);
//
//        imshow("Debug",outdShow);
        _vOut << outdShow;
       // waitKey();
        
    }
    

    
    return success;
}

bool CSfM::recovery() {
    bool success = false;
    
    return success;
}

void CSfM::setDistortionCoefficients(const vector<double> &d) {
    //deep copy
    _d.clear();
    _d.insert(std::begin(_d),std::begin(d),std::end(d));
    
    //calculate optimal new intrinsic matrix
    _Kopt = getOptimalNewCameraMatrix(_K, _d, _imSize, 0);
    _Kopt_i = _Kopt.inv();
    
    //calculate lens distortion maps
    initUndistortRectifyMap(_K, _d, noArray(), _Kopt, _imSize, CV_32FC1, _map1, _map2);
}

bool CSfM::startVideoOutput(string fileOut, int fourcc) {
    _dScale = 0.5;
    _vOut = VideoWriter(fileOut, fourcc, 25, Size(_imSize.width*_dScale
                                                         ,_imSize.height*_dScale));
    
    if(!_vOut.isOpened()) {
        cerr << "Couldn't open " << fileOut << endl;
        return false;
    }

    return true;
}

void CSfM::stopVideoOutput() {
    
    _vOut.release();
}

void CSfM::getReconstruction(vector<Point3d> &volume, vector<Vec3i> &colour) {
    _mapper.getPoints(volume);
}
