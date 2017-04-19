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

#ifndef CSfM_h
#define CSfM_h

#include <cmath>
#include <thread>
#include "CKeyFrame.h"
#include "CTracker.h"
#include "CMap.h"
#include <opencv2/core/eigen.hpp>
#include "../../cvUtils/Display2D.hpp"
#include "../../cvUtils/VectorUtils.hpp"
#include "../../cvUtils/GeometryUtils.hpp"
#include "brisk/brisk.h"


using namespace cv;
using namespace std;

#define DEBUGINFO 1
#define NOT_INITIALIZED 0
#define RUNNING 1
#define LOST 2

class CSfM {
public:
    CSfM(const Matx33d &K, const Size &imSize, const vector<double> &d);
    ~CSfM();
    void addFrame(Mat frameIn);
    void getReconstruction(vector<Matx31d> &volume, vector<Vec3i> &colour);
    
    bool startVideoOutput(string fileOut, int fourcc, Size imSize);
    void stopVideoOutput();
    
private:
    
    bool init();
    bool computeOpticalFlow();
    bool detectFeatures();
    bool detectFeaturesOpticalFlow();
    
    void filterMatches(const vector<uchar> &status);
    void filterMatches(const vector<int> &outIdx);
    template<typename T>
    void filterArray(const vector<int> &outIdx, vector<T> &v);
    bool matchFeatures();
    void matchFeatures(const vector<Point2d> &pts0, const Mat &desc0, const vector<Point2d> &pts1, const Mat &desc1, vector<int> &matchIdx0, vector<int> &matchIdx1, double minDistance, double maxDistance);
    bool matchFeaturesRadius();
    void matchFeaturesRadius(const vector<Point2d> &pts0, const Mat &desc0, const vector<Point2d> &pts1, const Mat &desc1, vector<int> &matchIdx0, vector<int> &matchIdx1, double minDistance, double maxDistance);
    
    bool tracking();
    bool recovery();
    bool mapping();
    bool addKeyFrame();
    int cullMapPoints();
    int cullKeyFrames(double thresholdRate = 0.9);
    
    void findMapPointsInFrame(int frameNo);
    void findMapPointsInCurrentFrame();
    void updateMotionHistory(const Matx33d &R, const Matx31d &t);
    void updateMapAndFrame(int frameNo, const vector<int> &pts3DIdx, const vector<int> &pts2DIdx);
    void addNewPointsMapAndFrame(int frameNo, const vector<Matx31d> &pts3D, const vector<int> &pts2DIdx, vector<int> &pts3DIdx);
    void predictFlow(vector<Point2f> &predPts);
    
    void bundleAdjustment(const vector<int> &frameIdx, int isStructAndPose);
    
    float calculateFundamentalScore(const vector<Point2f> &pts0, const vector<Point2f> &pts1, const Matx33f &F, const vector<uchar> &status, const float Th, const float Gamma);
    float calculateHomographyScore(const vector<Point2f> &pts0, const vector<Point2f> &pts1, const Matx33f &H, const vector<uchar> &status, const float Th, const float Gamma);
  
    //camera parameters
    int _state;
    
    vector<CKeyFrame> _kFrames;
    
    //parameters
    int _minFeatures;
    double _ratioTest;
    double _maxMatchDistance;
    double _minMatchDistance;
    double _maxMatchDistanceSq;
    double _minMatchDistanceSq;
    double _maxReprErr;
    double _maxOrgFeatDist;
    double _maxHammingDistance;
    int _lostCount;
    int _maxLost;
    int _frameCount;
    int _minCovisibilityStrength;
    int _covisibilityThreshold;
    int _newKFrameTimeLag;
    int _minVisibilityFrameNo;
    
    //tracker
    CTracker _tracker;
    
    //mapper
    CMap _mapper;
    
    //motion history for predictive motion model
    vector<Matx33d> _R;
    vector<Matx31d> _t;
    Matx33d _Rpred;
    Matx31d _tpred;
    int _motionHistoryLength;
    
    //feature detection and matching
    Ptr<FeatureDetector> _detector;
    Ptr<DescriptorExtractor> _descriptor;
    brisk::BruteForceMatcher _matcher;
    
    CFrame _currFrame;
    CFrame _prevFrame;
    vector<Point2f> _prevMatch;
    vector<Point2f> _currMatch;
    vector<double> _parallax;
    
    vector<int> _matchStatus;
    vector<float> _matchDistance;
    vector<int> _matchedIdx;
    
    vector<int> _prevIdx;
    vector<int> _currIdx;
    
    //output video
    VideoWriter _vOut;
    
    //lookup table for keyframe index to frame number
    unordered_map<int,int> _kFrameIdxToFrameNo;
    unordered_map<int,int> _FrameNoTokFrameIdx;
    
    //signals
    bool _keyFrameAdded;
};

#endif /* CSfM_h */
