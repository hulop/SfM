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
#include "CVUtils/Display2D.hpp"
#include "CVUtils/VectorUtils.hpp"
#include "CVUtils/GeometryUtils.hpp"
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

    //SLAM threads
    bool init();
    bool tracking();
    bool recovery();
    bool mapping();
    
    void filterMatches(const vector<uchar> &status);
    
    bool addKeyFrame();
    int cullMapPoints();
    int cullKeyFrames(double thresholdRate = 0.9);
    
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
    int _lostCount;
    int _maxLost;
    int _frameCount;
    int _covisibilityThreshold;
    int _newKFrameTimeLag;
    int _minVisibilityFrameNo;
    double _maxReprErr;
    
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
    
    //output video
    VideoWriter _vOut;
    
    //lookup table for keyframe index to frame number
    unordered_map<int,int> _kFrameIdxToFrameNo;
    unordered_map<int,int> _FrameNoTokFrameIdx;

    //signals
    bool _keyFrameAdded;
};

#endif /* CSfM_h */
