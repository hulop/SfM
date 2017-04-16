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
#include <algorithm>
#include "CKeyFrame.h"
#include "CTracker.h"
#include "CMap.h"
#include <opencv2/core/eigen.hpp>

using namespace cv;
using namespace std;

//#define DEBUGINFO 1
#define NOT_INITIALIZED 0
#define RUNNING 1
#define LOST 2

class CSfM {
public:
    CSfM(double fx, double fy, double s, double xc, double yc, Size imSize, vector<double> &d);
    ~CSfM();
    void addFrame(Mat frameIn);
    void getReconstruction(vector<Point3d> &volume, vector<Vec3i> &colour);
    
    bool startVideoOutput(string fileOut, int fourcc);
    void stopVideoOutput();
    
private:
    void setDistortionCoefficients(const vector<double> &d);
    
    bool init();
    bool computeOpticalFlow();
    bool detectFeatures();
    bool detectFeaturesOpticalFlow();
    
    void filterMatches(const vector<uchar> &status);
    bool matchFeatures();
    bool matchFeatures(CFrame &f0, CFrame &f1);
    bool tracking();
    bool tracking2();
    bool recovery();
    
    float calculateFundamentalScore(const vector<Point2f> &pts0, const vector<Point2f> &pts1, const Matx33f &F, const vector<uchar> &status, const float Th, const float Gamma);
    float calculateHomographyScore(const vector<Point2f> &pts0, const vector<Point2f> &pts1, const Matx33f &H, const vector<uchar> &status, const float Th, const float Gamma);
  
    //camera parameters
    Matx33d _K;
    Matx33d _Kopt;
    Matx33d _Kopt_i;
    Size _imSize;
    vector<double> _d;
    Mat _map1, _map2;
    int _state;
    
    vector<CKeyFrame> _kFrames;
    
    //parameters
    int _minFeatures;
    float _ratioTest;
    float _maxMatchDistance;
    float _maxReprErr;
    float _maxOrgFeatDist;
    int _lostCount;
    int _maxLost;
    int _frameCount;
    
    //tracker
    CTracker _tracker;
    
    //mapper
    CMap _mapper;
    
    //feature detection and matching
    Ptr<FeatureDetector> _detector;
    Ptr<DescriptorMatcher> _matcher;
    CFrame _currFrame;
    CFrame _prevFrame;
    vector<Point2f> _prevMatch;
    vector<Point2f> _currMatch;
    
    vector<int> _matchStatus;
    vector<float> _matchDistance;
    vector<int> _matchedIdx;
    
    vector<int> _prevIdx;
    vector<int> _currIdx;

    
    //debug parameters
    double _dScale;
    
    //output video
    VideoWriter _vOut;
};

#endif /* CSfM_h */
