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
#ifndef CTracker_hpp
#define CTracker_hpp

#include <stdio.h>
#include "CKeyFrame.h"
#include "opencv2/opencv.hpp"
#include "Eigen/Dense"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "../../cvUtils/VectorUtils.hpp"
#include "../../cvUtils/GeometryUtils.hpp"
#include "brisk/brisk.h"

#endif /* CTracker_hpp */

using namespace std;
using namespace cv;

class CTracker {
    friend class CSfM;
public:
    
    CTracker(const Matx33d &K, const vector<double> &d, const Size &imSize);
    ~CTracker();

    //feature detection
    bool detectFeaturesOpticalFlow();
    bool detectFeatures();
    
    //tracking and matching
    void matchFeaturesRadius(const vector<Point2d> &pts0, const Mat &desc0, const vector<Point2d> &pts1, const Mat &desc1, vector<int> &matchIdx0, vector<int> &matchIdx1);
    void matchFeatures(const vector<Point2d> &pts0, const Mat &desc0, const vector<Point2d> &pts1, const Mat &desc1, vector<int> &matchIdx0, vector<int> &matchIdx1);
    void matchFeaturesRadius(const vector<Point2d> &pts0, const Mat &desc0, const vector<Point2d> &pts1, const Mat &desc1, vector<int> &matchIdx0, vector<int> &matchIdx1, const double minDistance, const double maxDistance);
    void matchFeatures(const vector<Point2d> &pts0, const Mat &desc0, const vector<Point2d> &pts1, const Mat &desc1, vector<int> &matchIdx0, vector<int> &matchIdx1, const double minDistance, const double maxDistance);
    void matchFeatures(const vector<int> &prevIdx, const vector<int> &currIdx, vector<int> &prevMatchIdx, vector<int> &currMatchIdx);
    bool matchFeaturesRadius();
    bool matchFeatures();
    bool computeOpticalFlow();
    
    void setCurrentFrame(const Mat &frameIn, const int frameNo) {_currFrame.setFrame(frameIn,frameNo);}
    void setCurrentFrame(const Mat &frameIn, const int frameNo, const Mat &K);
    
    //bundle adjustment
    void bundleAdjustmentStructAndPose(const vector<Point2d> &observations, const vector<int> &camIdx, const vector<Matx33d> &K, vector<double*> &R, vector<double*> &t, vector<double*> &pts3D, int isStructOrPose);
    
    enum BA_TYPE { STRUCT_ONLY = 0, POSE_ONLY = 1, STRUCT_AND_POSE = 2};
    
private:
    void initialiseBAOptions();
    
    //feature matching and description
    brisk::BriskFeatureDetector *_detector;
    brisk::BriskDescriptorExtractor *_descriptor;
    brisk::BruteForceMatcher _matcher;
    
    //thresholds
    int _minFeatures;
    double _ratioTest;
    double _maxMatchDistance;
    double _minMatchDistance;
    double _maxMatchDistanceSq;
    double _minMatchDistanceSq;
    double _maxHammingDistance;
    double _maxOrgFeatDist;
    
    //matching structures
    vector<Point2f> _prevMatch;
    vector<Point2f> _currMatch;
    
    vector<int> _prevIdx;
    vector<int> _currIdx;
    
    //frames
    CFrame _prevFrame;
    CFrame _currFrame;
    
    //bundle adjustment structures
    ceres::Solver::Options _opts;
    
    struct BAStructAndPoseFunctor {
        
        BAStructAndPoseFunctor(double pt2d_x, double pt2d_y, const double *k);
        
        template <typename T>
        bool operator()(const T *const R, const T *const t, const T* const point, T* residuals) const;
        
        static ceres::CostFunction* Create(const double pt2d_x,const double pt2d_y, const double *k) {
            return (new ceres::AutoDiffCostFunction<CTracker::BAStructAndPoseFunctor, 2, 3, 3, 3>(new CTracker::BAStructAndPoseFunctor(pt2d_x, pt2d_y, k)));
        }
    
        double _pt2d_x;
        double _pt2d_y;
        //intrisic parameters
        const double *_k;
    };
    
    struct BAPoseFunctor{
        BAPoseFunctor(double pt2d_x, double pt2d_y, const double *pt3d, const double *k);
        
        template <typename T>
        bool operator()(const T *const R, const T *const t, T* residuals) const;
        
        static ceres::CostFunction* Create(const double pt2d_x,const double pt2d_y, const double *pt3d, const double *k) {
            return (new ceres::AutoDiffCostFunction<CTracker::BAPoseFunctor, 2, 3, 3>(new CTracker::BAPoseFunctor(pt2d_x, pt2d_y, pt3d, k)));
        }
        
        double _pt2d_x;
        double _pt2d_y;
        const double *_pt3d;
        //intrisic parameters
        const double *_k;
    };
    
        struct BAStructFunctor{
            BAStructFunctor(double pt2d_x, double pt2d_y, const double *R, const double *t, const double *k);
            
            template <typename T>
            bool operator()(const T *const pt3d, T* residuals) const;
            
            static ceres::CostFunction* Create(const double pt2d_x,const double pt2d_y, const double *R, const double *t, const double *k) {
                return (new ceres::AutoDiffCostFunction<CTracker::BAStructFunctor, 2, 3>(new CTracker::BAStructFunctor(pt2d_x, pt2d_y, R, t, k)));
            }
            
            double _pt2d_x;
            double _pt2d_y;
            const double *_R;
            const double *_t;
            //intrisic parameters
            const double *_k;
        };
};
