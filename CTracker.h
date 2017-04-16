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
#include "opencv2/opencv.hpp"
#include "Eigen/Dense"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

#endif /* CTracker_hpp */

using namespace std;
using namespace cv;

class CTracker {
public:
    
    CTracker();
    ~CTracker();
    
    //triangulation
    static void triangulatePoints(const Matx34d &P0, const Matx34d &P1, const Matx33d &K0, const Matx33d &K1, const vector<Point2d> &f0, const vector<Point2d> &f1, vector<Point3d> &outPts);
    static void triangulatePoints(const Matx34d &P0, const Matx34d &P1, const Matx33d &K0, const Matx33d &K1, const vector<Point2f> &f0, const vector<Point2f> &f1, vector<Point3d> &outPts);
    
    //matrix decomposition
    static bool RtFromEssentialMatrix(const Matx33d &E, const Matx33d &K, const vector<Point2d> &pts0, const vector<Point2d> &pts1, Matx33d &R, Vec3d &t);
    static bool RtFromEssentialMatrix(const Matx33f &E, const Matx33f &K, const vector<Point2f> &pts0, const vector<Point2f> &pts1, Matx33d &R, Vec3d &t);
    static bool RtFromHomographyMatrix(const Matx33f &H, const Matx33f &K, const vector<Point2f> &pts0, const vector<Point2f> &pts1, Matx33d &R, Vec3d &t);
    
    //projection errors
    static double calculateFundamentalAvgError(const vector<Point2d> &pts0, const vector<Point2d> &pts1, const Matx33d &F);
    static double calculateHomographyAvgError(const vector<Point2d> &pts0, const vector<Point2d> &pts1, const Matx33d &H);
    static float calculateFundamentalAvgError(const vector<Point2f> &pts0, const vector<Point2f> &pts1, const Matx33f &F);
    static float calculateHomographyAvgError(const vector<Point2f> &pts0, const vector<Point2f> &pts1, const Matx33f &H);
    
    //geometry operations
    static double distancePointLine2D(const Point2d &pt, const Vec3d &l);
    
    //bundle adjustment
    void bundleAdjustmentStructAndPose(const vector<Point2d> &observations, const vector<int> &camIdx, const vector<int> &pt3DIdx, const vector<Matx33d> &K);
    void bundleAdjustmentPose();
    void bundleAdjustmentStruct();
    
private:
    
    static Point3d linearTriangulation(const Matx34d &P0, const Matx34d &P1, const Point3d pt0, const Point3d pt1, int iter = 10);
    
    
    
    //bundle adjustment structures
    struct BAStructAndPoseFunctor {
        
        BAStructAndPoseFunctor(double observed_x, double observed_y, double fx, double fy, double s, double xc, double yc);
        
        template <typename T>
        bool operator()(const T* const camera, const T* const point, T* residuals) const;
        
        static ceres::CostFunction* Create(const double observed_x,const double observed_y, const double fx, const double fy, const double s, const double xc, const double yc) {
            return (new ceres::AutoDiffCostFunction<CTracker::BAStructAndPoseFunctor, 2, 6, 3>(new CTracker::BAStructAndPoseFunctor(observed_x, observed_y, fx, fy, s, xc, yc)));
        }
    
        double _observed_x;
        double _observed_y;
        //intrisic parameters
        double _fx;
        double _fy;
        double _s;
        double _xc;
        double _yc;
    };
};
