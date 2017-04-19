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

CTracker::CTracker() {
    
    
}

CTracker::~CTracker() {
    
    
}
                


//--------------------------------------//
//                                      //
//                                      //
//          BUNDLE ADJUSTMENT           //
//                                      //
//                                      //
//--------------------------------------//
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

void CTracker::bundleAdjustmentStructAndPose(const vector<Point2d> &observations, const vector<int> &camIdx, const vector<int> &pt3DIdx, const vector<Matx33d> &K, vector<double*> &R, vector<double*> &t, vector<double*> &pts3D, int isStructOrPose) {
    //create ceres problem
    ceres::Problem prob;
    
    //populate problem
    ceres::CostFunction* cost_function;
    for (int i = 0; i < observations.size(); i++) {
        int camNo = camIdx[i];
        int ptNo = pt3DIdx[i];
        
        
        switch (isStructOrPose) {
            case 0:
                cost_function = CTracker::BAStructFunctor::Create(observations[i].x, observations[i].y, R[camNo], t[camNo], K[camNo].val);
                prob.AddResidualBlock(cost_function, NULL, pts3D[ptNo]);
                break;
            case 1:
                cost_function = CTracker::BAPoseFunctor::Create(observations[i].x, observations[i].y, pts3D[ptNo], K[camNo].val);
                prob.AddResidualBlock(cost_function, NULL, R[camNo], t[camNo]);
                break;
            case 2:
                cost_function = CTracker::BAStructAndPoseFunctor::Create(observations[i].x, observations[i].y, K[camNo].val);
                prob.AddResidualBlock(cost_function, NULL, R[camNo], t[camNo], pts3D[ptNo]);
                break;
            default:
                break;
        }
        
            }
    
    
    //create solution options
    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_SCHUR;
    opts.minimizer_progress_to_stdout = false;
    opts.logging_type = ceres::LoggingType::SILENT;
    
    //solve BA
    ceres::Solver::Summary summ;
    ceres::Solve(opts, &prob, &summ);
}


