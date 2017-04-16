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

Point3d CTracker::linearTriangulation(const Matx34d &P0, const Matx34d &P1, const Point3d pt0, const Point3d pt1, int iter) {
    
    //TODO: include two or three equations from each image?
    //TODO: currently using inhomogeneous solution
    Mat X;
    double wi = 1, wi1 = 1, p2x = 0, p2x1 = 0;
    double eps = 1e-04;
    for(int i =0; i < iter; i++) {

        Matx43d A((pt0.x*P0(2,0) - P0(0,0))/wi, (pt0.x*P0(2,1) - P0(0,1))/wi, (pt0.x*P0(2,2) - P0(0,2))/wi,
                  (pt0.y*P0(2,0) - P0(1,0))/wi, (pt0.y*P0(2,1) - P0(1,1))/wi, (pt0.y*P0(2,2) - P0(1,2))/wi,
                  (pt1.x*P1(2,0) - P1(0,0))/wi1, (pt1.x*P1(2,1) - P1(0,1))/wi1, (pt1.x*P1(2,2) - P1(0,2))/wi1,
                  (pt1.y*P1(2,0) - P1(1,0))/wi1, (pt1.y*P1(2,1) - P1(1,1))/wi1, (pt1.y*P1(2,2) - P1(1,2))/wi1);
        Matx41d B(-(pt0.x*P0(2,3) - P0(0,3))/wi, -(pt0.y*P0(2,3) - P0(1,3))/wi, -(pt1.x*P1(2,3) - P1(0,3))/wi1, -(pt1.y*P1(2,3) - P1(1,3))/wi1);
        
        solve(A,B,X,DECOMP_SVD);
        //cout << "Iter " << i << ": " << X << endl;
        
        //check if time to break
        Matx41d xcol(X.ptr<double>(0)[0],X.ptr<double>(0)[1],X.ptr<double>(0)[2],1.0);
        p2x = (P0.row(2)*xcol)(0);
        p2x1 = (P1.row(2)*xcol)(0);
        
        if ((fabs(wi - p2x) <= eps) && (fabs(wi1 - p2x1) <= eps))
            break;
        
        //update weights
        wi = p2x;
        wi1 = p2x1;
    }
    Point3d sol(X.ptr<double>(0)[0],X.ptr<double>(0)[1],X.ptr<double>(0)[2]);
    return sol;
}

void CTracker::triangulatePoints(const Matx34d &P0, const Matx34d &P1, const Matx33d &K0, const Matx33d &K1, const vector<Point2d> &f0, const vector<Point2d> &f1, vector<Point3d> &outPts) {
    
    //preallocate for speed
    outPts.reserve(f0.size());
    
    Matx33d K0i = K0.inv();
    Matx33d K1i = K1.inv();
    for (int i = 0; i < f0.size(); i++) {
        Point3d pt0(f0[i].x,f0[i].y,1);
        Point3d pt1(f1[i].x,f1[i].y,1);
        //convert to normalised coordinates
        Point3d pt0n = K0i*pt0;
        Point3d pt1n = K1i*pt1;
        //solve linear system
        Point3d X = linearTriangulation(P0, P1, pt0n, pt1n, 10);
        outPts.push_back(X);
    }
}

void CTracker::triangulatePoints(const Matx34d &P0, const Matx34d &P1, const Matx33d &K0, const Matx33d &K1, const vector<Point2f> &f0, const vector<Point2f> &f1, vector<Point3d> &outPts) {
    
    //preallocate for speed
    outPts.reserve(f0.size());
    
    Matx33d K0i = K0.inv();
    Matx33d K1i = K1.inv();
    for (int i = 0; i < f0.size(); i++) {
        Point3d pt0(f0[i].x,f0[i].y,1);
        Point3d pt1(f1[i].x,f1[i].y,1);
        //convert to normalised coordinates
        Point3d pt0n = K0i*pt0;
        Point3d pt1n = K1i*pt1;
        //solve linear system
        Point3d X = linearTriangulation(P0, P1, pt0n, pt1n, 10);
        outPts.push_back(X);
    }
}


bool CTracker::RtFromEssentialMatrix(const Matx33d &E, const Matx33d &K, const vector<Point2d> &pts0, const vector<Point2d> &pts1,Matx33d &R, Vec3d &t) {
    //find SVD of the essential matrix
    SVD svd(E,SVD::MODIFY_A);
    
    const double minSVDRatio = 0.7;
    const double minGoodRatio = 0.85;
//    //debug
//    cout << "U: " << svd.u << endl;
//    cout << "W: " << svd.w << endl;
//    cout << "V: " << svd.vt << endl;
    
    //two singular values should be equal and the third zero
    double ratio = fabs(svd.w.ptr<float>(0)[0]/svd.w.ptr<float>(0)[1]);
    if (ratio < 0.7) {
        cerr << "singular values too far apart" << endl;
        return false;
    }
 
    Matx33d W(0,-1,0,1,0,0,0,0,1);
    Matx33d Wt(0,1,0,-1,0,0,0,0,1);
    
    Mat R0 = svd.u*Mat(W)*svd.vt;
    Mat R1 = svd.u*Mat(Wt)*svd.vt;
    Mat t0 = svd.u.col(2);
    Mat t1 = -svd.u.col(2);
    
    //calculate determinant of rotation to check validity of essential matrix
    double d = determinant(R0);
    double tol = 1e-05;
    if (d + 1.0 < tol) {
        svd(-E,SVD::MODIFY_A);
        R0 = svd.u*Mat(W)*svd.vt;
        R1 = svd.u*Mat(Wt)*svd.vt;
        t0 = svd.u.col(2);
        t1 = -svd.u.col(2);
        d = determinant(R0);
    }
    if (d -1.0 > tol) {
        cerr << "Not a proper rotation" << endl;
        return false;
    }
    
    //test all possibilities
    Matx34d P0(1,0,0,0,0,1,0,0,0,0,1,0);
    vector<Mat> rots{R0,R1};
    vector<Mat> trans{t0,t1};
    vector<Point3d> pts3D;
    int bestCount = 0, bestRIdx, bestTIdx;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            //triangulate points
            //TODO: only triangulate a subset to limit complexity?
            pts3D.clear();
            Matx34d P(rots[i].ptr<float>(0)[0],rots[i].ptr<float>(0)[1],rots[i].ptr<float>(0)[2],trans[j].ptr<float>(0)[0],rots[i].ptr<float>(1)[0],rots[i].ptr<float>(1)[1],rots[i].ptr<float>(1)[2],trans[j].ptr<float>(0)[1],rots[i].ptr<float>(2)[0],rots[i].ptr<float>(2)[1],rots[i].ptr<float>(2)[2],trans[j].ptr<float>(0)[2]);
            triangulatePoints(P0,P,K,K,pts0,pts1,pts3D);
            
            //check if points are in front of the camera
            int countGood = 0;
            for (int k = 0; k < pts3D.size(); k++) {
                if (pts3D[k].z > 0)
                    countGood++;
            }
            //save best transformation
            if (countGood > bestCount) {
                bestCount = countGood;
                bestRIdx = i;
                bestTIdx = j;
            }
        }
    }
    
    if (bestCount/pts3D.size() < minGoodRatio) {
        cerr << "No valid rotations/translations" << endl;
        return false;
    }
    
    R = rots[bestRIdx];
    t = trans[bestTIdx];
    
    return true;
}

bool CTracker::RtFromEssentialMatrix(const Matx33f &E, const Matx33f &K, const vector<Point2f> &pts0, const vector<Point2f> &pts1,Matx33d &R, Vec3d &t) {
    //find SVD of the essential matrix
    SVD svd(E,SVD::MODIFY_A);
    
    const double minSVDRatio = 0.7;
    const double minGoodRatio = 0.85;
    //    //debug
    //    cout << "U: " << svd.u << endl;
    //    cout << "W: " << svd.w << endl;
    //    cout << "V: " << svd.vt << endl;
    
    //two singular values should be equal and the third zero
    double ratio = fabs(svd.w.ptr<float>(0)[1]/svd.w.ptr<float>(0)[0]);
    if (ratio < 0.7) {
        cerr << "singular values too far apart" << endl;
        return false;
    }
    
    Matx33f W(0,-1,0,1,0,0,0,0,1);
    Matx33f Wt(0,1,0,-1,0,0,0,0,1);
    
    Mat R0 = svd.u*Mat(W)*svd.vt;
    Mat R1 = svd.u*Mat(Wt)*svd.vt;
    Mat t0 = svd.u.col(2);
    Mat t1 = -svd.u.col(2);
    
    //calculate determinant of rotation to check validity of essential matrix
    double d = determinant(R0);
    double tol = 1e-05;
    if (d + 1.0 < tol) {
        svd(-E,SVD::MODIFY_A);
        R0 = svd.u*Mat(W)*svd.vt;
        R1 = svd.u*Mat(Wt)*svd.vt;
        t0 = svd.u.col(2);
        t1 = -svd.u.col(2);
        d = determinant(R0);
    }
    if (d -1.0 > tol) {
        cerr << "Not a proper rotation" << endl;
        return false;
    }
    
    //test all possibilities
    Matx34d P0(1,0,0,0,0,1,0,0,0,0,1,0);
    vector<Mat> rots{R0,R1};
    vector<Mat> trans{t0,t1};
    vector<Point3d> pts3D;
    int bestCount = 0, bestRIdx, bestTIdx;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            //triangulate points
            //TODO: only triangulate a subset to limit complexity?
            pts3D.clear();
            Matx34d P(rots[i].ptr<float>(0)[0],rots[i].ptr<float>(0)[1],rots[i].ptr<float>(0)[2],trans[j].ptr<float>(0)[0],rots[i].ptr<float>(1)[0],rots[i].ptr<float>(1)[1],rots[i].ptr<float>(1)[2],trans[j].ptr<float>(0)[1],rots[i].ptr<float>(2)[0],rots[i].ptr<float>(2)[1],rots[i].ptr<float>(2)[2],trans[j].ptr<float>(0)[2]);
            triangulatePoints(P0,P,K,K,pts0,pts1,pts3D);
            
            //check if points are in front of the camera
            int countGood = 0;
            for (int k = 0; k < pts3D.size(); k++) {
                if (pts3D[k].z > 0)
                    countGood++;
            }
            //save best transformation
            if (countGood > bestCount) {
                bestCount = countGood;
                bestRIdx = i;
                bestTIdx = j;
            }
        }
    }
    
    if ((float)bestCount/pts3D.size() < minGoodRatio) {
        cerr << "No valid rotations/translations" << endl;
        return false;
    }
    
    R = Matx33d(rots[bestRIdx]);
    t = Vec3d(trans[bestTIdx]);
    
    return true;
}

bool CTracker::RtFromHomographyMatrix(const Matx33f &H, const Matx33f &K, const vector<Point2f> &pts0, const vector<Point2f> &pts1, Matx33d &R, Vec3d &t) {
    const double minGoodRatio = 0.85;
    
    //find all possible decompositions
    vector<Mat> rots;
    vector<Mat> trans;
    vector<Mat> nh;
    decomposeHomographyMat(H, K, rots, trans, nh);
    
    //try triangulating
    Matx34d P0(1,0,0,0,0,1,0,0,0,0,1,0);
    vector<Point3d> pts3D;
    int bestCount = 0, bestIdx;
    for (int i = 0; i < rots.size(); i++) {
       
        //triangulate points
        //TODO: only triangulate a subset to limit complexity?
        pts3D.clear();
        Matx34d P(rots[i].ptr<double>(0)[0],rots[i].ptr<double>(0)[1],rots[i].ptr<double>(0)[2],trans[i].ptr<double>(0)[0],rots[i].ptr<double>(1)[0],rots[i].ptr<double>(1)[1],rots[i].ptr<double>(1)[2],trans[i].ptr<double>(0)[1],rots[i].ptr<double>(2)[0],rots[i].ptr<double>(2)[1],rots[i].ptr<double>(2)[2],trans[i].ptr<double>(0)[2]);
        triangulatePoints(P0,P,K,K,pts0,pts1,pts3D);
        
        //check if points are in front of the camera
        int countGood = 0;
        for (int k = 0; k < pts3D.size(); k++) {
            if (pts3D[k].z > 0)
                countGood++;
        }
       
        //save best transformation
        if (countGood > bestCount) {
            bestCount = countGood;
            bestIdx = i;
        }
    }
    
    if ((float)bestCount/pts3D.size() < minGoodRatio) {
        cerr << "No valid rotations/translations" << endl;
        return false;
    }
    
    R = Matx33d(rots[bestIdx]);
    t = Vec3d(trans[bestIdx]);

    return true;
}

double CTracker::distancePointLine2D(const Point2d &pt, const Vec3d &l) {
    return (l[0]*pt.x + l[1]*pt.y + l[2])*(l[0]*pt.x + l[1]*pt.y + l[2])/(l[0]*l[0] + l[1]*l[1]);
}

double CTracker::calculateFundamentalAvgError(const vector<Point2d> &pts0, const vector<Point2d> &pts1, const Matx33d &F) {
    //return average symmetric distance from epilines
    double e = 0;
    
    //compute epipolar lines
    vector<Vec3d> epiLines0, epiLines1;
    computeCorrespondEpilines(pts0, 1, F, epiLines1);
    computeCorrespondEpilines(pts1, 2, F, epiLines0);
    
    double e01,e10;
    int count = 0;
    for (int i = 0; i < pts0.size(); i++) {
        //compute distance from epilines
        e01 = CTracker::distancePointLine2D(pts0[i], epiLines0[i]);
        e10 = CTracker::distancePointLine2D(pts1[i], epiLines1[i]);
        e += e10 + e01;
        count++;
    }
    
    return e/count;
}

float CTracker::calculateFundamentalAvgError(const vector<Point2f> &pts0, const vector<Point2f> &pts1, const Matx33f &F) {
    //return average symmetric distance from epilines
    double e = 0;
    
    //compute epipolar lines
    vector<Vec3f> epiLines0, epiLines1;
    computeCorrespondEpilines(pts0, 1, F, epiLines1);
    computeCorrespondEpilines(pts1, 2, F, epiLines0);
    
    float e01,e10;
    int count = 0;
    for (int i = 0; i < pts0.size(); i++) {
        //compute distance from epilines
        e01 = CTracker::distancePointLine2D(pts0[i], epiLines0[i]);
        e10 = CTracker::distancePointLine2D(pts1[i], epiLines1[i]);
        e += e10 + e01;
        count++;
    }
    
    return e/count;
}

double CTracker::calculateHomographyAvgError(const vector<Point2d> &pts0, const vector<Point2d> &pts1, const Matx33d &H) {
    double e = 0;
    //average symmetric transfer error
    
    //compute matrix inverse
    Matx33d Hinv = H.inv();
    
    //get forward and backward transformed points
    vector<Point2d> fwd, bwd;
    perspectiveTransform(pts0, fwd, H);
    perspectiveTransform(pts1, bwd, Hinv);
    
    double e01, e10;
    int count = 0;
    for (int i = 0; i < pts0.size(); i++) {
        e01 = (pts1[i].x - fwd[i].x)*(pts1[i].x - fwd[i].x) + (pts1[i].y - fwd[i].y)*(pts1[i].y - fwd[i].y);
        e10 = (pts0[i].x - bwd[i].x)*(pts0[i].x - bwd[i].x) + (pts0[i].y - bwd[i].y)*(pts0[i].y - bwd[i].y);
        e += e01 + e10;
        count++;
    }
    
    return e/count;
}

float CTracker::calculateHomographyAvgError(const vector<Point2f> &pts0, const vector<Point2f> &pts1, const Matx33f &H) {
    float e = 0;
    //average symmetric transfer error
    
    //compute matrix inverse
    Matx33f Hinv = H.inv();
    
    //get forward and backward transformed points
    vector<Point2f> fwd, bwd;
    perspectiveTransform(pts0, fwd, H);
    perspectiveTransform(pts1, bwd, Hinv);
    
    float e01, e10;
    int count = 0;
    for (int i = 0; i < pts0.size(); i++) {
        e01 = (pts1[i].x - fwd[i].x)*(pts1[i].x - fwd[i].x) + (pts1[i].y - fwd[i].y)*(pts1[i].y - fwd[i].y);
        e10 = (pts0[i].x - bwd[i].x)*(pts0[i].x - bwd[i].x) + (pts0[i].y - bwd[i].y)*(pts0[i].y - bwd[i].y);
        e += e01 + e10;
        count++;
    }
    
    return e/count;
}

//--------------------------------------//
//                                      //
//                                      //
//          BUNDLE ADJUSTMENT           //
//                                      //
//                                      //
//--------------------------------------//
CTracker::BAStructAndPoseFunctor::BAStructAndPoseFunctor(double observed_x, double observed_y, double fx, double fy, double s, double xc, double yc) {
    
    _observed_x = observed_x;
    _observed_y = observed_y;
    _fx = fx;
    _fy = fy;
    _s = s;
    _xc = xc;
    _yc = yc;
}
template <typename T> bool CTracker::BAStructAndPoseFunctor::operator()(const T *const camera, const T *const point, T *residuals) const {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        // camera[3,4,5] are the translation.
        p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];
        
        //project
        T xp =  p[0] / p[2];
        T yp =  p[1] / p[2];
        
        //intrinsics
        T predicted_x = T(_fx)*xp + T(_s)*yp + T(_xc);
        T predicted_y = T(_fy)*yp + T(_yc);
        
        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(_observed_x);
        residuals[1] = predicted_y - T(_observed_y);
        return true;
}


void CTracker::bundleAdjustmentStructAndPose(const vector<Point2d> &observations, const vector<int> &camIdx, const vector<int> &pt3DIdx, const vector<Matx33d> &K) {
    //create ceres problem
    ceres::Problem prob;
    
    //populate problem
    for (int i = 0; i < observations.size(); i++) {
        int camNo = camIdx[i];
        int ptNo = pt3DIdx[i];
        
        ceres::CostFunction* cost_function = CTracker::BAStructAndPoseFunctor::Create(observations[i].x, observations[i].y, K[camNo].val[0], K[camNo].val[4], K[camNo].val[1], K[camNo].val[2], K[camNo].val[5]);
        
        
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



