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

#ifndef CFrame_hpp
#define CFrame_hpp
#include "opencv2/opencv.hpp"
#include "Eigen/Dense"
#include <stdio.h>

using namespace cv;
using namespace std;

class CFrame {
public:
    CFrame(const Matx33d &K, const vector<double> &d, const Size &imSize);
    CFrame(const Mat &frameIn, const Matx33d &K, const vector<double> &d, const Size &imSize);
    ~CFrame();
    CFrame(const CFrame &frame);
    CFrame& operator= (const CFrame &frame);
    
    //set frames
    void setFrame(const Mat &frameIn, int frameNo);
    void setFrame(const Mat &frameIn, int frameNo, const Matx33d &K, const vector<double> &d, const Size &imSize);
    
    //get frames
    const Mat& getFrame() {return _frame;};
    const Mat& getFrameGrey() {return _frameGrey;};
    
    //set features
    void setKeyPoints(const vector<KeyPoint> &kp, const Mat &desc);
    void setKeyPoints(const vector<KeyPoint> &kp);
    void setPoints(const vector<Point2d> &p);
    void setPoints(const vector<Point2f> &p);
    
    //get features
    const vector<KeyPoint>& getKeyPoints() const {return _keypts;};
    const vector<Point2d>& getPoints() const {return _pts;};
    void getPoints(vector<Point2f> &pts);
    void getPoints(vector<Point2d> &pts);
    const vector<Point2d>& getPointsDistorted() const {return _pts_dist;};
    const Mat& getDescriptors() const {return _descriptors;};
    
    //set geometric pose
    void setPose();
    void setPose(const Matx33d &R, const Matx31d &t);
    void calculateProjectionMatrix();
    
    //get geometric pose
    const Matx34d& getProjectionMatrix() const {return _P;};
    const Matx31d& getRotationRodrigues() {return _rot;};
    double *getRotationRodrigues_Mutable() {return _rot.val;};
    const Matx31d& getTranslation() const {return _t;};
    double* getTranslation_Mutable() {return _t.val;};
    const Matx33d& getRotation() const {return _R;};
    
    //get intrinsics
    const Matx33d& getIntrinsic() const {return _K;};
    const Matx33d& getIntrinsicUndistorted() const {return _Kopt;};
    int getFrameNo() {return _frameNo;};
    const Size& getImageSize() const {return _imSize;};
    
    //culling
    void cullPoints(const vector<int> &pts3DIdx);
    
    //get points
    int findClosestPointIndex(Point2f pt);
    int findClosestPointIndexDistorted(Point2f pt);
    int getNPoints() {return _pts.size();}
    int getNMatchedPoints();
    void getMatchedPoints(vector<Point2d> &pts2D, vector<int> &pts3DIdx);
    void getMatchedPoints(vector<Point2d> &pts2D, vector<int> &pts3DIdx, vector<int> &pts2DIdx);
    void getMatchedPoints(vector<int> &pts2DIdx, vector<int> &pts3DIdx);
    void getMatchedPoints(vector<int> &pts3DIdx);
    void getMatchedIndices(vector<int> &pts2DIdx);
    void getUnmatchedPoints(vector<Point2d> &pts2D, vector<int> &pts2DIdx);
    void getUnmatchedPoints(vector<Point2d> &pts2D, Mat &desc, vector<int> &pts2DIdx);
    
    void getPointsAt(const vector<int> &pts2DIdx, vector<Point2d> &pts2D);
    void getPoints3DIdxAt(const vector<int> &pts2DIdx, vector<int> &pts3DIdx);
    void getPointsDistortedAt(const vector<int> &pts2DIdx, vector<Point2d> &pts2D);
    void getPointsAt(const vector<int> &pts2DIdx, vector<Point2f> &pts2D);
    void getDescriptorsAt(const vector<int> &pts2DIdx, Mat &descriptors);
    
    //errors
    double getMeanError() {return _meanErr;};
    double getMaxError() {return _maxErr;};
    void updateFrameErrorStatistics(const double meanErr, const double maxErr);
    
    //update
    void updatePoints(const vector<Point2d> &pts, const vector<int> &idx, const vector<int> &idx3D);
    void updatePoints(const vector<Point2f> &pts, const vector<int> &idx, const vector<int> &idx3D);
    void updatePoints(const vector<int> &pts2DIdx, const vector<int> &pts3DIdx);
    
private:
    
    void resetInternals();
    
    int _frameNo;
    
    //image data
    Mat _frame;
    Mat _frameGrey;
    
    //feature data
    Mat _descriptors;
    vector<KeyPoint> _keypts;
    vector<Point2d> _pts_dist;
    vector<Point2d> _pts;
    vector<uchar> _status; //check if point has been matched already
    vector<float> _statusDist;
    int _nMatched;
    
    //3d links
    vector<int> _pts3DIdx;
    
    //camera data
    Matx33d _K;
    vector<double> _d;
    Matx33d _Kopt;
    Size _imSize;
    
    //pose data
    Matx33d _R;
    Matx31d _rot;
    Matx31d _t;
    Matx34d _P;
    
    //error statistics
    double _meanErr;
    double _maxErr;
};


#endif /* CFrame_hpp */
