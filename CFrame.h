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
    CFrame();
    CFrame(cv::Mat frameIn);
    ~CFrame();
    CFrame(const CFrame &frame);
    CFrame& operator= (const CFrame &frame);
    
    void setFrame(Mat frameIn, int frameNo);
    void setFrame(Mat frameIn, int frameNo, Matx33d K);
    void setIntrinsic(Matx33d K);
    void setKeyPoints(const vector<KeyPoint> &kp, const Mat desc);
    void setKeyPoints(const vector<KeyPoint> &kp);

    void setPoints(const vector<Point2d> &p);
    void setPoints(const vector<Point2f> &p);
    
    const Mat& getFrame() {return _frame;};
    const Mat& getFrameGrey() {return _frameGrey;};
    
    void setPose();
    void setPose(const Matx33d &R, const Vec3d &t);
    
    const vector<KeyPoint>& getKeyPoints() const {return _keypts;};
    const vector<Point2d>& getPoints() const {return _pts;};
    void getPoints(vector<Point2f> &pts);
    void getPoints(vector<Point2d> &pts);
    
    const Mat& getDescriptors() const {return _descriptors;};
    const Matx34d& getProjectionMatrix() const {return _P;};
    Mat getRotationRodrigues();
    const Vec3d& getTranslation() const {return _t;};
    double* getTranslationMutable() {return _t.val;};
    const Matx33d& getRotation() const {return _R;};
    
    int findClosestPointIndex(Point2f pt);
    
    int getNPoints() {return _pts.size();}
    int getNMatchedPoints();
    void getMatchedPoints(vector<Point2d> &pts2D, vector<int> &pts3DIdx);
    void getMatchedPoints(vector<Point2d> &pts2D, vector<int> &pts3DIdx, vector<int> &pts2DIdx);
    int getFrameNo() {return _frameNo;};
    
    void updatePoints(const vector<Point2d> &pts, const vector<int> &idx, const vector<int> &idx3D);
    void updatePoints(const vector<Point2f> &pts, const vector<int> &idx, const vector<int> &idx3D);
    void updatePoints(const vector<int> &pts2DIdx, const vector<int> &pts3DIdx);
    
private:
    void calculateProjectionMatrix();
    
    int _frameNo;
    
    //image data
    Mat _frame;
    Mat _frameGrey;
    
    //feature data
    Mat _descriptors;
    vector<KeyPoint> _keypts;
    vector<Point2d> _pts;
    vector<uchar> _status; //check if point has been matched already
    vector<float> _statusDist;
    int _nMatched;
    
    //3d links
    vector<int> _pts3DIdx;
    
   // flann::IndexParams _params;
    
    //camera data
    Matx33d _K;
    
    //pose data
    Matx33d _R;
    Vec3d _t;
    Matx34d _P;
};


#endif /* CFrame_hpp */
