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

#ifndef CMap_hpp
#define CMap_hpp

#include "CFrame.h"

using namespace std;
using namespace cv;

class CMap {
public:
    CMap();
    ~CMap();
    
    void addPointMatches(const vector<int> &pts3DIdx, const vector<int> &pts2DIdx, const int frameNo);
    void addNewPoints(const vector<Point3d> &pts3D, const vector<vector<int>> &ptIdx, const vector<int> &frameIdx, vector<int> &pts3DIdx);
    
    void getProjectionsOnFrame(const CFrame &frame, vector<Point2d> &projPoints);
    void getPointsAtIdx(const vector<int> &pts2DIdx, vector<Point3d> &pts3D);
    void getPointsInFrame(vector<Point3d> &pts3D, vector<int> &pts2DIdx, const int frameNo);
    void getPointsInFrame(vector<Point3d> &pts3D, vector<int> &pts3DIdx, vector<int> &pts2DIdx, const int frameNo);
    void getPoints(vector<Point3d> &pts3D);
    
    int getNPoints() {return _pts3D.size();};
    
    Point3d getCentroid();

private:
    void updateCentroid();
    
    
    vector<Point3d> _pts3D;
    vector<vector<int>> _frameNo;
    vector<vector<int>> _pts2DIdx;
    vector<int> _pts3DIdx;
    int _lastPtNo;
    
    Point3d _centroid;
};


#endif /* CMap_hpp */

