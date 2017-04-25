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
#include "../../cvUtils/Hashing.h"
#include "../../cvUtils/VectorUtils.hpp"
#include <unordered_map>
#include <bitset>
#include <list>

using namespace std;
using namespace cv;

class CMap {
public:
    CMap();
    ~CMap();
    
    void addPointMatches(const vector<int> &pts3DIdx, const vector<int> &pts2DIdx, const int frameNo);
    void addNewPoints(const vector<Matx31d> &pts3D, const vector<vector<int>> &ptIdx, const vector<int> &frameIdx, vector<int> &pts3DIdx);
    void addDescriptors(const vector<int> &pts3DIdx, const Mat &descriptors);
    
    void removePoints(const vector<int> &ptsCullIdx, vector<int> &newPtsIdx);
    void removePoints(int threshold, vector<int> &ptsCullIdx, vector<int> &newPtsIdx);
    void removeFrame(const int frameNo);
    
    
    void getPointsAtIdx(const vector<int> &pts3DIdx, vector<Matx31d> &pts3D);
    void getPointsInFrame(vector<Matx31d> &pts3D, vector<int> &pts2DIdx, const int frameNo);
    void getPointsInFrame(vector<int> &pts3DIdx, const int frameNo);
    void getPointsInFrame(vector<Matx31d> &pts3D, vector<int> &pts3DIdx, vector<int> &pts2DIdx, const int frameNo);
    void getPointsInFrame_Mutable(vector<double*> &pts3D, vector<int> &pts3DIdx, vector<int> &pts2DIdx, const int frameNo);
    void getPointsInFrame_Mutable(vector<double*> &pts3D, vector<int> &pts2DIdx, const int frameNo);
    void getPointsInFrame(vector<int> &pts3DIdx, vector<int> &pts2DIdx, const int frameNo);
    void getPointsInFrames(vector<Matx31d> &pts3D, vector<int> &pts3DIdx, const vector<int> &frameNo);
    void getPointsInFrames(vector<int> &pts3DIdx, const vector<int> &frameNo);
    
    void getPoints_Mutable(vector<double*> &pts3D);
    void getPoints(vector<Matx31d> &pts3D);
    void getFramesConnectedToFrame(int frameNo, vector<int> &covisibleFrames, int threshold = 0);
    void getRepresentativeDescriptors(const vector<int> &pts3DIdx, Mat &descriptors);
    
    int getPointFrameVisibility(const int pt3DIdx);
    
    int getNPoints();
    
    Point3d getCentroid();

private:
    
    int countMatchesBetweenFrames(int f0, int f1);
    void addCovisiblePoint(int idx0, int idx1, int increment = 1);
    void updateCentroid();
    
    
    vector<Matx31d> _pts3D;
    vector<Mat> _descriptor;
    vector<vector<Matx31d>> _viewDir;
    vector<vector<int>> _frameNo;
    vector<vector<int>> _pts2DIdx;
    vector<int> _pts3DIdx;
    int _lastPtNo;
    
    unordered_multimap<int,int> _covisibilityFrameIdx;
    unordered_map<tuple<int,int>,int> _covisibilityGraph;
    unordered_multimap<int,int> _pointInFrameIdx;
    unordered_multimap<int,int> _frameViewsPointIdx;
    
    Point3d _centroid;
};


#endif /* CMap_hpp */

