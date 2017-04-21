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

#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "Eigen/Dense"
#include "CSfM.h"
#include "../../STLReader/CSTLReader.hpp"

using namespace cv;
using namespace std;


int main(int argc, const char * argv[]) {
    
    cout << "OpenCV version: " << CV_VERSION << endl;
    
    //load video
    string videoInput = "/Users/marco/Box Sync/Data/testVideos/tissuebox/tissuebox.mov";
    namedWindow("Input");
    VideoCapture videoIn(videoInput);
    
    if (!videoIn.isOpened())
        cerr << "Could not open video file" << endl;
    
    //camera information (iPhone 6s front camera)
    double fx = 1072.606693272117800;
    double fy = 1067.197515608619600;
    double xc = 648.780750477178910;
    double yc = 364.503435962496890;
    Matx33d K(fx, 0, xc, 0, fy, yc, 0, 0, 1);
    vector<double> d = {0.039530469484242, -0.149827721353905, -0.000091102042149, 0.001209830654934};
    int width = videoIn.get(CAP_PROP_FRAME_WIDTH);
    int height = videoIn.get(CAP_PROP_FRAME_HEIGHT);
    int frameCount = videoIn.get(CAP_PROP_FRAME_COUNT);
    Size imSize(width,height);
    
    //create class for 3d file output
    STLReader stl;
    
    //instantiate slam engine
    CSfM sfm(K, imSize, d);
   // sfm.startVideoOutput("/Users/marco/Box Sync/Data/testVideos/tissuebox/reproj.avi",videoIn.get(CAP_PROP_FOURCC));

#ifdef DEBUGINFO
  //  sfm.startVideoOutput("/Users/marco/Box Sync/Data/testVideos/tissuebox/tracker.avi",videoIn.get(CAP_PROP_FOURCC),imSize);
#endif
    
    //read loop
    Mat firstFrame, frameIn;
    for (int i = 0; i < 350; i++) {
#ifdef DEBUGINFO
        cout << "Frame " << i << endl;
#endif
        videoIn >> frameIn;
        
        //imshow("Input",frameIn);
        //waitKey();
        sfm.addFrame(frameIn);
        int key = waitKey(1) & 255;
        if (key != 255)
            break;
    }

#ifdef DEBUGINFO
   // sfm.stopVideoOutput();
#endif
    
    //output 3d object
    cout << "Outputting 3D volume..." << endl;
    vector<Matx31d> volume;
    vector<Vec3i> colour;
    string volumeOut = "/Users/marco/Box Sync/Data/testVideos/checkerboard/3dOut.ply";
    sfm.getReconstruction(volume, colour);
    stl.addPointsToCloud(volume);
    //stl.normaliseVolume();
    stl.centerVolume();
    stl.scaleVolume(500);
    stl.writePLYPointCloud(volumeOut);
    cout << "Complete!" << endl;
    
    return 0;
}
