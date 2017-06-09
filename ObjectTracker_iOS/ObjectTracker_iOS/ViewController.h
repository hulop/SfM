//
//  ViewController.h
//  OpenCVTest
//
//  Created by Marco on 1/19/17.
//  Copyright © 2017 IBM. All rights reserved.
//

#import <opencv2/videoio/cap_ios.h>
#import <opencv2/imgcodecs/ios.h>
using namespace cv;
#import <UIKit/UIKit.h>


@interface ViewController : UIViewController <CvVideoCameraDelegate>
{
    CvVideoCamera* videoCamera;
    BOOL isCapturing;
}

@property (nonatomic, strong) CvVideoCamera *videoCamera;
@property (nonatomic, strong) IBOutlet UIImageView *viewFinder;
@property (nonatomic, strong) IBOutlet UIToolbar* toolbar;
@property (nonatomic, strong) IBOutlet UIBarButtonItem* startCaptureButton;
@property (nonatomic, strong) IBOutlet UIBarButtonItem* stopCaptureButton;
@property cv::Size frameSize;

-(IBAction)startCaptureButtonPressed:(id)sender;
-(IBAction)stopCaptureButtonPressed:(id)sender;
@end

