#ifndef PLUGIN_PANO_H
#define PLUGIN_PANO_H

#include <opencv2/core/mat.hpp>
#include <opencv2/stitching.hpp>

class panoStitcher {
private:
    cv::Mat _pano;
    float * _lastCamPos;
    int pressTime = 0;

    cv::Ptr<cv::Stitcher>  _stitcher;
    std::vector<cv::Mat> _imgs, _panos;

    const double kDistanceCoef = 4.0;
    const int kMaxMatchingSize = 50;

    bool check_stitch_condition();
public:
    static panoStitcher * mPtr;
    static panoStitcher * instance();
    panoStitcher();
    void StitchImage(cv::Mat mat);
    void StitchCurrentView();

    unsigned char * getPanoImageData(int & width, int & height);
    cv::Mat getPanoImage(){return _pano;}
};

#endif
