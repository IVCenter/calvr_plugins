#ifndef VROOM_VIEW_PLUGIN_H
#define VROOM_VIEW_PLUGIN_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrUtil/MultiListenSocket.h>
#include <cvrUtil/CVRSocket.h>

#include <osg/Camera>
#include <osg/Image>
#include <osg/Texture2D>

#include <vector>

struct SubImageInfo
{
	bool takeImage, imageDone;
	osg::ref_ptr<osg::Camera> camera;
	osg::ref_ptr<osg::Image> image;
	osg::ref_ptr<osg::Texture2D> depthTex;
	cvr::CVRSocket * socket;
};

class VroomView : public cvr::CVRPlugin
{
public:
    VroomView();
    virtual ~VroomView();

    bool init();
    void preFrame();
    
protected:
    std::vector<SubImageInfo*> _imageInfoList;
    cvr::MultiListenSocket * _mls;
};

#endif
