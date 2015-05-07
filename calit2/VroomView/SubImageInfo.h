#ifndef SUB_IMAGE_INFO_H
#define SUB_IMAGE_INFO_H

#include <osg/Camera>
#include <osg/Image>
#include <osg/Texture2D>

#include <string>

struct SubImageInfo
{
        std::string label;
	bool takeImage;
        osg::Vec3 center;
        float width, height;
	osg::ref_ptr<osg::Camera> camera;
	osg::ref_ptr<osg::Image> image;
	osg::ref_ptr<osg::Texture2D> depthTex;
};

#endif
