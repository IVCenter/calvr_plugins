#ifndef SUB_IMAGE_INFO_H
#define SUB_IMAGE_INFO_H

struct SubImageInfo
{
	bool takeImage;
        osg::Vec3 center;
        float width, height;
	osg::ref_ptr<osg::Camera> camera;
	osg::ref_ptr<osg::Image> image;
	osg::ref_ptr<osg::Texture2D> depthTex;
};

#endif
