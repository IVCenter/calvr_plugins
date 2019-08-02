#pragma once

#include <memory>
#include <string>
#include <osg/Image>
#include <osg/Vec3>
#include <osg/Texture>


// Win32 LoadImage macro
#ifdef LoadImage
#undef LoadImage
#endif

class ImageLoader {
public:
	static osg::Image* LoadImage(const std::string& imagePath, osg::Vec3& size);
	static osg::Image* LoadVolume(const std::string& folder, osg::Vec3& size);
};