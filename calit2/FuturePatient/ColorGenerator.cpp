#include "ColorGenerator.h"

osg::Vec4 ColorGenerator::_defaultColors[7] = { osg::Vec4(0.89412,0.10196,0.109804,1.0), osg::Vec4(0.21569,0.49412,0.72157,1.0), osg::Vec4(0.302,0.6863,0.2902,1.0), osg::Vec4(0.59608,0.305882,0.63922,1.0), osg::Vec4(1.0,0.5,0,1.0), osg::Vec4(1.0,1.0,0.2,1.0), osg::Vec4(0.6510,0.3373,0.15686,1.0) };

osg::Vec4 ColorGenerator::makeColor(int colorNum, int totalColors)
{
    if(colorNum >= totalColors)
    {
	return osg::Vec4(1,1,1,1);
    }

    if(totalColors <= 7)
    {
	return _defaultColors[colorNum];
    }
    else
    {
	return makeColor(((float)colorNum)/((float)totalColors));
    }
}

osg::Vec4 ColorGenerator::makeColor(float f)
{
    if(f < 0)
    {
        f = 0;
    }
    else if(f > 1.0)
    {
        f = 1.0;
    }

    osg::Vec4 color;
    color.w() = 1.0;

    if(f <= 0.33)
    {
        float part = f / 0.33;
        float part2 = 1.0 - part;

        color.x() = part2;
        color.y() = part;
        color.z() = 0;
    }
    else if(f <= 0.66)
    {
        f = f - 0.33;
        float part = f / 0.33;
        float part2 = 1.0 - part;

        color.x() = 0;
        color.y() = part2;
        color.z() = part;
    }
    else if(f <= 1.0)
    {
        f = f - 0.66;
        float part = f / 0.33;
        float part2 = 1.0 - part;

        color.x() = part;
        color.y() = 0;
        color.z() = part2;
    }

    //std::cerr << "Color x: " << color.x() << " y: " << color.y() << " z: " << color.z() << std::endl;

    return color;
}
