#ifndef FP_COLOR_GENERATOR_H
#define FP_COLOR_GENERATOR_H

#include <osg/Vec4>

class ColorGenerator
{
    public:
        static osg::Vec4 makeColor(int colorNum, int totalColors);

    protected:
        static osg::Vec4 makeColor(float f);

        static osg::Vec4 _defaultColors[7];
};

#endif
