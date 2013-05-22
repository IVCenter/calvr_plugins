#ifndef GRAPH_GLOBALS_H
#define GRAPH_GLOBALS_H

#include <osgText/Text>
#include <osg/Vec3>
#include <osg/Vec4>

#include <string>
#include <map>

class GraphGlobals
{
    public:
        static osgText::Text * makeText(std::string text, osg::Vec4 color);
        static void makeTextFit(osgText::Text * text, float maxSize);
        
        static const osg::Vec4 & getBackgroundColor();
        static const osg::Vec4 & getDataBackgroundColor();

        static const std::map<std::string,osg::Vec4> & getPhylumColorMap();
        static osg::Vec4 getDefaultPhylumColor();

    protected:
        static void checkInit();
        static void init();

        static bool _init;
        
        static osg::ref_ptr<osgText::Font> _font;
        static osg::Vec4 _bgColor;
        static osg::Vec4 _dataBGColor;

        static std::map<std::string,osg::Vec4> _phylumColorMap;
        static osg::Vec4 _defaultPhylumColor;
};

#endif
