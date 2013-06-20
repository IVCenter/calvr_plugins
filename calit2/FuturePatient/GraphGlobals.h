#ifndef GRAPH_GLOBALS_H
#define GRAPH_GLOBALS_H

#include <osgText/Text>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Drawable>

#include <string>
#include <map>

enum FPAxisType
{
    FPAT_LINEAR=0,
    FPAT_LOG
};

class GraphGlobals
{
    public:
        static osgText::Text * makeText(std::string text, osg::Vec4 color);
        static void makeTextFit(osgText::Text * text, float maxSize, bool horizontal = true);
        
        static const osg::Vec4 & getBackgroundColor();
        static const osg::Vec4 & getDataBackgroundColor();

        static const std::map<std::string,osg::Vec4> & getPhylumColorMap();
        static osg::Vec4 getDefaultPhylumColor();

        static const std::map<std::string,osg::Vec4> & getPatientColorMap();

    protected:
        static void checkInit();
        static void init();

        static bool _init;
        
        static osg::ref_ptr<osgText::Font> _font;
        static osg::Vec4 _bgColor;
        static osg::Vec4 _dataBGColor;

        static std::map<std::string,osg::Vec4> _phylumColorMap;
        static osg::Vec4 _defaultPhylumColor;

        static std::map<std::string,osg::Vec4> _patientColorMap;
};

struct SetBoundsCallback : public osg::Drawable::ComputeBoundingBoxCallback
{
    osg::BoundingBox computeBound(const osg::Drawable &) const
    {
        return bbox;
    }
    osg::BoundingBox bbox;
};

#endif
