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

        static osg::Vec4 getColorLow();
        static osg::Vec4 getColorNormal();
        static osg::Vec4 getColorHigh1();
        static osg::Vec4 getColorHigh10();
        static osg::Vec4 getColorHigh100();

        static const std::map<std::string,osg::Vec4> & getPatientColorMap();

        static bool getDeferUpdate();
        static void setDeferUpdate(bool defer);

    protected:
        static void checkInit();
        static void init();

        static bool _init;
        static bool _deferUpdate;
        
        static osg::ref_ptr<osgText::Font> _font;
        static osg::Vec4 _bgColor;
        static osg::Vec4 _dataBGColor;

        static osg::Vec4 _lowColor;
        static osg::Vec4 _normColor;
        static osg::Vec4 _high1Color;
        static osg::Vec4 _high10Color;
        static osg::Vec4 _high100Color;

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
