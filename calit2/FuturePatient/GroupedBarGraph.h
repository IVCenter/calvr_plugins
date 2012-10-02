#ifndef FP_GROUPED_BAR_GRAPH_H
#define FP_GROUPED_BAR_GRAPH_H

#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/Geometry>
#include <osgText/Text>

#include <string>
#include <map>

enum BarGraphAxisType
{
    BGAT_LINEAR=0,
    BGAT_LOG
};

class GroupedBarGraph
{
    public:
        GroupedBarGraph(float width, float height);
        virtual ~GroupedBarGraph();

        bool setGraph(std::string title, std::map<std::string, std::vector<std::pair<std::string, float> > > & data, std::vector<std::string> & groupOrder, BarGraphAxisType axisType, std::string axisLabel, std::string axisUnits, std::string groupLabel, osg::Vec4 color);

        osg::Group * getRootNode()
        {
            return _root.get();
        }

    protected:
        void makeGraph();
        void makeBG();
        void update();
        void updateGraph();
        void updateAxis();

        osgText::Text * makeText(std::string text, osg::Vec4 color);
        void makeTextFit(osgText::Text * text, float maxSize);

        float _width, _height;
        float _maxGraphValue, _minGraphValue;
        float _topPaddingMult, _leftPaddingMult, _maxBottomPaddingMult, _currentBottomPaddingMult;
        int _numBars;
        float _minDisplayRange;
        float _maxDisplayRange;

        std::string _title, _axisLabel, _axisUnits, _groupLabel;
        BarGraphAxisType _axisType;
        std::map<std::string, std::vector<std::pair<std::string, float> > > _data;
        std::vector<std::string> _groupOrder;
        osg::Vec4 _color;

        osg::ref_ptr<osg::Group> _root;
        osg::ref_ptr<osg::MatrixTransform> _bgScaleMT;
        osg::ref_ptr<osg::Geode> _barGeode;
        osg::ref_ptr<osg::Geode> _axisGeode;
        osg::ref_ptr<osg::Geode> _bgGeode;
        osg::ref_ptr<osg::Geometry> _barGeom;

        osg::ref_ptr<osgText::Font> _font;
};

#endif
