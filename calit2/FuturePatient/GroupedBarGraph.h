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

enum BarGraphColorMode
{
    BGCM_SOLID=0,
    BGCM_GROUP
};

class GroupedBarGraph
{
    public:
        GroupedBarGraph(float width, float height);
        virtual ~GroupedBarGraph();

        bool setGraph(std::string title, std::map<std::string, std::vector<std::pair<std::string, float> > > & data, std::vector<std::string> & groupOrder, BarGraphAxisType axisType, std::string axisLabel, std::string axisUnits, std::string groupLabel, osg::Vec4 color);

        float getWidth()
        {
            return _width;
        }

        float getHeight()
        {
            return _height;
        }

        void setDisplaySize(float width, float height);
        void setDisplayRange(float min, float max);

        float getDisplayRangeMin()
        {
            return _defaultMinDisplayRange;
        }

        float getDisplayRangeMax()
        {
            return _defaultMaxDisplayRange;
        }

        float getDataMax()
        {
            return _maxGraphValue;
        }

        float getDataMin()
        {
            return _minGraphValue;
        }

        void setColorMode(BarGraphColorMode bgcm);
        BarGraphColorMode getColorMode()
        {
            return _colorMode;
        }

        void setColorMapping(osg::Vec4 def, const std::map<std::string,osg::Vec4> & colorMap);

        void setColor(osg::Vec4 color);
        const osg::Vec4 & getColor()
        {
            return _color;
        }

        osg::Group * getRootNode()
        {
            return _root.get();
        }

        void setHover(osg::Vec3 intersect);
        void clearHoverText();

        const std::string & getHoverGroup()
        {
            return _hoverGroup;
        }

        const std::string & getHoverItem()
        {
            return _hoverItem;
        }

        void selectItems(std::string & group, std::vector<std::string> & keys);

        bool processClick(osg::Vec3 & hitPoint, std::string & selectedGroup, std::vector<std::string> & selectedKeys);

    protected:
        void makeGraph();
        void makeHover();
        void makeBG();
        void update();
        void updateGraph();
        void updateAxis();
        void updateShading();
        void updateColors();

        float _width, _height;
        float _maxGraphValue, _minGraphValue;
        float _defaultMaxDisplayRange, _defaultMinDisplayRange;
        float _topPaddingMult, _leftPaddingMult, _rightPaddingMult, _maxBottomPaddingMult, _currentBottomPaddingMult;
        float _titleMult, _topLabelMult, _groupLabelMult;
        int _numBars;
        float _minDisplayRange;
        float _maxDisplayRange;

        std::string _title, _axisLabel, _axisUnits, _groupLabel;
        BarGraphAxisType _axisType;
        std::map<std::string, std::vector<std::pair<std::string, float> > > _data;
        std::vector<std::string> _groupOrder;
        osg::Vec4 _color;

        osg::Vec4 _defaultGroupColor;
        std::map<std::string,osg::Vec4> _groupColorMap;

        osg::ref_ptr<osg::Group> _root;
        osg::ref_ptr<osg::MatrixTransform> _bgScaleMT;
        osg::ref_ptr<osg::Geode> _barGeode;
        osg::ref_ptr<osg::Geode> _axisGeode;
        osg::ref_ptr<osg::Geode> _bgGeode;
        osg::ref_ptr<osg::Geode> _selectGeode;
        osg::ref_ptr<osg::Geometry> _barGeom;
        osg::ref_ptr<osg::Geode> _shadingGeode;

        osg::ref_ptr<osg::Geode> _hoverGeode;
        osg::ref_ptr<osg::Geometry> _hoverBGGeom;
        osg::ref_ptr<osgText::Text> _hoverText;

        std::string _hoverGroup;
        std::string _hoverItem;

        BarGraphColorMode _colorMode;
};

#endif
