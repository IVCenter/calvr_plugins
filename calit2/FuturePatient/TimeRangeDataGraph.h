#ifndef FP_TIME_RANGE_DATA_GRAPH_H
#define FP_TIME_RANGE_DATA_GRAPH_H

#include <osg/Group>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/LineWidth>
#include <osgText/Text>

#include <string>
#include <vector>

class TimeRangeDataGraph
{
    public:
        TimeRangeDataGraph();
        virtual ~TimeRangeDataGraph();

        void setDisplaySize(float width, float height);
        bool getBarVisible();
        void setBarVisible(bool vis);
        float getBarPosition();
        void setBarPosition(float pos);
        bool getGraphSpacePoint(const osg::Matrix & mat, osg::Vec3 & point);
        void setGLScale(float scale);
        void setColorOffset(int offset);

        void addGraph(std::string name, std::vector<std::pair<time_t,time_t> > & rangeList, std::vector<int> & valueList, int maxValue);

        void setValueLabelMap(std::map<int,std::string> & labelMap)
        {
            _labelMap = labelMap;
        }

        void setDisplayRange(time_t & start, time_t & end);
        void getDisplayRange(time_t & start, time_t & end);
        time_t getMaxTimestamp();
        time_t getMinTimestamp();

        osg::Group * getGraphRoot();

        void setHover(osg::Vec3 intersect);
        void clearHoverText();

    protected:
        struct RangeDataInfo
        {
            std::string name;
            std::vector<std::pair<time_t,time_t> > ranges;
            std::vector<int> values;
            int maxValue;
            osg::ref_ptr<osg::Geometry> barGeometry;
            osg::ref_ptr<osg::Geometry> barOutlineGeometry;
        };

        float calcPadding();

        void initGeometry(RangeDataInfo * rdi);
        void makeBG();
        void makeHover();
        void makeBar();
        void update();
        void updateGraphs();
        void updateAxis();
        void updateShading();

        void updateSizes();
        osgText::Text * makeText(std::string text, osg::Vec4 color);

        float _graphLeft, _graphRight, _graphTop, _graphBottom;
        float _barHeight, _barPadding;

        std::vector<RangeDataInfo *> _graphList;
        std::map<int,std::string> _labelMap;

        time_t _displayMin;
        time_t _displayMax;

        time_t _timeMin;
        time_t _timeMax;

        float _width;
        float _height;

        int _colorOffset;

        osg::ref_ptr<osg::Group> _root;
        osg::ref_ptr<osg::Geode> _axisGeode;
        osg::ref_ptr<osg::Geode> _bgGeode;
        osg::ref_ptr<osg::MatrixTransform> _bgScaleMT;
        osg::ref_ptr<osg::Geode> _graphGeode;
        osg::ref_ptr<osg::Geode> _shadingGeode;

        osg::ref_ptr<osg::Geode> _hoverGeode;
        osg::ref_ptr<osg::Geometry> _hoverBGGeom;
        osg::ref_ptr<osgText::Text> _hoverText;
        int _currentHoverIndex;
        int _currentHoverGraph;

        osg::ref_ptr<osg::MatrixTransform> _barTransform;
        osg::ref_ptr<osg::MatrixTransform> _barPosTransform;
        osg::ref_ptr<osg::Geode> _barGeode;
        osg::ref_ptr<osg::Geometry> _barGeometry;
        float _barPos;
        osg::ref_ptr<osg::LineWidth> _barLineWidth;
        float _pointLineScale;
        float _glScale;
        float _masterLineScale;

        osg::ref_ptr<osgText::Font> _font;
};

#endif
