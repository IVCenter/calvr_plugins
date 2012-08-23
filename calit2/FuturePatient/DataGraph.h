#ifndef FP_DATA_GRAPH_H
#define FP_DATA_GRAPH_H

#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/ClipNode>
#include <osg/Point>
#include <osg/LineWidth>
#include <osgText/Text>

#include <string>
#include <vector>
#include <map>
#include <ctime>

enum AxisType
{
    LINEAR = 1,
    TIMESTAMP
};

enum GraphDisplayType
{
    POINTS = 1,
    POINTS_WITH_LINES
};

struct GraphDataInfo
{
    std::string name;
    osg::ref_ptr<osg::Vec3Array> data;
    osg::ref_ptr<osg::Vec4Array> colorArray;
    osg::ref_ptr<osg::Vec4Array> secondaryColorArray;
    osg::Vec4 color;
    GraphDisplayType displayType;
    std::string xLabel;
    std::string zLabel;
    AxisType xAxisType;
    AxisType zAxisType;
    float xMin;
    float xMax;
    time_t xMinT;
    time_t xMaxT;
    float zMin;
    float zMax;
};

class DataGraph
{
    public:
        DataGraph();
        virtual ~DataGraph();

        void setDisplaySize(float width, float height);
        void addGraph(std::string name, osg::Vec3Array * points, GraphDisplayType displayType, std::string xLabel, std::string zLabel, osg::Vec4 color, osg::Vec4Array * perPointColor = NULL, osg::Vec4Array * secondaryPerPointColor = NULL);
        void setXDataRangeTimestamp(std::string graphName, time_t & start, time_t & end);
        void setZDataRange(std::string graphName, float min, float max);
        void setXDisplayRange(float min, float max);
        void setXDisplayRangeTimestamp(time_t & start, time_t & end);
        void setZDisplayRange(float min, float max);

        osg::MatrixTransform * getGraphRoot();

        void getGraphNameList(std::vector<std::string> & nameList);

        void getXDisplayRange(float & min, float & max)
        {
            min = _minDisplayX;
            max = _maxDisplayX;
        }
        void getXDisplayRangeTimestamp(time_t & start, time_t & end)
        {
            start = _minDisplayXT;
            end = _maxDisplayXT;
        }

        float getDisplayWidth()
        {
            return _width;
        }
        float getDisplayHeight()
        {
            return _height;
        }
        time_t getMaxTimestamp(std::string graphName);
        time_t getMinTimestamp(std::string graphName);

        bool displayHoverText(osg::Matrix & mat);
        void clearHoverText();

        void setBarPosition(float pos);
        float getBarPosition();
        void setBarVisible(bool b);
        bool getBarVisible();

        bool getGraphSpacePoint(const osg::Matrix & mat, osg::Vec3 & point);
        int getNumGraphs()
        {
            return _dataInfoMap.size();
        }

    protected:
        void makeHover();
        void makeBar();
        void update();
        void updateAxis();
        void updateBar();
        void updateClip();
        float calcPadding();

        osg::Vec3 makeColor(float f);

        osgText::Text * makeText(std::string text, osg::Vec4 color);

        std::map<std::string, osg::ref_ptr<osg::MatrixTransform> > _graphTransformMap;
        std::map<std::string, osg::ref_ptr<osg::Geometry> > _graphGeometryMap;
        std::map<std::string, GraphDataInfo> _dataInfoMap;
        osg::ref_ptr<osg::Geometry> _axisGeometry;
        osg::ref_ptr<osg::Geometry> _bgGeometry;
        osg::ref_ptr<osg::Geode> _graphGeode;
        osg::ref_ptr<osg::Geode> _axisGeode;
        osg::ref_ptr<osg::MatrixTransform> _graphTransform;
        osg::ref_ptr<osg::ClipNode> _clipNode;
        osg::ref_ptr<osg::MatrixTransform> _root;

        osg::ref_ptr<osg::MatrixTransform> _hoverTransform;
        osg::ref_ptr<osg::MatrixTransform> _hoverBGScale;
        osg::ref_ptr<osg::Geode> _hoverBGGeode;
        osg::ref_ptr<osg::Geode> _hoverTextGeode;
        osg::ref_ptr<osgText::Text> _hoverText;

        std::string _hoverGraph;
        int _hoverPoint;

        osg::ref_ptr<osg::MatrixTransform> _barTransform;
        osg::ref_ptr<osg::MatrixTransform> _barPosTransform;
        osg::ref_ptr<osg::Geode> _barGeode;
        osg::ref_ptr<osg::Geometry> _barGeometry;

        bool _xAxisTimestamp;
        float _width;
        float _height;
        float _minDisplayX;
        float _maxDisplayX;
        time_t _minDisplayXT;
        time_t _maxDisplayXT;
        float _minDisplayZ;
        float _maxDisplayZ;

        osg::ref_ptr<osg::Point> _point;
        osg::ref_ptr<osg::LineWidth> _lineWidth;
        float _pointLineScale;
        osg::ref_ptr<osgText::Font> _font;
};

#endif
