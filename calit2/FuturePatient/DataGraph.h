#ifndef FP_DATA_GRAPH_H
#define FP_DATA_GRAPH_H

#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/ClipNode>
#include <osg/Point>
#include <osg/LineWidth>
#include <osg/LineStipple>
#include <osg/Depth>
#include <osg/PointSprite>
#include <osg/Program>
#include <osg/Uniform>
#include <osgText/Text>

#include <string>
#include <vector>
#include <map>
#include <ctime>

#include "PointActions.h"
#include "GraphGlobals.h"

enum AxisType
{
    LINEAR = 1,
    TIMESTAMP
};

enum GraphDisplayType
{
    GDT_NONE = 0,
    GDT_POINTS,
    GDT_POINTS_WITH_LINES
};

enum MultiGraphDisplayMode
{
    MGDM_NORMAL=0,
    MGDM_COLOR,
    MGDM_COLOR_SOLID,
    MGDM_COLOR_PT_SIZE,
    MGDM_SHAPE,
    MGDM_COLOR_SHAPE
};

enum LabelDisplayMode
{
    LDM_NONE=0,
    LDM_MIN_MAX,
    LDM_ALL
};

struct GraphDataInfo
{
    std::string name;
    osg::ref_ptr<osg::Vec3Array> data;
    osg::ref_ptr<osg::Vec4Array> colorArray;
    osg::ref_ptr<osg::Vec4Array> secondaryColorArray;
    osg::ref_ptr<osg::Vec4Array> singleColorArray;
    osg::ref_ptr<osg::Geode> pointGeode;
    osg::ref_ptr<osg::Geometry> pointGeometry;
    osg::ref_ptr<osg::Geode> connectorGeode;
    osg::ref_ptr<osg::Geometry> connectorGeometry;
    osg::ref_ptr<osg::Geode> labelGeode;
    osg::ref_ptr<SetBoundsCallback> boundsCallback;
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
    osg::ref_ptr<osg::Geode> pointActionGeode;
    osg::ref_ptr<osg::Geometry> pointActionGeometry;
};

class MathFunction;

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
        void getZDisplayRange(float & min, float & max)
        {
            min = _minDisplayZ;
            max = _maxDisplayZ;
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

        void setBGRanges(std::vector<std::pair<float,float> > & ranges, std::vector<osg::Vec4> & colors);

        int getNumGraphs()
        {
            return _dataInfoMap.size();
        }

        void setDisplayType(std::string graphName, GraphDisplayType displayType);
        void setDisplayTypes(GraphDisplayType displayType);
        GraphDisplayType getDisplayType(std::string graphName);

        void setMultiGraphDisplayMode(MultiGraphDisplayMode mgdm)
        {
            _multiGraphDisplayMode = mgdm;
            update();
        }

        MultiGraphDisplayMode getMultiGraphDisplayMode()
        {
            return _multiGraphDisplayMode;
        }

        void setLabelDisplayMode(LabelDisplayMode ldm);
        LabelDisplayMode getLabelDisplayMode()
        {
            return _labelDisplayMode;
        }

        void setGLScale(float scale);

        void setPointActions(std::string graphname, std::map<int,PointAction*> & actionMap);
        void updatePointAction();
        bool pointClick();

        void addMathFunction(MathFunction * mf);
        void removeMathFunction(MathFunction * mf);
        int getNumMathFunctions();
        MathFunction * getMathFunction(int i);

    protected:
        void setupMultiGraphDisplayModes();
        void makeHover();
        void makeBar();
        void update();
        void updateAxis();
        void updateBar();
        void updateClip();
        void updateBGRanges();
        float calcPadding();

        std::map<std::string, osg::ref_ptr<osg::MatrixTransform> > _graphTransformMap;
        //std::map<std::string, osg::ref_ptr<osg::Geometry> > _graphGeometryMap;
        std::map<std::string, GraphDataInfo> _dataInfoMap;
        osg::ref_ptr<osg::Geometry> _axisGeometry;
        osg::ref_ptr<osg::Geometry> _bgGeometry;
        osg::ref_ptr<osg::Geode> _graphGeode;
        osg::ref_ptr<osg::Geode> _axisGeode;
        osg::ref_ptr<osg::MatrixTransform> _graphTransform;
        osg::ref_ptr<osg::ClipNode> _clipNode;
        osg::ref_ptr<osg::MatrixTransform> _root;
        osg::ref_ptr<osg::Group> _labelGroup;
        osg::ref_ptr<osg::Geode> _mathGeode;

        osg::ref_ptr<osg::MatrixTransform> _hoverTransform;
        osg::ref_ptr<osg::MatrixTransform> _hoverBGScale;
        osg::ref_ptr<osg::Geode> _hoverBGGeode;
        osg::ref_ptr<osg::Geode> _hoverTextGeode;
        osg::ref_ptr<osgText::Text> _hoverText;

        osg::ref_ptr<osg::Geode> _bgRangesGeode;
        
        std::vector<std::pair<float,float> > _bgRanges;
        std::vector<osg::Vec4> _bgRangesColors;

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

        float _masterPointScale;
        float _masterLineScale;

        float _glScale;

        osg::ref_ptr<osg::Point> _point;
        osg::ref_ptr<osg::LineWidth> _lineWidth;
        float _pointLineScale;

        MultiGraphDisplayMode _multiGraphDisplayMode;
        MultiGraphDisplayMode _currentMultiGraphDisplayMode;

        osg::ref_ptr<osg::PointSprite> _shapePointSprite;
        osg::ref_ptr<osg::Depth> _shapeDepth;
        osg::ref_ptr<osg::Program> _shapeProgram;

        osg::ref_ptr<osg::Program> _sizeProgram;
        osg::ref_ptr<osg::Uniform> _pointSizeUniform;

        std::map<std::string,std::map<int,PointAction*> > _pointActionMap;
        osg::ref_ptr<osg::Point> _pointActionPoint;
        float _pointActionAlpha;
        bool _pointActionAlphaDir;

        LabelDisplayMode _labelDisplayMode;

        std::vector<MathFunction *> _mathFunctions;
};

class MathFunction
{
    public:
        virtual void added(osg::Geode * geode) = 0;
        virtual void removed(osg::Geode * geode) = 0;
        virtual void update(float width, float height, std::map<std::string, GraphDataInfo> & data, std::map<std::string, std::pair<float,float> > & displayRanges, std::map<std::string,std::pair<int,int> > & dataPointRanges) = 0;
};

class AverageFunction : public MathFunction
{
    public:
        AverageFunction();
        virtual ~AverageFunction();

        void added(osg::Geode * geode);
        void removed(osg::Geode * geode);
        void update(float width, float height, std::map<std::string, GraphDataInfo> & data, std::map<std::string, std::pair<float,float> > & displayRanges, std::map<std::string,std::pair<int,int> > & dataPointRanges);

    protected:
        osg::ref_ptr<osg::Geometry> _averageGeometry;
        osg::ref_ptr<osgText::Text> _averageText;
        osg::ref_ptr<osg::LineStipple> _averageStipple;
        osg::ref_ptr<osg::LineWidth> _averageLineWidth;
        osg::ref_ptr<SetBoundsCallback> _boundsCallback;
};

#endif
