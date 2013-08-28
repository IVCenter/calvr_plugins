#ifndef POINT_LINE_GRAPH_H
#define POINT_LINE_GRAPH_H

#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Point>
#include <osg/LineWidth>
#include <osgText/Text>

#include <string>
#include <vector>

enum PLGAxisType
{
    PLG_LINEAR=0,
    PLG_LOG
};

class PointLineGraph
{
    public:
        PointLineGraph(float width, float height);
        ~PointLineGraph();

        bool setGraph(std::string title, std::vector<std::string> & groupNames, std::vector<std::string> & catNames, std::vector<std::vector<std::string> > & dataNames, std::vector<std::vector<std::vector<float> > > & data);

        void setDisplaySize(float width, float height);

        void setAxisType(PLGAxisType type);
        PLGAxisType getAxisType()
        {
            return _axisType;
        }

        void setColorMapping(const std::map<std::string,osg::Vec4> & colorMap);

        bool processClick(osg::Vec3 point, std::string & group, std::vector<std::string> & labels);
        void selectItems(std::string & group, std::vector<std::string> & labels);

        void setHover(osg::Vec3 intersect);
        void clearHoverText();

        osg::Group * getRootNode()
        {
            return _root.get();
        }

    protected:
        void makeBG();
        void makeHover();
        void update();
        void updateAxis();
        void updateGraph();
        void updateSizes();
        void updateColors();

        float _width;
        float _height;

        std::string _title;
        std::vector<std::string> _groupLabels;
        std::vector<std::string> _catLabels;
        std::vector<std::vector<std::string> > _dataLabels;
        std::vector<std::vector<std::vector<float> > > _data;
        std::map<std::string,osg::Vec4> _colorMap;
        PLGAxisType _axisType;
        float _leftPaddingMult, _rightPaddingMult, _topPaddingMult, _bottomPaddingMult, _labelPaddingMult;
        float _titlePaddingMult, _catLabelPaddingMult;
        float _graphLeft, _graphRight, _graphTop, _graphBottom;

        float _minLin, _maxLin, _minLog, _maxLog;
        float _minDispLin, _minDispLog, _maxDispLin, _maxDispLog;

        osg::ref_ptr<osg::Group> _root;
        osg::ref_ptr<osg::MatrixTransform> _bgScaleMT;
        osg::ref_ptr<osg::Geode> _bgGeode;
        osg::ref_ptr<osg::Geode> _axisGeode;
        osg::ref_ptr<osg::Geode> _dataGeode;
        osg::ref_ptr<osg::Geode> _hoverGeode;
        osg::ref_ptr<osg::Geometry> _hoverBGGeom;
        osg::ref_ptr<osgText::Text> _hoverText;
        osg::ref_ptr<osg::Geometry> _dataGeometry;
        osg::ref_ptr<osg::DrawElementsUInt> _pointElements;
        osg::ref_ptr<osg::DrawElementsUInt> _lineElements;
        osg::ref_ptr<osg::Point> _point;
        osg::ref_ptr<osg::LineWidth> _line;

        int _currentHoverGroup, _currentHoverItem, _currentHoverPoint;
        std::string _selectedGroup;
        std::vector<std::string> _selectedLabels;
};

#endif
