#ifndef VERTICAL_STACKED_BAR_GRAPH_H
#define VERTICAL_STACKED_BAR_GRAPH_H

#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/LineWidth>
#include <osgText/Text>

#include <string>
#include <vector>

class VerticalStackedBarGraph
{
    public:
        VerticalStackedBarGraph(std::string title);
        virtual ~VerticalStackedBarGraph();

        void setDataLabels(std::vector<std::string> & dataLabels);
        void setDataGroups(std::vector<std::string> & groupList);

        bool addBar(std::string label, std::vector<float> & values, std::vector<int> & groupIndexList);

        int getNumBars();
        std::string getBarLabel(int bar);
        float getValue(std::string group, std::string key, int bar);
        float getGroupValue(std::string group, int bar);

        void setDisplaySize(float width, float height);
        void setHover(osg::Vec3 intersect);
        void clearHoverText();

        void selectItems(std::string & group, std::vector<std::string> & keys);
        bool processClick(osg::Vec3 & intersect, std::string & selectedGroup, std::vector<std::string> & selectedKeys, bool & selectValid);

        osg::Group * getRoot()
        {
            return _root;
        }
    protected:
        void makeBG();
        void makeHover();
        osg::Geometry * makeGeometry(int elements);
        void update();
        void updateAxis();
        void updateGraph();
        void updateValues();

        std::string _title;
        float _width;
        float _height;

        std::vector<std::vector<float> > _dataList;
        std::vector<std::vector<int> > _dataGroupIndexLists;
        std::vector<std::string> _barLabels;

        std::vector<std::string> _dataLabels;
        std::vector<std::string> _dataGroups;

        osg::ref_ptr<osg::Group> _root;
        osg::ref_ptr<osg::MatrixTransform> _bgScaleMT;
        osg::ref_ptr<osg::Geode> _bgGeode;
        osg::ref_ptr<osg::Geode> _axisGeode;
        osg::ref_ptr<osg::Geode> _graphGeode;
        
        std::vector<osg::ref_ptr<osg::Geometry> > _geometryList;
        std::vector<osg::ref_ptr<osg::Geometry> > _connectionGeometryList;

        osg::ref_ptr<osg::Geometry> _lineGeometry;
        osg::ref_ptr<osg::DrawArrays> _linePrimitive;

        osg::ref_ptr<osg::Geode> _hoverGeode;
        osg::ref_ptr<osg::Geometry> _hoverBGGeom;
        osg::ref_ptr<osgText::Text> _hoverText;
        std::string _currentHoverValue;

        osg::ref_ptr<osg::LineWidth> _lineWidth;

        float _leftPaddingMult, _rightPaddingMult, _topPaddingMult, _bottomPaddingMult;
        float _barToConnectorRatio;
        float _graphTop, _graphBottom, _graphLeft, _graphRight;
        float _barWidth, _connectorWidth;

        std::string _lastSelectGroup;
        std::vector<std::string> _lastSelectKeys;

        
};

#endif
