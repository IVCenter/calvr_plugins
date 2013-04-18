#ifndef GROUPED_SCATTER_PLOT_H
#define GROUPED_SCATTER_PLOT_H

#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Point>
#include <osgText/Text>

#include <string>
#include <vector>
#include <map>

enum GSPAxisType
{
    GSP_LINEAR=0,
    GSP_LOG
};

class GroupedScatterPlot
{
    public:
        GroupedScatterPlot(float width, float height);
        ~GroupedScatterPlot();

        void setLabels(std::string title, std::string firstLabel, std::string secondLabel);
        void setAxisTypes(GSPAxisType first, GSPAxisType second);
        bool addGroup(int index, std::vector<std::pair<float,float> > & data);

        void setDisplaySize(float width, float height);
        void setGLScale(float scale);

        osg::Group * getRootNode()
        {
            return _root.get();
        }

    protected:
        void makeBG();
        void update();
        void updateAxis();
        void updateGraph();
        void updateSizes();

        osgText::Text * makeText(std::string text, osg::Vec4 color);

        float _width;
        float _height;

        std::string _title;
        std::string _firstLabel;
        std::string _secondLabel;

        GSPAxisType _firstAxisType;
        GSPAxisType _secondAxisType;

        std::map<int,std::vector<std::pair<float,float> > > _plotData;
        float _firstDataMax;
        float _firstDataMin;
        float _firstDisplayMax;
        float _firstDisplayMin;
        float _secondDataMax;
        float _secondDataMin;
        float _secondDisplayMax;
        float _secondDisplayMin;
        int _maxIndex;

        osg::ref_ptr<osg::Group> _root;
        osg::ref_ptr<osg::MatrixTransform> _bgScaleMT;
        osg::ref_ptr<osg::Geode> _bgGeode;
        osg::ref_ptr<osg::Geode> _pointsGeode;
        osg::ref_ptr<osg::Geometry> _pointsGeom;
        osg::ref_ptr<osg::Geode> _axisGeode;
        osg::ref_ptr<osg::Geode> _dataGeode;

        float _leftPaddingMult, _rightPaddingMult, _topPaddingMult, _bottomPaddingMult, _labelPaddingMult;
        float _graphLeft, _graphRight, _graphTop, _graphBottom;

        osg::ref_ptr<osgText::Font> _font;
        osg::ref_ptr<osg::Point> _point;

        float _glScale;
        float _pointLineScale;
        float _masterPointScale;
};

#endif
