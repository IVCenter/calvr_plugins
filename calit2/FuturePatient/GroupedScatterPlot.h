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
#include <list>

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
        bool addGroup(int index, std::string indexLabel, std::vector<std::pair<float,float> > & data, std::vector<std::string> & dataLabels, float firstLogMinValue = 0.00001, float secondLogMinValue = 0.00001);

        void setDisplaySize(float width, float height);
        void setGLScale(float scale);

        float getFirstMax()
        {
            return _firstDataMax;
        }
        float getFirstMin()
        {
            return _firstDataMin;
        }
        float getSecondMax()
        {
            return _secondDataMax;
        }
        float getSecondMin()
        {
            return _secondDataMin;
        }
        float getFirstDisplayMax()
        {
            return _firstDisplayMax;
        }
        float getFirstDisplayMin()
        {
            return _firstDisplayMin;
        }
        float getSecondDisplayMax()
        {
            return _secondDisplayMax;
        }
        float getSecondDisplayMin()
        {
            return _secondDisplayMin;
        }        

        void setFirstDisplayRange(float min, float max);
        void setSecondDisplayRange(float min, float max);
        void resetDisplayRange();

        bool processClick(std::string & group, std::vector<std::string> & labels);
        void selectPoints(std::map<std::string,std::vector<std::string> > & selectMap);

        void setHover(osg::Vec3 intersect);
        void clearHoverText();

        void setColorMapping(osg::Vec4 def, const std::map<std::string,osg::Vec4> & colorMap);

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

        float _width;
        float _height;

        std::string _title;
        std::string _firstLabel;
        std::string _secondLabel;

        GSPAxisType _firstAxisType;
        GSPAxisType _secondAxisType;

        std::map<int,std::vector<std::pair<float,float> > > _plotData;
        std::map<int,std::string> _indexLabels;
        std::map<int,std::vector<std::string> > _pointLabels;
        std::list<std::pair<int,int> > _pointMapping;
        float _firstDataMax;
        float _firstDataMin;
        float _firstDisplayMax;
        float _firstDisplayMin;
        float _secondDataMax;
        float _secondDataMin;
        float _secondDisplayMax;
        float _secondDisplayMin;

        float _myFirstDisplayMin;
        float _myFirstDisplayMax;
        float _mySecondDisplayMin;
        float _mySecondDisplayMax;
        int _maxIndex;

        osg::ref_ptr<osg::Group> _root;
        osg::ref_ptr<osg::MatrixTransform> _bgScaleMT;
        osg::ref_ptr<osg::Geode> _bgGeode;
        osg::ref_ptr<osg::Geode> _axisGeode;
        osg::ref_ptr<osg::Geode> _dataGeode;
        osg::ref_ptr<osg::Geode> _hoverGeode;
        osg::ref_ptr<osg::Geometry> _hoverBGGeom;
        osg::ref_ptr<osgText::Text> _hoverText;

        int _currentHoverIndex, _currentHoverOffset;

        float _leftPaddingMult, _rightPaddingMult, _topPaddingMult, _bottomPaddingMult, _labelPaddingMult;
        float _graphLeft, _graphRight, _graphTop, _graphBottom;

        osg::ref_ptr<osg::Point> _point;

        float _glScale;
        float _pointLineScale;
        float _masterPointScale;

        std::map<std::string,std::vector<std::string> > _selectedMap;

        osg::Vec4 _defaultGroupColor;
        std::map<std::string,osg::Vec4> _groupColorMap;
};

#endif
