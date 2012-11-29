#ifndef FP_TIME_RANGE_DATA_GRAPH_H
#define FP_TIME_RANGE_DATA_GRAPH_H

#include <osg/Group>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/MatrixTransform>

#include <string>
#include <vector>
#include <map>

class TimeRangeDataGraph
{
    public:
        TimeRangeDataGraph();
        virtual ~TimeRangeDataGraph();

        void setDisplaySize(float width, float height);

        void addGraph(std::string name, std::vector<std::pair<time_t,time_t> > & rangeList);

        void setDisplayRange(time_t & start, time_t & end);
        void getDisplayRange(time_t & start, time_t & end);
        time_t getMaxTimestamp();
        time_t getMinTimestamp();

        osg::Group * getGraphRoot();

    protected:
        struct RangeDataInfo
        {
            std::string name;
            std::vector<std::pair<time_t,time_t> > ranges;
            osg::ref_ptr<osg::Geometry> barGeometry;
        };

        void makeBG();
        void update();

        std::map<std::string,RangeDataInfo *> _graphMap;

        time_t _displayMin;
        time_t _displayMax;

        time_t _timeMin;
        time_t _timeMax;

        float _width;
        float _height;

        osg::ref_ptr<osg::Group> _root;
        osg::ref_ptr<osg::Geode> _axisGeode;
        osg::ref_ptr<osg::Geode> _bgGeode;
        osg::ref_ptr<osg::MatrixTransform> _bgScaleMT;
        osg::ref_ptr<osg::Geode> _graphGeode;
};

#endif
