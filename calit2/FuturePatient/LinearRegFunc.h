#ifndef FP_LINEAR_REG_FUNC
#define FP_LINEAR_REG_FUNC

#include <string>
#include <map>

#include "DataGraph.h"

class LinearRegFunc : public MathFunction
{
    public:
        LinearRegFunc();
        virtual ~LinearRegFunc();

        void added(osg::Geode * geode);
        void removed(osg::Geode * geode);
        void update(float width, float height, std::map<std::string, GraphDataInfo> & data, std::map<std::string, std::pair<float,float> > & displayRanges, std::map<std::string,std::pair<int,int> > & dataPointRanges);

        void setDataRange(std::string name, float min, float max);
        void setTimeRange(std::string name, time_t min, time_t max);
        void setHealthyRange(std::string name, float min, float max);

        time_t getHealthyIntersectTime()
        {
            return _healthyIntersectTime;
        }

    protected:
        osg::ref_ptr<osg::Geometry> _lrGeometry;
        osg::ref_ptr<osg::LineWidth> _lrLineWidth;
        osg::ref_ptr<SetBoundsCallback> _lrBoundsCallback;

        std::map<std::string,std::pair<float,float> > _dataRangeMap;
        std::map<std::string,std::pair<time_t,time_t> > _timeRangeMap;
        std::map<std::string,std::pair<float,float> > _healthyRangeMap;

        time_t _healthyIntersectTime;
};

#endif
