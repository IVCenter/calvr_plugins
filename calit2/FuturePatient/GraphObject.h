#ifndef FP_GRAPH_OBJECT_H
#define FP_GRAPH_OBJECT_H

#include <cvrMenu/MenuList.h>
#include <cvrMenu/MenuCheckbox.h>

#include <string>

#include "DBManager.h"

#include "DataGraph.h"
#include "LayoutInterfaces.h"

class LinearRegFunc;

class GraphObject : public LayoutTypeObject, public TimeRangeObject
{
    public:
        GraphObject(DBManager * dbm, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~GraphObject();

        bool addGraph(std::string patient, std::string name, bool averageColor = false);
        int getNumGraphs()
        {
            return _graph->getNumGraphs();
        }

        void setGraphSize(float width, float height);

        void setGraphDisplayRange(time_t start, time_t end);
        void resetGraphDisplayRange();
        void getGraphDisplayRange(time_t & start, time_t & end);
        time_t getMaxTimestamp();
        time_t getMinTimestamp();

        void setBarPosition(float pos);
        float getBarPosition();
        void setBarVisible(bool b);
        bool getBarVisible();
        bool getGraphSpacePoint(const osg::Matrix & mat, osg::Vec3 & point);

        virtual void dumpState(std::ostream & out);
        virtual bool loadState(std::istream & in);

        void setLayoutDoesDelete(bool b)
        {
            _layoutDoesDelete = b;
        }
        bool getLayoutDoesDelete()
        {
            return _layoutDoesDelete;
        }

        void setGLScale(float scale);

        void setLinearRegression(bool lr);

        void perFrame();

        virtual void menuCallback(cvr::MenuItem * item);

        virtual bool processEvent(cvr::InteractionEvent * ie);
        virtual void enterCallback(int handID, const osg::Matrix &mat);
        virtual void updateCallback(int handID, const osg::Matrix &mat);
        virtual void leaveCallback(int handID);

        int getNumMathFunctions();
        MathFunction * getMathFunction(int i);
    protected:
        DBManager * _dbm;
        std::vector<std::string> _nameList;
        DataGraph * _graph;

        cvr::MenuList * _mgdList;
        cvr::MenuList * _ldmList;

        std::string _pdfDir;

        int _activeHand;
        bool _layoutDoesDelete;

        struct LoadData
        {
            std::string patient;
            std::string name;
            std::string displayName;
        };

        AverageFunction * _averageFunc;
        cvr::MenuCheckbox * _averageCB;
        LinearRegFunc * _linRegFunc;
        cvr::MenuCheckbox * _linRegCB;

        std::vector<LoadData> _loadedGraphs;
};

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
