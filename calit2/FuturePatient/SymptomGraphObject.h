#ifndef FP_SYMPTOM_GRAPH_OBJECT
#define FP_SYMPTOM_GRAPH_OBJECT

#include <string>

#include <mysql++/mysql++.h>

#include "LayoutInterfaces.h"
#include "TimeRangeDataGraph.h"

class SymptomGraphObject : public LayoutTypeObject, public TimeRangeObject
{
    public:
        SymptomGraphObject(mysqlpp::Connection * conn, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~SymptomGraphObject();

        bool addGraph(std::string name);

        void setGraphSize(float width, float height);
        void setBarVisible(bool vis);
        float getBarPosition();
        void setBarPosition(float pos);
        bool getGraphSpacePoint(const osg::Matrix & mat, osg::Vec3 & point);
        void setGLScale(float scale);
        virtual void dumpState(std::ostream & out);
        virtual bool loadState(std::istream & in);

        void setGraphDisplayRange(time_t start, time_t end);
        void resetGraphDisplayRange();
        void getGraphDisplayRange(time_t & start, time_t & end);
        time_t getMaxTimestamp();
        time_t getMinTimestamp();

        virtual void updateCallback(int handID, const osg::Matrix & mat);
        virtual void leaveCallback(int handID);
        virtual bool eventCallback(cvr::InteractionEvent * ie);

    protected:
        bool addGraphMicrobe(std::string name);

        mysqlpp::Connection * _conn;

        TimeRangeDataGraph * _graph;

        std::map<int,std::string> _intensityLabels;

        float _width;
        float _height;

        bool _desktopMode;

        struct LoadData
        {
            std::string name;
        };

        std::vector<struct LoadData> _loadedGraphs;
};

#endif
