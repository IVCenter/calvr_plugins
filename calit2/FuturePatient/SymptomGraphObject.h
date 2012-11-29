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

        void setGraphDisplayRange(time_t start, time_t end);
        void resetGraphDisplayRange();
        void getGraphDisplayRange(time_t & start, time_t & end);
        time_t getMaxTimestamp();
        time_t getMinTimestamp();

    protected:
        mysqlpp::Connection * _conn;

        TimeRangeDataGraph * _graph;

        float _width;
        float _height;

};

#endif
