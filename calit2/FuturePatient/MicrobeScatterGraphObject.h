#ifndef MICROBE_SCATTER_GRAPH_OBJECT_H
#define MICROBE_SCATTER_GRAPH_OBJECT_H

#include <string>

#include <mysql++/mysql++.h>

#include "LayoutInterfaces.h"
#include "GroupedScatterPlot.h"

class MicrobeScatterGraphObject : public LayoutTypeObject, public LogValueRangeObject, public PatientSelectObject
{
    public:
        MicrobeScatterGraphObject(mysqlpp::Connection * conn, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~MicrobeScatterGraphObject();

        bool setGraph(std::string title, std::string primaryPhylum, std::string secondaryPhylum);

        virtual void setGraphSize(float width, float height);

        virtual void selectPatients(std::vector<std::string> & patients);

        virtual float getGraphXMaxValue();
        virtual float getGraphXMinValue();
        virtual float getGraphZMaxValue();
        virtual float getGraphZMinValue();

        virtual float getGraphXDisplayRangeMax();
        virtual float getGraphXDisplayRangeMin();
        virtual float getGraphZDisplayRangeMax();
        virtual float getGraphZDisplayRangeMin();

        virtual void setGraphXDisplayRange(float min, float max);
        virtual void setGraphZDisplayRange(float min, float max);
        virtual void resetGraphDisplayRange();

        virtual bool processEvent(cvr::InteractionEvent * ie);
        virtual void updateCallback(int handID, const osg::Matrix & mat);
        virtual void leaveCallback(int handID);

    protected:
        mysqlpp::Connection * _conn;
        GroupedScatterPlot * _graph;

        bool _desktopMode;
};

#endif
