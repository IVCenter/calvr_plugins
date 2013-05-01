#ifndef MICROBE_SCATTER_GRAPH_OBJECT_H
#define MICROBE_SCATTER_GRAPH_OBJECT_H

#include <string>

#include <osg/Geode>
#include <osg/Geometry>

#include <mysql++/mysql++.h>

#include "LayoutInterfaces.h"
#include "GroupedScatterPlot.h"
#include "GraphKeyObject.h"

class MicrobeScatterGraphObject : public LayoutTypeObject, public LogValueRangeObject, public PatientSelectObject, public SelectableObject
{
    public:
        MicrobeScatterGraphObject(mysqlpp::Connection * conn, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~MicrobeScatterGraphObject();

        bool setGraph(std::string title, std::string primaryPhylum, std::string secondaryPhylum);

        virtual void objectAdded();
        virtual void objectRemoved();

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

        virtual void menuCallback(cvr::MenuItem * item);

    protected:
        void initData();
        void makeSelect();
        void updateSelect();

        void makeGraphKey();

        mysqlpp::Connection * _conn;
        GroupedScatterPlot * _graph;

        bool _desktopMode;

        static bool _dataInit;
        struct DataEntry
        {
            std::string name;
            time_t timestamp;
            float value;
        };
        static std::vector<std::vector<struct DataEntry> > _data;
        static std::map<std::string,int> _phylumIndexMap;

        static GraphKeyObject * _graphKey;

        osg::ref_ptr<osg::Geode> _selectGeode;
        osg::ref_ptr<osg::Geometry> _selectGeom;
};

#endif
