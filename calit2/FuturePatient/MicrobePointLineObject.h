#ifndef MICROBE_POINT_LINE_OBJECT_H
#define MICROBE_POINT_LINE_OBJECT_H

#include <string>

#include <osg/Geode>
#include <osg/Geometry>

#include <mysql++/mysql++.h>

#include "LayoutInterfaces.h"
#include "PointLineGraph.h"

class MicrobePointLineObject : public LayoutTypeObject, public PatientSelectObject
{
    public:
        MicrobePointLineObject(mysqlpp::Connection * conn, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~MicrobePointLineObject();

        bool setGraph(bool expandAxis = false);

        virtual void objectAdded();
        virtual void objectRemoved();

        virtual void setGraphSize(float width, float height);

        virtual void selectPatients(std::string & group, std::vector<std::string> & patients);

        virtual bool processEvent(cvr::InteractionEvent * ie);
        virtual void updateCallback(int handID, const osg::Matrix & mat);
        virtual void leaveCallback(int handID);

    protected:
        PointLineGraph * _graph;
        mysqlpp::Connection * _conn;
        bool _desktopMode;
};

#endif
