#ifndef MICROBE_POINT_LINE_OBJECT_H
#define MICROBE_POINT_LINE_OBJECT_H

#include <string>

#include <osg/Geode>
#include <osg/Geometry>

#include "DBManager.h"

#include "LayoutInterfaces.h"
#include "PointLineGraph.h"

class MicrobePointLineObject : public LayoutTypeObject, public PatientSelectObject
{
    public:
        MicrobePointLineObject(DBManager * dbm, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~MicrobePointLineObject();

        bool setGraph(std::string microbeTableSuffix, std::string measureTableSuffix, bool expandAxis = false);

        virtual void objectAdded();
        virtual void objectRemoved();

        virtual void setGraphSize(float width, float height);

        virtual void selectPatients(std::string & group, std::vector<std::string> & patients);

        virtual bool processEvent(cvr::InteractionEvent * ie);
        virtual void updateCallback(int handID, const osg::Matrix & mat);
        virtual void leaveCallback(int handID);

    protected:
        PointLineGraph * _graph;
        DBManager * _dbm;
        bool _desktopMode;
};

#endif
