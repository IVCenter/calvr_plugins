#ifndef STRAIN_GRAPH_OBJECT_H
#define STRAIN_GRAPH_OBJECT_H

#include <string>
#include <vector>

#include "DBManager.h"

#include "GroupedBarGraph.h"
#include "LayoutInterfaces.h"

class StrainGraphObject: public LayoutTypeObject, public PatientSelectObject, public ValueRangeObject
{
    public:
        StrainGraphObject(DBManager * dbm, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~StrainGraphObject();

        bool setGraph(std::string title, int taxId, bool larryOnly = false);

        virtual void objectAdded();
        virtual void objectRemoved();

        void setGraphSize(float width, float height);
        void setColor(osg::Vec4 color);

        float getGraphMaxValue();
        float getGraphMinValue();

        float getGraphDisplayRangeMax();
        float getGraphDisplayRangeMin();

        void setGraphDisplayRange(float min, float max);
        void resetGraphDisplayRange();

        void selectPatients(std::string & group, std::vector<std::string> & patients);

        virtual bool processEvent(cvr::InteractionEvent * ie);
        virtual void updateCallback(int handID, const osg::Matrix & mat);
        virtual void leaveCallback(int handID);

    protected:
        DBManager * _dbm;

        GroupedBarGraph * _graph;

        bool _desktopMode;
};

#endif
