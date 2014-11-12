#ifndef MICROBE_SCATTER_GRAPH_OBJECT_H
#define MICROBE_SCATTER_GRAPH_OBJECT_H

#include <string>

#include <osg/Geode>
#include <osg/Geometry>

#include "DBManager.h"

#include "LayoutInterfaces.h"
#include "GroupedScatterPlot.h"
#include "GraphKeyObject.h"
#include "MicrobeGraphObject.h"

class MicrobeScatterGraphObject : public LayoutTypeObject, public LogValueRangeObject, public PatientSelectObject, public SelectableObject
{
    public:
        MicrobeScatterGraphObject(DBManager * dbm, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~MicrobeScatterGraphObject();

        bool setGraph(std::string title, std::string primaryPhylum, std::string secondaryPhylum, MicrobeGraphType type, std::string microbeTableSuffix, std::string measureTableSuffix);

        virtual void objectAdded();
        virtual void objectRemoved();

        virtual void setGraphSize(float width, float height);

        virtual void selectPatients(std::map<std::string,std::vector<std::string> > & selectMap);

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

        void setLogScale(bool logScale);

    protected:
        void initData();
        void makeSelect();
        void updateSelect();

        DBManager * _dbm;
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

        osg::ref_ptr<osg::Geode> _selectGeode;
        osg::ref_ptr<osg::Geometry> _selectGeom;
};

#endif
