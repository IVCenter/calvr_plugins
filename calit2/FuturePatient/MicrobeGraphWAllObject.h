#ifndef FP_MICROBE_GRAPH_WALL_OBJECT_H
#define FP_MICROBE_GRAPH_WALL_OBJECT_H

#include <cvrMenu/MenuText.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuList.h>

#include <string>
#include <map>

#include "DBManager.h"

#include "GroupedBarGraphWPoints.h"
#include "LayoutInterfaces.h"
#include "MicrobeTypes.h"

class MicrobeGraphWAllObject : public LayoutTypeObject, public MicrobeSelectObject, public PatientSelectObject, public ValueRangeObject, public SelectableObject
{
    public:
        MicrobeGraphWAllObject(DBManager * dbm, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~MicrobeGraphWAllObject();

        bool setGraph(std::string title, int patientid, std::string testLabel, time_t testTime, int microbes, std::string microbeTableSuffix, std::string measureTableSuffix, bool group = true, bool lsOrdering = true, MicrobeGraphType type = MGT_SPECIES);
        bool setSpecialGraph(SpecialMicrobeGraphType smgt, int microbes, std::string region, std::string microbeSuffix, std::string measureSuffix, bool group = true, bool lsOrdering = true, MicrobeGraphType type = MGT_SPECIES);

        virtual void objectAdded();
        virtual void objectRemoved();

        void setGraphSize(float width, float height);
        void setColor(osg::Vec4 color);
        void setBGColor(osg::Vec4 color);

        float getGraphMaxValue();
        float getGraphMinValue();

        float getGraphDisplayRangeMax();
        float getGraphDisplayRangeMin();

        void setGraphDisplayRange(float min, float max);
        void resetGraphDisplayRange();

        void selectPatients(std::map<std::string,std::vector<std::string> > & selectMap);

        void selectMicrobes(std::string & group, std::vector<std::string> & keys);
        float getGroupValue(std::string group, int i);
        float getMicrobeValue(std::string group, std::string key, int i);
        int getNumDisplayValues();
        std::string getDisplayLabel(int i);

        virtual void dumpState(std::ostream & out);
        virtual bool loadState(std::istream & in);

        virtual std::string getTitle();

        virtual bool processEvent(cvr::InteractionEvent * ie);
        virtual void updateCallback(int handID, const osg::Matrix & mat);
        virtual void leaveCallback(int handID);

        virtual void menuCallback(cvr::MenuItem * item);

    protected:
        bool loadGraphData(std::string valueQuery, std::string orderQuery, bool group, bool lsOrdering, MicrobeGraphType familyLevel);
        void makeSelect();
        void updateSelect();

        static void makePatientMap();

        static bool _mapInit;
        static std::map<int,std::pair<std::string,std::string> > _patientMap;

        static DBManager * _dbm;
        
        std::string _graphTitle;
        std::map<std::string, std::vector<std::pair<std::string, float> > > _graphData;
        std::vector<std::string> _graphOrder;

        cvr::MenuList * _colorModeML;
        cvr::MenuText * _microbeText;
        cvr::MenuButton * _searchButton;
        std::string _menuMicrobe;

        float _width, _height;

        GroupedBarGraphWPoints * _graph;

        bool _desktopMode;

        // used to dump state
        bool _specialGraph;
        SpecialMicrobeGraphType _specialType;
        int _patientid;
        std::string _testLabel;
        int _microbes;
        bool _lsOrdered;

        osg::ref_ptr<osg::Geode> _selectGeode;
        osg::ref_ptr<osg::Geometry> _selectGeom;
};

#endif
