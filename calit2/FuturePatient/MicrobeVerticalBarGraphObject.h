#ifndef MICROBE_VERTICAL_BAR_GRAPH_OBJECT_H
#define MICROBE_VERTICAL_BAR_GRAPH_OBJECT_H

#include <string>
#include <vector>

#include "DBManager.h"

#include "VerticalStackedBarGraph.h"
#include "MicrobeGraphObject.h"
#include "LayoutInterfaces.h"

class MicrobeVerticalBarGraphObject : public LayoutTypeObject, public MicrobeSelectObject, public SelectableObject
{
    public:
        MicrobeVerticalBarGraphObject(DBManager * dbm, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~MicrobeVerticalBarGraphObject();

        bool addGraph(std::string label, int patientid, std::string testLabel, time_t testTime, std::string seqType, std::string microbeTableSuffix, std::string measureTableSuffix, MicrobeGraphType type = MGT_SPECIES);
        bool addSpecialGraph(SpecialMicrobeGraphType smgt, std::string seqType, std::string microbeTableSuffix, std::string measureTableSuffix, std::string region, MicrobeGraphType type = MGT_SPECIES);

        virtual void objectAdded();
        virtual void objectRemoved();

        void setGraphSize(float width, float height);

        void setNameList(std::vector<std::string> & nameList);
        void setGroupList(std::vector<std::string> & groupList);

        void selectMicrobes(std::string & group, std::vector<std::string> & keys);
        float getGroupValue(std::string group, int i);
        float getMicrobeValue(std::string group, std::string key, int i);
        int getNumDisplayValues();
        std::string getDisplayLabel(int i);

        virtual void dumpState(std::ostream & out);
        virtual bool loadState(std::istream & in);

        virtual bool processEvent(cvr::InteractionEvent * ie);
        virtual void updateCallback(int handID, const osg::Matrix & mat);
        virtual void leaveCallback(int handID);

        virtual void menuCallback(cvr::MenuItem * item);

    protected:
        bool addGraph(std::string & label, std::string query);
        void makeSelect();
        void updateSelect();

        DBManager * _dbm;

        std::vector<std::string> _nameList;
        std::vector<std::string> _groupList;
        std::map<std::string,int> _groupIndexMap;

        float _width, _height;

        VerticalStackedBarGraph * _graph;

        struct Microbe
        {
            char superkingdom[128];
            char phylum[128];
            char mclass[128];
            char order[128];
            char family[128];
            char genus[128];
            char species[128];
            float value;
        };

        static Microbe* _microbeList;
        static int _microbeCount;

        bool _desktopMode;

        struct LoadData
        {
            bool special;
            SpecialMicrobeGraphType type;
            std::string label;
            int patientid;
            std::string testLabel;
        };

        std::vector<struct LoadData> _loadedGraphs;

        osg::ref_ptr<osg::Geode> _selectGeode;
        osg::ref_ptr<osg::Geometry> _selectGeom;
};

#endif
