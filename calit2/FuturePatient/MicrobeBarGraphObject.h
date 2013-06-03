#ifndef MICROBE_BAR_GRAPH_OBJECT_H
#define MICROBE_BAR_GRAPH_OBJECT_H

#include <cvrKernel/TiledWallSceneObject.h>

#include <string>
#include <vector>

#include <mysql++/mysql++.h>

#include "StackedBarGraph.h"
#include "MicrobeGraphObject.h"
#include "LayoutInterfaces.h"

class MicrobeBarGraphObject : public LayoutTypeObject, public MicrobeSelectObject
{
    public:
        MicrobeBarGraphObject(mysqlpp::Connection * conn, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~MicrobeBarGraphObject();

        bool addGraph(std::string label, int patientid, std::string testLabel);
        bool addSpecialGraph(SpecialMicrobeGraphType smgt);

        void setGraphSize(float width, float height);

        void selectMicrobes(std::string & group, std::vector<std::string> & keys);

        virtual void dumpState(std::ostream & out);
        virtual bool loadState(std::istream & in);

        virtual bool processEvent(cvr::InteractionEvent * ie);
        virtual void updateCallback(int handID, const osg::Matrix & mat);
        virtual void leaveCallback(int handID);

    protected:
        bool addGraph(std::string & label, std::string query);

        mysqlpp::Connection * _conn;

        float _width, _height;

        StackedBarGraph * _graph;

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
};

#endif
