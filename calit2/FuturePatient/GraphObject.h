#ifndef FP_GRAPH_OBJECT_H
#define FP_GRAPH_OBJECT_H

#include <cvrKernel/TiledWallSceneObject.h>

#include <string>

#include <mysql++/mysql++.h>

#include "DataGraph.h"

class GraphObject : public cvr::TiledWallSceneObject
{
    public:
        GraphObject(mysqlpp::Connection * conn, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~GraphObject();

        bool addGraph(std::string name);

        void setGraphSize(float width, float height);
        void setGraphDisplayRange(time_t start, time_t end);
        void resetGraphDisplayRange();

        time_t getMaxTimestamp();
        time_t getMinTimestamp();

        virtual void enterCallback(int handID, const osg::Matrix &mat);
        virtual void updateCallback(int handID, const osg::Matrix &mat);
        virtual void leaveCallback(int handID);
    protected:
        mysqlpp::Connection * _conn;
        std::vector<std::string> _nameList;
        DataGraph * _graph;

        int _activeHand;
};

#endif
