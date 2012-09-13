#ifndef FP_GRAPH_OBJECT_H
#define FP_GRAPH_OBJECT_H

#include <cvrKernel/TiledWallSceneObject.h>
#include <cvrMenu/MenuList.h>

#include <string>

#include <mysql++/mysql++.h>

#include "DataGraph.h"

class GraphObject : public cvr::TiledWallSceneObject
{
    public:
        GraphObject(mysqlpp::Connection * conn, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~GraphObject();

        bool addGraph(std::string name);
        int getNumGraphs()
        {
            return _graph->getNumGraphs();
        }

        void setGraphSize(float width, float height);
        void setGraphDisplayRange(time_t start, time_t end);
        void resetGraphDisplayRange();

        void getGraphDisplayRange(time_t & start, time_t & end);
        time_t getMaxTimestamp();
        time_t getMinTimestamp();
        void setBarPosition(float pos);
        float getBarPosition();
        void setBarVisible(bool b);
        bool getBarVisible();
        bool getGraphSpacePoint(const osg::Matrix & mat, osg::Vec3 & point);

        void setLayoutDoesDelete(bool b)
        {
            _layoutDoesDelete = b;
        }
        bool getLayoutDoesDelete()
        {
            return _layoutDoesDelete;
        }

        virtual void menuCallback(cvr::MenuItem * item);

        virtual void enterCallback(int handID, const osg::Matrix &mat);
        virtual void updateCallback(int handID, const osg::Matrix &mat);
        virtual void leaveCallback(int handID);
    protected:
        mysqlpp::Connection * _conn;
        std::vector<std::string> _nameList;
        DataGraph * _graph;

        cvr::MenuList * _mgdList;

        int _activeHand;
        bool _layoutDoesDelete;
};

#endif
