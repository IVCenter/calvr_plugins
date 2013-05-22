#ifndef FP_GRAPH_OBJECT_H
#define FP_GRAPH_OBJECT_H

#include <cvrKernel/TiledWallSceneObject.h>
#include <cvrMenu/MenuList.h>

#include <string>

#include <mysql++/mysql++.h>

#include "DataGraph.h"
#include "LayoutInterfaces.h"

class GraphObject : public LayoutTypeObject, public TimeRangeObject
{
    public:
        GraphObject(mysqlpp::Connection * conn, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~GraphObject();

        bool addGraph(std::string patient, std::string name);
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

        virtual void dumpState(std::ostream & out);
        virtual bool loadState(std::istream & in);

        void setLayoutDoesDelete(bool b)
        {
            _layoutDoesDelete = b;
        }
        bool getLayoutDoesDelete()
        {
            return _layoutDoesDelete;
        }

        void setGLScale(float scale);

        void perFrame();

        virtual void menuCallback(cvr::MenuItem * item);

        virtual bool processEvent(cvr::InteractionEvent * ie);
        virtual void enterCallback(int handID, const osg::Matrix &mat);
        virtual void updateCallback(int handID, const osg::Matrix &mat);
        virtual void leaveCallback(int handID);
    protected:
        mysqlpp::Connection * _conn;
        std::vector<std::string> _nameList;
        DataGraph * _graph;

        cvr::MenuList * _mgdList;
        cvr::MenuList * _ldmList;

        std::string _pdfDir;

        int _activeHand;
        bool _layoutDoesDelete;

        struct LoadData
        {
            std::string patient;
            std::string name;
            std::string displayName;
        };

        std::vector<LoadData> _loadedGraphs;
};

#endif
