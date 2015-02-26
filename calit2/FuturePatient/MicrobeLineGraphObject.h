#ifndef FP_MICROBE_LINE_GRAPH_OBJECT_H
#define FP_MICROBE_LINE_GRAPH_OBJECT_H

#include <cvrMenu/MenuList.h>
#include <cvrMenu/MenuCheckbox.h>

#include <string>

#include "DBManager.h"

#include "MicrobeGraphObject.h"
#include "DataGraph.h"
#include "LayoutInterfaces.h"
#include "LinearRegFunc.h"

class MicrobeLineGraphObject : public LayoutTypeObject, public TimeRangeObject
{
    public:
        MicrobeLineGraphObject(DBManager * dbm, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~MicrobeLineGraphObject();

        bool addGraph(std::string patient, std::string microbe, MicrobeGraphType type, std::string microbeTableSuffix, std::string measureTableSuffix);
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

        void setLinearRegression(bool lr);

        void perFrame();

        virtual void menuCallback(cvr::MenuItem * item);

        virtual bool processEvent(cvr::InteractionEvent * ie);
        virtual void enterCallback(int handID, const osg::Matrix &mat);
        virtual void updateCallback(int handID, const osg::Matrix &mat);
        virtual void leaveCallback(int handID);

        int getNumMathFunctions();
        MathFunction * getMathFunction(int i);
    protected:
        DBManager * _dbm;
        std::vector<std::string> _nameList;
        DataGraph * _graph;

        cvr::MenuList * _mgdList;
        cvr::MenuList * _ldmList;

        int _activeHand;
        bool _layoutDoesDelete;

        AverageFunction * _averageFunc;
        cvr::MenuCheckbox * _averageCB;
        LinearRegFunc * _linRegFunc;
        cvr::MenuCheckbox * _linRegCB;
};

#endif
