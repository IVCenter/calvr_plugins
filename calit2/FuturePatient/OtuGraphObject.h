#ifndef FP_OTU_GRAPH_OBJECT_H
#define FP_OTU_GRAPH_OBJECT_H

#include <cvrMenu/MenuText.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuList.h>

#include <string>
#include <map>

#include "DBManager.h"

#include "GroupedBarGraph.h"
#include "LayoutInterfaces.h"

class OtuGraphObject : public LayoutTypeObject, public MicrobeSelectObject, public ValueRangeObject, public SelectableObject
{
    public:
        OtuGraphObject(DBManager * dbm, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~OtuGraphObject();

        bool setGraph(std::string sample, int displayCount);

        void setGraphSize(float width, float height);
        void setColor(osg::Vec4 color);
        void setBGColor(osg::Vec4 color);

        float getGraphMaxValue();
        float getGraphMinValue();

        float getGraphDisplayRangeMax();
        float getGraphDisplayRangeMin();

        void setGraphDisplayRange(float min, float max);
        void resetGraphDisplayRange();

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
        void makeSelect();
        void updateSelect();

        DBManager * _dbm;
        
        std::string _graphTitle;
        std::map<std::string, std::vector<std::pair<std::string, float> > > _graphData;
        std::vector<std::string> _graphOrder;

        float _width, _height;

        GroupedBarGraph * _graph;

        bool _desktopMode;

        osg::ref_ptr<osg::Geode> _selectGeode;
        osg::ref_ptr<osg::Geometry> _selectGeom;
};

#endif
