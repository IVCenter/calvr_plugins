#ifndef FP_MICROBE_GRAPH_OBJECT_H
#define FP_MICROBE_GRAPH_OBJECT_H

#include <cvrKernel/TiledWallSceneObject.h>

#include <string>
#include <map>

#include <mysql++/mysql++.h>

#include "GroupedBarGraph.h"
#include "LayoutInterfaces.h"

enum SpecialMicrobeGraphType
{
    SMGT_AVERAGE=0,
    SMGT_HEALTHY_AVERAGE,
    SMGT_CROHNS_AVERAGE,
    SMGT_SRS_AVERAGE,
    SMGT_SRX_AVERAGE
};

class MicrobeGraphObject : public LayoutTypeObject, public MicrobeSelectObject, public ValueRangeObject
{
    public:
        MicrobeGraphObject(mysqlpp::Connection * conn, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~MicrobeGraphObject();

        bool setGraph(std::string title, int patientid, std::string testLabel, int microbes);
        bool setSpecialGraph(SpecialMicrobeGraphType smgt, int microbes);

        void setGraphSize(float width, float height);
        void setColor(osg::Vec4 color);

        float getGraphMaxValue();
        float getGraphMinValue();

        float getGraphDisplayRangeMax();
        float getGraphDisplayRangeMin();

        void setGraphDisplayRange(float min, float max);
        void resetGraphDisplayRange();

        void selectMicrobes(std::string & group, std::vector<std::string> & keys);

        virtual bool processEvent(cvr::InteractionEvent * ie);
        virtual void updateCallback(int handID, const osg::Matrix & mat);
        virtual void leaveCallback(int handID);

    protected:
        bool loadGraphData(std::string valueQuery, std::string orderQuery);

        mysqlpp::Connection * _conn;
        
        std::string _graphTitle;
        std::map<std::string, std::vector<std::pair<std::string, float> > > _graphData;
        std::vector<std::string> _graphOrder;

        float _width, _height;

        GroupedBarGraph * _graph;

        bool _desktopMode;
};

#endif
