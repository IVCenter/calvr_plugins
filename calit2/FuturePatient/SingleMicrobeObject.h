#ifndef SINGLE_MICROBE_OBJECT_H
#define SINGLE_MICROBE_OBJECT_H

#include <string>
#include <map>

#include "DBManager.h"

#include "LayoutInterfaces.h"
#include "GroupedBarGraph.h"
#include "MicrobeGraphObject.h"

#include <osg/LineWidth>

class SingleMicrobeObject : public LayoutTypeObject, public PatientSelectObject
{
    public:
        SingleMicrobeObject(DBManager * dbm, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~SingleMicrobeObject();

        bool setGraph(std::string microbe, std::string titleSuffix, int taxid, std::string microbeTableSuffix, std::string measureTableSuffix, MicrobeGraphType type = MGT_SPECIES, bool rankOrder=true, bool labels=true, bool firstOnly=false, bool groupPatients=false);

        virtual void objectAdded();
        virtual void objectRemoved();

        virtual void setGraphSize(float width, float height);

        virtual void selectPatients(std::map<std::string,std::vector<std::string> > & selectMap);

        virtual bool processEvent(cvr::InteractionEvent * ie);
        virtual void updateCallback(int handID, const osg::Matrix & mat);
        virtual void leaveCallback(int handID);

    protected:
        GroupedBarGraph * _graph;
        DBManager * _dbm;
        bool _desktopMode;

        std::map<std::string,bool> _cdCountMap;
        std::map<std::string,bool> _ucCountMap;
};

class BandingFunction : public MicrobeMathFunction
{
    public:
        virtual void added(osg::Geode * geode);
        virtual void removed(osg::Geode * geode);
        virtual void update(float left, float right, float top, float bottom, float barWidth, std::map<std::string, std::vector<std::pair<std::string, float> > > & data, BarGraphDisplayMode displayMode, const std::vector<std::string> & groupOrder, const std::vector<std::pair<std::string,int> > & customOrder, float displayMin, float displayMax, BarGraphAxisType axisType, const std::vector<std::pair<float,float> > & groupRanges);

    protected:
        osg::ref_ptr<osg::Geometry> _bandGeometry;
        osg::ref_ptr<osg::Geometry> _lineGeometry;
        osg::ref_ptr<SetBoundsCallback> _boundsCallback;
        osg::ref_ptr<osg::LineWidth> _lineWidth;
};

#endif
