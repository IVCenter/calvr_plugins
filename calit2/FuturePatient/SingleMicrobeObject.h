#ifndef SINGLE_MICROBE_OBJECT_H
#define SINGLE_MICROBE_OBJECT_H

#include <string>
#include <map>

#include <mysql++/mysql++.h>

#include "LayoutInterfaces.h"
#include "GroupedBarGraph.h"

class SingleMicrobeObject : public LayoutTypeObject, public PatientSelectObject
{
    public:
        SingleMicrobeObject(mysqlpp::Connection * conn, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~SingleMicrobeObject();

        bool setGraph(std::string microbe, int taxid, std::string tableSuffix);

        virtual void objectAdded();
        virtual void objectRemoved();

        virtual void setGraphSize(float width, float height);

        virtual void selectPatients(std::string & group, std::vector<std::string> & patients);

        virtual bool processEvent(cvr::InteractionEvent * ie);
        virtual void updateCallback(int handID, const osg::Matrix & mat);
        virtual void leaveCallback(int handID);

    protected:
        GroupedBarGraph * _graph;
        mysqlpp::Connection * _conn;
        bool _desktopMode;

        std::map<std::string,bool> _cdCountMap;
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
};

#endif
