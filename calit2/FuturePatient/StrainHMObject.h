#ifndef STRAIN_HM_OBJECT_H
#define STRAIN_HM_OBJECT_H

#include <string>
#include <map>

#include <mysql++/mysql++.h>

#include "HeatMapGraph.h"
#include "LayoutInterfaces.h"

class StrainHMObject : public LayoutTypeObject, public ValueRangeObject
{
    public:
        StrainHMObject(mysqlpp::Connection * conn, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~StrainHMObject();

        void setGraphSize(float width, float height);
        
        bool setGraph(std::string title, std::string patientName, int patientid, int taxid, osg::Vec4 color);

        float getGraphMaxValue();
        float getGraphMinValue();

        float getGraphDisplayRangeMax();
        float getGraphDisplayRangeMin();

        void setGraphDisplayRange(float min, float max);
        void resetGraphDisplayRange();

    protected:
        HeatMapGraph * _graph;
        mysqlpp::Connection * _conn;
};

#endif
