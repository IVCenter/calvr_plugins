#ifndef MICROBE_SCATTER_GRAPH_OBJECT_H
#define MICROBE_SCATTER_GRAPH_OBJECT_H

#include <string>

#include <mysql++/mysql++.h>

#include "LayoutInterfaces.h"
#include "GroupedScatterPlot.h"

class MicrobeScatterGraphObject : public LayoutTypeObject
{
    public:
        MicrobeScatterGraphObject(mysqlpp::Connection * conn, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~MicrobeScatterGraphObject();

        bool setGraph(std::string title, std::string primaryPhylum, std::string secondaryPhylum);

        virtual void setGraphSize(float width, float height);

    protected:
        mysqlpp::Connection * _conn;
        GroupedScatterPlot * _graph;
};

#endif
