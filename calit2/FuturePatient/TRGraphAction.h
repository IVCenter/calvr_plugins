#ifndef TR_GRAPH_ACTION_H
#define TR_GRAPH_ACTION_H

#include <string>

#include <cvrKernel/SceneObject.h>

#include <mysql++/mysql++.h>

class TRGraphAction
{
    public:
        virtual void action(std::string name, time_t start, time_t end, int value) = 0;
};

class MicrobeGraphAction : public TRGraphAction
{
    public:
        virtual void action(std::string name, time_t start, time_t end, int value);

        cvr::SceneObject * symptomObject;
        mysqlpp::Connection * conn;
};

#endif
