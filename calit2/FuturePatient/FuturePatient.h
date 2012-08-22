#ifndef CVRPLUGIN_FUTURE_PATIENT_H
#define CVRPLUGIN_FUTURE_PATIENT_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/MenuList.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/SubMenu.h>

#include <string>
#include <map>
#include <vector>

#include <mysql++/mysql++.h>

#include "GraphObject.h"
#include "GraphLayoutObject.h"

class FuturePatient : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        FuturePatient();
        virtual ~FuturePatient();

        bool init();

        virtual void menuCallback(cvr::MenuItem * item);

    protected:
        void loadGraph(std::string name);
        //void makeGraph(std::string name);

        mysqlpp::Connection * _conn;

        cvr::SubMenu * _fpMenu;
        cvr::MenuList * _testList;
        cvr::MenuButton * _loadButton;
        cvr::MenuButton * _removeAllButton;
        
        cvr::SubMenu * _groupLoadMenu;
        cvr::MenuList * _groupList;
        cvr::MenuButton * _groupLoadButton;

        std::map<std::string,std::vector<std::string> > _groupTestMap;

        cvr::SubMenu * _presetMenu;
        cvr::MenuButton * _inflammationButton;
        cvr::MenuButton * _loadAll;

        //Temp until layout so is created
        std::map<std::string,GraphObject*> _graphObjectMap;

        GraphLayoutObject * _layoutObject;
};

#endif
