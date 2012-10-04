#ifndef CVRPLUGIN_FUTURE_PATIENT_H
#define CVRPLUGIN_FUTURE_PATIENT_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/MenuList.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuRangeValueCompact.h>
#include <cvrMenu/SubMenu.h>

#include <string>
#include <map>
#include <vector>

#include <mysql++/mysql++.h>

#include "GraphObject.h"
#include "GraphLayoutObject.h"
#include "MicrobeGraphObject.h"


class FuturePatient : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        FuturePatient();
        virtual ~FuturePatient();

        bool init();

        virtual void menuCallback(cvr::MenuItem * item);

    protected:
        void checkLayout();
        void loadGraph(std::string name);
        //void makeGraph(std::string name);
        
        void setupMicrobePatients();
        void updateMicrobeTests(int patientid);


        mysqlpp::Connection * _conn;

        cvr::SubMenu * _fpMenu;
        cvr::MenuList * _testList;
        cvr::MenuButton * _loadButton;
        cvr::MenuButton * _removeAllButton;
        
        cvr::SubMenu * _groupLoadMenu;
        cvr::MenuList * _groupList;
        cvr::MenuButton * _groupLoadButton;

        cvr::MenuCheckbox * _multiAddCB;

        std::map<std::string,std::vector<std::string> > _groupTestMap;

        cvr::SubMenu * _presetMenu;
        cvr::MenuButton * _inflammationButton;
        cvr::MenuButton * _loadAll;

        cvr::SubMenu * _microbeMenu;
        cvr::MenuList * _microbePatients;
        cvr::MenuList * _microbeTest;
        cvr::MenuButton * _microbeLoad;
        cvr::MenuRangeValueCompact * _microbeNumBars;

        cvr::SubMenu * _microbeSpecialMenu;
        cvr::MenuButton * _microbeLoadAverage;
        cvr::MenuButton * _microbeLoadHealthyAverage;
        cvr::MenuButton * _microbeLoadCrohnsAverage;
        cvr::MenuButton * _microbeLoadSRSAverage;
        cvr::MenuButton * _microbeLoadSRXAverage;

        //Temp until layout so is created
        std::map<std::string,GraphObject*> _graphObjectMap;
        GraphObject * _multiObject;

        GraphLayoutObject * _layoutObject;

        std::vector<MicrobeGraphObject *> _microbeGraphList;
};

#endif
