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
#include "MicrobeBarGraphObject.h"
#include "SymptomGraphObject.h"


class FuturePatient : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        FuturePatient();
        virtual ~FuturePatient();

        bool init();
        void preFrame();

        virtual void menuCallback(cvr::MenuItem * item);

        static mysqlpp::Connection * getConnection()
        {
            return _conn;
        }

    protected:
        void checkLayout();
        void loadGraph(std::string patient, std::string test);
        //void makeGraph(std::string name);
        
        void setupMicrobePatients();
        void updateMicrobeTests(int patientid);

        void saveLayout();
        void loadLayout(const std::string & file);

        static mysqlpp::Connection * _conn;

        cvr::SubMenu * _fpMenu;
        cvr::SubMenu * _layoutMenu;
        cvr::MenuButton * _saveLayoutButton;
        cvr::SubMenu * _loadLayoutMenu;
        std::vector<cvr::MenuButton*> _loadLayoutButtons;
        cvr::MenuList * _chartPatientList;
        cvr::MenuList * _testList;
        cvr::MenuButton * _loadButton;
        cvr::MenuButton * _removeAllButton;
        cvr::MenuButton * _closeLayoutButton;
        
        cvr::SubMenu * _groupLoadMenu;
        cvr::MenuList * _groupList;
        cvr::MenuButton * _groupLoadButton;

        cvr::MenuCheckbox * _multiAddCB;

        std::map<std::string,std::vector<std::string> > _patientTestMap;
        std::map<std::string,std::vector<std::string> > _groupTestMap;

        cvr::SubMenu * _chartMenu;
        cvr::SubMenu * _presetMenu;
        cvr::MenuButton * _inflammationButton;
        cvr::MenuButton * _big4MultiButton;
        cvr::MenuButton * _cholesterolButton;
        cvr::MenuButton * _insGluButton;
        cvr::MenuButton * _inflammationImmuneButton;
        cvr::MenuButton * _loadAll;

        cvr::SubMenu * _microbeMenu;
        cvr::MenuList * _microbeGraphType;
        cvr::MenuList * _microbePatients;
        cvr::MenuList * _microbeTest;
        cvr::MenuButton * _microbeLoad;
        cvr::MenuCheckbox * _microbeOrdering;
        cvr::MenuRangeValueCompact * _microbeNumBars;
        cvr::MenuButton * _microbeDone;

        cvr::SubMenu * _microbeSpecialMenu;
        cvr::MenuButton * _microbeLoadAverage;
        cvr::MenuButton * _microbeLoadHealthyAverage;
        cvr::MenuButton * _microbeLoadCrohnsAverage;
        cvr::MenuButton * _microbeLoadSRSAverage;
        cvr::MenuButton * _microbeLoadSRXAverage;
        cvr::MenuButton * _microbeLoadCrohnsAll;
        cvr::MenuButton * _microbeLoadHealthyAll;
        cvr::MenuButton * _microbeLoadUCAll;

        cvr::SubMenu * _eventMenu;
        cvr::MenuList * _eventName;
        cvr::MenuButton * _eventLoad;
        cvr::MenuButton * _eventLoadAll;
        cvr::MenuButton * _eventDone;

        MicrobeBarGraphObject * _currentSBGraph;
        SymptomGraphObject * _currentSymptomGraph;

        std::map<std::string,GraphObject*> _graphObjectMap;
        GraphObject * _multiObject;

        GraphLayoutObject * _layoutObject;

        std::vector<MicrobeGraphObject *> _microbeGraphList;

        std::string _layoutDirectory;
};

#endif
