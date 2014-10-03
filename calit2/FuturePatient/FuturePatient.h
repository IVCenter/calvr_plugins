#ifndef CVRPLUGIN_FUTURE_PATIENT_H
#define CVRPLUGIN_FUTURE_PATIENT_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/MenuList.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuRangeValueCompact.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuTextEntryItem.h>
#include <cvrMenu/MenuIntEntryItem.h>
#include <cvrUtil/MultiListenSocket.h>
#include <cvrUtil/CVRSocket.h>

#include <string>
#include <map>
#include <vector>

//#include <mysql++/mysql++.h>
#include "DBManager.h"

#include "GraphObject.h"
#include "GraphLayoutObject.h"
#include "MicrobeGraphObject.h"
#include "MicrobeBarGraphObject.h"
#include "MicrobeScatterGraphObject.h"
#include "SymptomGraphObject.h"
#include "MicrobePointLineObject.h"

struct PhenoStats
{
    std::string name;
    float avg;
    float stdev;
    int taxid;
    std::vector<float> values;
};

struct SortCriteria
{
    float primaryValue;
    float secondaryValue;
};


class FuturePatient : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        FuturePatient();
        virtual ~FuturePatient();

        bool init();
        void preFrame();

        virtual void menuCallback(cvr::MenuItem * item);

        /*static mysqlpp::Connection * getConnection()
        {
            return _conn;
        }*/

        static DBManager * getDBManager()
        {
            return _dbm;
        }

    protected:
        void checkLayout();
        void loadGraph(std::string patient, std::string test, bool averageColor=false);
        //void makeGraph(std::string name);
       
        void setupMicrobes(); 
        void setupMicrobePatients();
        void setupStrainMenu();
        void updateMicrobeTests(int patientid);

        void saveLayout();
        void loadLayout(const std::string & file);

        //static mysqlpp::Connection * _conn;
        static DBManager * _dbm;

        void checkSockets(std::vector<int> & messageList);
        bool processSocketInput(cvr::CVRSocket * socket, std::vector<int> & messageList);

        void loadScatter();
        void loadPhenotype();
        std::vector<std::pair<PhenoStats*,SortCriteria> > createListWithFilters(MicrobeGraphType type, std::string phenotype = "", bool pvalSort = false, bool tvalSort = false, bool averageThresh = false, float avgVal = 0.0, bool reqMax = false, float reqMaxVal = 0.0, bool zeroLimit = false, float zeroVal = 0.0);

        void initPhenoStats(std::map<std::string,std::map<std::string,struct PhenoStats > > & statMap, std::map<std::string,std::map<std::string,struct PhenoStats > > & familyStatMap, std::string microbeSuffix, std::string measureSuffix);

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
        std::map<std::string,std::vector<std::string> > _strainGroupMap;
        std::map<std::string,int> _strainIdMap;
        std::vector<time_t> _microbeTestTime;

        struct MicrobeTableInfo
        {
            std::string microbeSuffix;
            std::string measureSuffix;
            std::map<int,std::vector<std::string> > testMap;
            std::map<int,std::vector<time_t> > testTimeMap;
            std::vector<std::string> microbeList;
            std::vector<int> microbeIDList;
            std::vector<std::string> familyList;
            std::vector<std::string> genusList;
            std::map<std::string,std::map<std::string,struct PhenoStats > > statsMap;
            std::map<std::string,std::map<std::string,struct PhenoStats > > familyStatsMap;
        };

        std::vector<MicrobeTableInfo*> _microbeTableList;

        /*std::map<int,std::vector<std::string> > _patientMicrobeTestMap;
        std::map<int,std::vector<time_t> > _patientMicrobeTestTimeMap;

        std::map<int,std::vector<std::string> > _patientMicrobeV2TestMap;
        std::map<int,std::vector<time_t> > _patientMicrobeV2TestTimeMap;

        std::vector<std::string> _microbeList;
        std::vector<int> _microbeIDList;
        std::vector<std::string> _microbeV2List;
        std::vector<int> _microbeV2IDList;*/

        cvr::MenuCheckbox * _dbCache;

        cvr::SubMenu * _chartMenu;
        cvr::SubMenu * _presetMenu;
        cvr::MenuButton * _inflammationButton;
        cvr::MenuButton * _big4MultiButton;
        cvr::MenuButton * _cholesterolButton;
        cvr::MenuButton * _insGluButton;
        cvr::MenuButton * _inflammationImmuneButton;
        cvr::MenuButton * _loadAll;

        cvr::SubMenu * _microbeMenu;
        cvr::MenuList * _microbeTable;
        cvr::MenuList * _microbeGraphType;
        cvr::MenuList * _microbePatients;
        cvr::MenuList * _microbeTest;
        cvr::MenuButton * _microbeLoad;
        cvr::MenuCheckbox * _microbeOrdering;
        cvr::MenuCheckbox * _microbeGrouping;
        //cvr::MenuCheckbox * _microbeFamilyLevel;
        cvr::MenuList * _microbeLevel;
        cvr::MenuIntEntryItem * _microbeNumBars;
        cvr::MenuButton * _microbeDone;

        cvr::SubMenu * _sMicrobeMenu;
        cvr::MenuTextEntryItem * _sMicrobeEntry;
        cvr::MenuList * _sMicrobes;
        cvr::MenuList * _sMicrobeType;
        cvr::MenuButton * _sMicrobeLoad;
        cvr::MenuCheckbox * _sMicrobeRankOrder;
        cvr::MenuCheckbox * _sMicrobeLabels;
        cvr::MenuCheckbox * _sMicrobeFirstTimeOnly;
        cvr::MenuCheckbox * _sMicrobeGroupPatients;
        cvr::MenuList * _sMicrobePhenotypes;
        cvr::MenuCheckbox * _sMicrobePvalSort;
        cvr::MenuCheckbox * _sMicrobeTvalSort;
        cvr::MenuRangeValueCompact * _sMicrobeSortResults;
        cvr::MenuButton * _sMicrobePhenotypeLoad;
        cvr::SubMenu * _sMicrobePresetMenu;
        //cvr::MenuButton * _sMicrobeBFragilis;
        std::vector<cvr::MenuButton*> _sMicrobePresetList;
        cvr::SubMenu * _sMicrobeFilterMenu;
        cvr::MenuCheckbox * _sMicrobeAvgEnable;
        cvr::MenuRangeValueCompact * _sMicrobeAvgValue;
        cvr::MenuCheckbox * _sMicrobeReqMaxEnable;
        cvr::MenuRangeValueCompact * _sMicrobeReqMaxValue;
        cvr::MenuCheckbox * _sMicrobeZerosEnable;
        cvr::MenuRangeValueCompact * _sMicrobeZerosValue;

        cvr::SubMenu * _microbeSpecialMenu;
        cvr::MenuList * _microbeRegionList;
        cvr::MenuButton * _microbeLoadAverage;
        cvr::MenuButton * _microbeLoadHealthyAverage;
        cvr::MenuButton * _microbeLoadCrohnsAverage;
        cvr::MenuButton * _microbeLoadSRSAverage;
        cvr::MenuButton * _microbeLoadSRXAverage;
        cvr::MenuButton * _microbeLoadCrohnsAll;
        cvr::MenuButton * _microbeLoadHealthyAll;
        cvr::MenuButton * _microbeLoadHealthy105All;
        cvr::MenuButton * _microbeLoadHealthy252All;
        cvr::MenuButton * _microbeLoadUCAll;
        cvr::SubMenu * _microbePointLineMenu;
        cvr::MenuCheckbox * _microbePointLineExpand;
        cvr::MenuButton * _microbeLoadPointLine;

        cvr::SubMenu * _strainMenu;
        cvr::MenuList * _strainGroupList;
        cvr::MenuList * _strainList;
        cvr::MenuButton * _strainLoadButton;
        cvr::MenuButton * _strainLoadAllButton;
        cvr::MenuCheckbox * _strainLarryOnlyCB;
        cvr::MenuButton * _strainLoadHeatMap;

        cvr::SubMenu * _eventMenu;
        cvr::MenuList * _eventName;
        cvr::MenuButton * _eventLoad;
        cvr::MenuButton * _eventLoadAll;
        cvr::MenuButton * _eventLoadMicrobe;
        cvr::MenuButton * _eventDone;

        std::vector<std::string> _scatterPhylumList;
        cvr::SubMenu * _scatterMenu;
        cvr::MenuText * _scatterFirstLabel;
        cvr::MenuText * _scatterSecondLabel;
        cvr::MenuList * _scatterFirstList;
        cvr::MenuList * _scatterSecondList;
        cvr::MenuButton * _scatterLoad;
        cvr::MenuButton * _scatterLoadAll;
        cvr::MenuList * _scatterMicrobeType;
        cvr::MenuTextEntryItem * _scatterFirstEntry;
        cvr::MenuTextEntryItem * _scatterSecondEntry;
        cvr::SubMenu * _scatterFilterMenu;
        cvr::MenuCheckbox * _scatterPvalSort;
        cvr::MenuCheckbox * _scatterTvalSort;
        cvr::MenuCheckbox * _scatterAvgEnable;
        cvr::MenuRangeValueCompact * _scatterAvgValue;
        cvr::MenuCheckbox * _scatterReqMaxEnable;
        cvr::MenuRangeValueCompact * _scatterReqMaxValue;
        cvr::MenuCheckbox * _scatterZerosEnable;
        cvr::MenuRangeValueCompact * _scatterZerosValue;
        cvr::MenuRangeValueCompact * _scatterSortResults;
        cvr::MenuButton * _scatterLoadFilter;
        cvr::MenuList * _scatterPhenotypes;

        MicrobeBarGraphObject * _currentSBGraph;
        SymptomGraphObject * _currentSymptomGraph;

        std::map<std::string,GraphObject*> _graphObjectMap;
        GraphObject * _multiObject;

        GraphLayoutObject * _layoutObject;

        std::vector<MicrobeGraphObject *> _microbeGraphList;

        //std::map<std::string,std::map<std::string,struct PhenoStats > > _microbeStatsMap;
        //std::map<std::string,std::map<std::string,struct PhenoStats > > _microbeV2StatsMap;

        std::string _layoutDirectory;   
        cvr::MultiListenSocket * _mls;
        std::vector<cvr::CVRSocket*> _socketList;
};

#endif
