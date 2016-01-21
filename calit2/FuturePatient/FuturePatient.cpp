#include "FuturePatient.h"
#include "DataGraph.h"
#include "StrainGraphObject.h"
#include "StrainHMObject.h"
#include "SingleMicrobeObject.h"
#include "MicrobeLineGraphObject.h"
#include "MicrobeGraphWAllObject.h"
#include "OtuGraphObject.h"

#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/ComController.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/ScreenBase.h>
#include <cvrKernel/CVRViewer.h>
#include <cvrConfig/ConfigManager.h>
#include <cvrMenu/MenuBar.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <sstream>
#include <fstream>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <osgDB/WriteFile>

#include <mysql++/mysql++.h>

#include <sys/time.h>

#include <octave/oct.h>
#include <octave/octave.h>
#include <octave/parse.h>
#include <octave/toplev.h>

#define SAVED_LAYOUT_VERSION 3

using namespace cvr;

CVRPLUGIN(FuturePatient)

//mysqlpp::Connection * FuturePatient::_conn = NULL;
DBManager * FuturePatient::_dbm = NULL;

FuturePatient::FuturePatient()
{
    _layoutObject = NULL;
    _multiObject = NULL;
    _currentSBGraph = NULL;
    _currentSymptomGraph = NULL;
    _currentVBGraph = NULL;
    _mls = NULL;
    _takeSubImage = false;
    _subImageDone = false;

    string_vector argv (2);
    argv(0) = "embedded";
    argv(1) = "-q";

    octave_main(2, argv.c_str_vec(), 1);
}

FuturePatient::~FuturePatient()
{
    if(_mls)
    {
        delete _mls;
    }

    clean_up_and_exit(0);
}

bool FuturePatient::init()
{
    if(ComController::instance()->isMaster())
    {
        int port = ConfigManager::getInt("value","Plugin.FuturePatient.PresetListenPort",12012);
        _mls = new MultiListenSocket(port);
        if(!_mls->setup())
        {
            std::cerr << "Error setting up MultiListen Socket on port " << port << " ." << std::endl;
            delete _mls;
            _mls = NULL;
        }
    }

    _fpMenu = new SubMenu("FuturePatient");

    _layoutMenu = new SubMenu("Layouts");
    _fpMenu->addItem(_layoutMenu);

    _loadLayoutMenu = new SubMenu("Load");
    _layoutMenu->addItem(_loadLayoutMenu);

    _saveLayoutButton = new MenuButton("Save");
    _saveLayoutButton->setCallback(this);
    _layoutMenu->addItem(_saveLayoutButton);

    _chartMenu = new SubMenu("Charts");
    _fpMenu->addItem(_chartMenu);

    _presetMenu = new SubMenu("Presets");
    _chartMenu->addItem(_presetMenu);

    _inflammationButton = new MenuButton("Big 4 (Sep)");
    _inflammationButton->setCallback(this);
    _presetMenu->addItem(_inflammationButton);

    _big4MultiButton = new MenuButton("Big 4 (Multi)");
    _big4MultiButton->setCallback(this);
    _presetMenu->addItem(_big4MultiButton);

    _cholesterolButton = new MenuButton("Cholesterol");
    _cholesterolButton->setCallback(this);
    _presetMenu->addItem(_cholesterolButton);

    _insGluButton = new MenuButton("Insulin/Glucose");
    _insGluButton->setCallback(this);
    _presetMenu->addItem(_insGluButton);

    _inflammationImmuneButton = new MenuButton("Inflammation (Immune)");
    _inflammationImmuneButton->setCallback(this);
    _presetMenu->addItem(_inflammationImmuneButton);

    _loadAll = new MenuButton("All");
    _loadAll->setCallback(this);
    _presetMenu->addItem(_loadAll);

    _groupLoadMenu = new SubMenu("Group Load");
    _chartMenu->addItem(_groupLoadMenu);

    _chartPatientList = new MenuList();
    _chartPatientList->setCallback(this);
    _chartMenu->addItem(_chartPatientList);

    _testList = new MenuList();
    _testList->setCallback(this);
    _testList->setScrollingHint(MenuList::ONE_TO_ONE);
    _chartMenu->addItem(_testList);

    _linearRegCB = new MenuCheckbox("Linear Regression",false);
    _linearRegCB->setCallback(this);
    _chartMenu->addItem(_linearRegCB);

    _loadAllAvg = new MenuCheckbox("Load All Avg",true);
    _loadAllAvg->setCallback(this);
    _chartMenu->addItem(_loadAllAvg);

    std::vector<std::string> gdtText;
    gdtText.push_back("None");
    gdtText.push_back("Points");
    gdtText.push_back("Points w/ Lines");

    _graphTypeList = new MenuList();
    _graphTypeList->setCallback(this);
    _graphTypeList->setCallbackType(MenuList::ON_RELEASE);
    _graphTypeList->setValues(gdtText);
    _chartMenu->addItem(_graphTypeList);

    _requireRangeCB = new MenuCheckbox("Require Range",true);
    _requireRangeCB->setCallback(this);
    _chartMenu->addItem(_requireRangeCB);

    _loadButton = new MenuButton("Load");
    _loadButton->setCallback(this);
    _chartMenu->addItem(_loadButton);

    _multiAddCB = new MenuCheckbox("Multi Add", false);
    _multiAddCB->setCallback(this);
    _chartMenu->addItem(_multiAddCB);

    _microbeMenu = new SubMenu("Microbe Data");
    _fpMenu->addItem(_microbeMenu);

    _microbeSpecialMenu = new SubMenu("Special");
    _microbeMenu->addItem(_microbeSpecialMenu);

    _microbeRegionList = new MenuList();
    _microbeRegionList->setCallback(this);

    std::vector<std::string> regions;
    regions.push_back("All");
    regions.push_back("US");
    regions.push_back("EU");
    _microbeRegionList->setValues(regions);
    _microbeRegionList->setIndex(1);
    _microbeRegionList->setScrollingHint(MenuList::ONE_TO_ONE);

    _microbeSpecialMenu->addItem(_microbeRegionList);

    _microbeLoadAverage = new MenuButton("UC Average");
    _microbeLoadAverage->setCallback(this);
    _microbeSpecialMenu->addItem(_microbeLoadAverage);

    _microbeLoadHealthyAverage = new MenuButton("Healthy Average");
    _microbeLoadHealthyAverage->setCallback(this);
    _microbeSpecialMenu->addItem(_microbeLoadHealthyAverage);

    _microbeLoadCrohnsAverage = new MenuButton("Crohns Average");
    _microbeLoadCrohnsAverage->setCallback(this);
    _microbeSpecialMenu->addItem(_microbeLoadCrohnsAverage);

    _microbeLoadSmarrAverage = new MenuButton("Smarr Average");
    _microbeLoadSmarrAverage->setCallback(this);
    _microbeSpecialMenu->addItem(_microbeLoadSmarrAverage);

    _microbeLoadSRSAverage = new MenuButton("SRS Average");
    _microbeLoadSRSAverage->setCallback(this);
    //_microbeSpecialMenu->addItem(_microbeLoadSRSAverage);

    _microbeLoadSRXAverage = new MenuButton("SRX Average");
    _microbeLoadSRXAverage->setCallback(this);
    //_microbeSpecialMenu->addItem(_microbeLoadSRXAverage);

    _microbeLoadCrohnsAll = new MenuButton("Crohns All");
    _microbeLoadCrohnsAll->setCallback(this);
    _microbeSpecialMenu->addItem(_microbeLoadCrohnsAll);

    _microbeLoadHealthyAll = new MenuButton("Healthy 35");
    _microbeLoadHealthyAll->setCallback(this);
    _microbeSpecialMenu->addItem(_microbeLoadHealthyAll);

    _microbeLoadHealthy105All = new MenuButton("Healthy 155");
    _microbeLoadHealthy105All->setCallback(this);
    _microbeSpecialMenu->addItem(_microbeLoadHealthy105All);

    _microbeLoadHealthy252All = new MenuButton("Healthy 252");
    _microbeLoadHealthy252All->setCallback(this);
    _microbeSpecialMenu->addItem(_microbeLoadHealthy252All);

    _microbeLoadUCAll = new MenuButton("UC All");
    _microbeLoadUCAll->setCallback(this);
    _microbeSpecialMenu->addItem(_microbeLoadUCAll);

    _microbeLoadLarryAll = new MenuButton("Larry All");
    _microbeLoadLarryAll->setCallback(this);
    _microbeSpecialMenu->addItem(_microbeLoadLarryAll);

    _microbePointLineMenu = new SubMenu("Point Line Graph");
    _microbeMenu->addItem(_microbePointLineMenu);

    _microbePointLineExpand = new MenuCheckbox("Expand Axis", false);
    _microbePointLineExpand->setCallback(this);
    _microbePointLineMenu->addItem(_microbePointLineExpand);

    _microbeLoadPointLine = new MenuButton("Load");
    _microbeLoadPointLine->setCallback(this);
    _microbePointLineMenu->addItem(_microbeLoadPointLine);

    _sMicrobeMenu = new SubMenu("Single Microbe");
    _microbeMenu->addItem(_sMicrobeMenu);

    _sMicrobePresetMenu = new SubMenu("Presets");
    _sMicrobeMenu->addItem(_sMicrobePresetMenu);

    /*_sMicrobeBFragilis = new MenuButton("Bacteroides fragilis");
    _sMicrobeBFragilis->setCallback(this);
    _sMicrobePresetMenu->addItem(_sMicrobeBFragilis);*/

    std::vector<std::string> microbeTagList;
    ConfigManager::getChildren("Plugin.FuturePatient.MicrobePresets",microbeTagList);
    for(int i = 0; i < microbeTagList.size(); ++i)
    {
	std::string path = "Plugin.FuturePatient.MicrobePresets";
	path = path + "." + microbeTagList[i];
	MenuButton * mb = new MenuButton(ConfigManager::getEntry("value",path,"Name"));
	mb->setCallback(this);
	_sMicrobePresetList.push_back(mb);
	_sMicrobePresetMenu->addItem(mb);
    }

    _sMicrobeType = new MenuList();
    _sMicrobeType->setCallback(this);
    _sMicrobeMenu->addItem(_sMicrobeType);

    std::vector<std::string> mtypeList;
    mtypeList.push_back("Microbe");
    mtypeList.push_back("Family");
    mtypeList.push_back("Genus");
    mtypeList.push_back("Phylum");
    _sMicrobeType->setValues(mtypeList);

    _sMicrobes = new MenuList();
    _sMicrobes->setCallback(this);
    _sMicrobeMenu->addItem(_sMicrobes);
    _sMicrobes->setSensitivity(1.0);
    _sMicrobes->setScrollingHint(MenuList::CONTINUOUS);

    _sMicrobeEntry = new MenuTextEntryItem("Manual Entry","",MenuItemGroup::ALIGN_CENTER);
    _sMicrobeEntry->setCallback(this);
    _sMicrobeMenu->addItem(_sMicrobeEntry);

    _sMicrobeRankOrder = new MenuCheckbox("Rank Order",true);
    _sMicrobeRankOrder->setCallback(this);
    _sMicrobeMenu->addItem(_sMicrobeRankOrder);

    _sMicrobeLabels = new MenuCheckbox("Labels",true);
    _sMicrobeLabels->setCallback(this);
    _sMicrobeMenu->addItem(_sMicrobeLabels);

    _sMicrobeLogCB = new MenuCheckbox("Log Scale",true);
    _sMicrobeLogCB->setCallback(this);
    _sMicrobeMenu->addItem(_sMicrobeLogCB);

    _sMicrobeStdDevCB = new MenuCheckbox("Std Dev",false);
    _sMicrobeStdDevCB->setCallback(this);
    _sMicrobeMenu->addItem(_sMicrobeStdDevCB);

    _sMicrobeFirstTimeOnly = new MenuCheckbox("First Time Only",false);
    _sMicrobeFirstTimeOnly->setCallback(this);
    _sMicrobeMenu->addItem(_sMicrobeFirstTimeOnly);

    _sMicrobeGroup = new MenuCheckbox("Group",true);
    _sMicrobeGroup->setCallback(this);
    _sMicrobeMenu->addItem(_sMicrobeGroup);

    _sMicrobeGroupPatients = new MenuCheckbox("Split Patients",false);
    _sMicrobeGroupPatients->setCallback(this);
    _sMicrobeMenu->addItem(_sMicrobeGroupPatients);

    _sMicrobeChartType = new MenuList();
    _sMicrobeChartType->setCallback(this);
    _sMicrobeMenu->addItem(_sMicrobeChartType);

    std::vector<std::string> chartTypes;
    chartTypes.push_back("Bar");
    chartTypes.push_back("Line");

    _sMicrobeChartType->setValues(chartTypes);
    _sMicrobeChartType->setScrollingHint(MenuList::ONE_TO_ONE);

    _sMicrobeLoad = new MenuButton("Load");
    _sMicrobeLoad->setCallback(this);
    _sMicrobeMenu->addItem(_sMicrobeLoad);

    _sMicrobeLoadLines = new MenuButton("Load Line Chart");
    _sMicrobeLoadLines->setCallback(this);
    //_sMicrobeMenu->addItem(_sMicrobeLoadLines);

    _sMicrobePhenotypes = new MenuList();
    _sMicrobePhenotypes->setCallback(this);
    _sMicrobePhenotypes->setScrollingHint(MenuList::ONE_TO_ONE);
    _sMicrobeMenu->addItem(_sMicrobePhenotypes);

    std::vector<std::string> pheno;
    pheno.push_back("Smarr");
    pheno.push_back("Crohns");
    pheno.push_back("UC");
    pheno.push_back("Healthy");
    _sMicrobePhenotypes->setValues(pheno);

    _sMicrobeSecondPhenotype = new MenuList();
    _sMicrobeSecondPhenotype->setCallback(this);
    _sMicrobeSecondPhenotype->setScrollingHint(MenuList::ONE_TO_ONE);
    _sMicrobeMenu->addItem(_sMicrobeSecondPhenotype);

    pheno.clear();
    pheno.push_back("Smarr");
    pheno.push_back("Crohns");
    pheno.push_back("UC");
    pheno.push_back("Healthy");
    _sMicrobeSecondPhenotype->setValues(pheno);

    _sMicrobeFilterMenu = new SubMenu("Filters");
    _sMicrobeMenu->addItem(_sMicrobeFilterMenu);

    _sMicrobePvalSort = new MenuCheckbox("P Val Enable",false);
    _sMicrobePvalSort->setCallback(this);
    _sMicrobeFilterMenu->addItem(_sMicrobePvalSort);

    _sMicrobeTvalSort = new MenuCheckbox("With T Val",false);
    _sMicrobeTvalSort->setCallback(this);
    _sMicrobeFilterMenu->addItem(_sMicrobeTvalSort);

    _sMicrobeTvalSortType = new MenuList();
    _sMicrobeTvalSortType->setCallback(this);
    _sMicrobeFilterMenu->addItem(_sMicrobeTvalSortType);

    std::vector<std::string> tvaltype;
    tvaltype.push_back("Min Group T");
    tvaltype.push_back("Avg Group T");
    tvaltype.push_back("Group Pair T");
    tvaltype.push_back("Group Avg Dev");

    _sMicrobeTvalSortType->setValues(tvaltype);

    MenuBar * sMicrobeBar = new MenuBar(osg::Vec4(1.0,1.0,1.0,1.0));
    _sMicrobeFilterMenu->addItem(sMicrobeBar);

    _sMicrobeAvgEnable = new MenuCheckbox("Average Enable",false);
    _sMicrobeAvgEnable->setCallback(this);
    _sMicrobeFilterMenu->addItem(_sMicrobeAvgEnable);

    _sMicrobeAvgValue = new MenuRangeValueCompact("Avg Threshold",0.0,1.0,0.0001);
    _sMicrobeAvgValue->setCallback(this);
    _sMicrobeFilterMenu->addItem(_sMicrobeAvgValue);

    sMicrobeBar = new MenuBar(osg::Vec4(1.0,1.0,1.0,1.0));
    _sMicrobeFilterMenu->addItem(sMicrobeBar);

    _sMicrobeReqMaxEnable = new MenuCheckbox("Req Max Enable",false);
    _sMicrobeReqMaxEnable->setCallback(this);
    _sMicrobeFilterMenu->addItem(_sMicrobeReqMaxEnable);

    _sMicrobeReqMaxValue = new MenuRangeValueCompact("Value Threshold",0.0,1.0,0.0001);
    _sMicrobeReqMaxValue->setCallback(this);
    _sMicrobeFilterMenu->addItem(_sMicrobeReqMaxValue);

    sMicrobeBar = new MenuBar(osg::Vec4(1.0,1.0,1.0,1.0));
    _sMicrobeFilterMenu->addItem(sMicrobeBar);

    _sMicrobeZerosEnable = new MenuCheckbox("Zeros Enable",false);
    _sMicrobeZerosEnable->setCallback(this);
    _sMicrobeFilterMenu->addItem(_sMicrobeZerosEnable);

    _sMicrobeZerosValue = new MenuRangeValueCompact("Zeros Threshold",0.0,1.0,0.25);
    _sMicrobeZerosValue->setCallback(this);
    _sMicrobeFilterMenu->addItem(_sMicrobeZerosValue);

    _sMicrobeSortResults = new MenuRangeValueCompact("Num Results",5.0,50.0,20.0);
    _sMicrobeSortResults->setCallback(this);
    _sMicrobeMenu->addItem(_sMicrobeSortResults);

    _sMicrobePhenotypeLoad = new MenuButton("Load(Phenotype)");
    _sMicrobePhenotypeLoad->setCallback(this);
    _sMicrobeMenu->addItem(_sMicrobePhenotypeLoad);

    _microbeTable = new MenuList();
    _microbeTable->setCallback(this);
    _microbeTable->setScrollingHint(MenuList::ONE_TO_ONE);
    _microbeMenu->addItem(_microbeTable);

    std::vector<std::string> tableList;

    _microbeTableList.push_back(new MicrobeTableInfo);
    _microbeTableList.back()->microbeSuffix = "";
    _microbeTableList.back()->measureSuffix = "";
    tableList.push_back("V1");

    _microbeTableList.push_back(new MicrobeTableInfo);
    _microbeTableList.back()->microbeSuffix = "_V2";
    _microbeTableList.back()->measureSuffix = "_V2";
    tableList.push_back("V2");

    _microbeTableList.push_back(new MicrobeTableInfo);
    _microbeTableList.back()->microbeSuffix = "";
    _microbeTableList.back()->measureSuffix = "_2014_02_19";
    tableList.push_back("2014_02_19");

    _microbeTable->setValues(tableList);
    _microbeTable->setIndex(2);

    _microbeGraphType = new MenuList();
    _microbeGraphType->setCallback(this);
    _microbeGraphType->setScrollingHint(MenuList::ONE_TO_ONE);
    _microbeMenu->addItem(_microbeGraphType);

    std::vector<std::string> mGraphTypes;
    mGraphTypes.push_back("Bar Graph");
    mGraphTypes.push_back("Stacked Bar Graph");
    mGraphTypes.push_back("Vertical Bar Graph");
    mGraphTypes.push_back("Graph with All");
    _microbeGraphType->setValues(mGraphTypes);

    _microbePatients = new MenuList();
    _microbePatients->setCallback(this);
    _microbeMenu->addItem(_microbePatients);

    _microbeTest = new MenuList();
    _microbeTest->setCallback(this);
    _microbeTest->setScrollingHint(MenuList::ONE_TO_ONE);
    _microbeMenu->addItem(_microbeTest);
    _microbeTest->setSensitivity(2.0);

    //_microbeNumBars = new MenuRangeValueCompact("Microbes",1,100,25);
    //_microbeNumBars = new MenuRangeValueCompact("Microbes",1,2000,25,true);
    _microbeNumBars = new MenuIntEntryItem("Microbes: ",60);
    _microbeNumBars->setCallback(this);
    _microbeMenu->addItem(_microbeNumBars);

    _microbeOrdering = new MenuCheckbox("LS Ordering", true);
    _microbeOrdering->setCallback(this);
    _microbeMenu->addItem(_microbeOrdering);

    _microbeGrouping = new MenuCheckbox("Group",true);
    _microbeGrouping->setCallback(this);
    _microbeMenu->addItem(_microbeGrouping);

/*
    _microbeFamilyLevel = new MenuCheckbox("Family Level",false);
    _microbeFamilyLevel->setCallback(this);
    _microbeMenu->addItem(_microbeFamilyLevel);
*/
    
    _microbeLevel = new MenuList();
    _microbeLevel->setCallback(this);
    _microbeLevel->setScrollingHint(MenuList::ONE_TO_ONE);
    _microbeMenu->addItem(_microbeLevel);

    std::vector<std::string> mGraphLevel;
    mGraphLevel.push_back("Microbe");
    mGraphLevel.push_back("Family");
    mGraphLevel.push_back("Genus");
    _microbeLevel->setValues(mGraphLevel);

    _microbeLoad = new MenuButton("Load");
    _microbeLoad->setCallback(this);
    _microbeMenu->addItem(_microbeLoad);

    _microbeDone = new MenuButton("Done");
    _microbeDone->setCallback(this);

    _otuMenu = new SubMenu("OTUs");
    _fpMenu->addItem(_otuMenu);

    _otuType = new MenuList();
    _otuType->setCallback(this);
    _otuMenu->addItem(_otuType);

    std::vector<std::string> otypeList;
    otypeList.push_back("Species");
    otypeList.push_back("Family");
    otypeList.push_back("Genus");
    otypeList.push_back("Phylum");
    _otuType->setValues(otypeList);

    _otuSampleList = new MenuList();
    _otuSampleList->setCallback(this);
    _otuSampleList->setScrollingHint(MenuList::CONTINUOUS);
    _otuMenu->addItem(_otuSampleList);

    initOtuMenu();

    _otuCount = new MenuIntEntryItem("Count: ",200);
    _otuCount->setCallback(this);
    _otuMenu->addItem(_otuCount);

    _otuLoad = new MenuButton("Load");
    _otuLoad->setCallback(this);
    _otuMenu->addItem(_otuLoad);

    _strainMenu = new SubMenu("Strains");
    _fpMenu->addItem(_strainMenu);

    _strainGroupList = new MenuList();
    _strainGroupList->setCallback(this);
    _strainGroupList->setScrollingHint(MenuList::ONE_TO_ONE);
    _strainMenu->addItem(_strainGroupList);

    _strainList = new MenuList();
    _strainList->setCallback(this);
    _strainList->setScrollingHint(MenuList::CONTINUOUS);
    _strainMenu->addItem(_strainList);

    _strainLarryOnlyCB = new MenuCheckbox("Larry Only",false);
    _strainLarryOnlyCB->setCallback(this);
    _strainMenu->addItem(_strainLarryOnlyCB);

    _strainLoadButton = new MenuButton("Load");
    _strainLoadButton->setCallback(this);
    _strainMenu->addItem(_strainLoadButton);

    _strainLoadAllButton = new MenuButton("Load All");
    _strainLoadAllButton->setCallback(this);
    _strainMenu->addItem(_strainLoadAllButton);

    _strainLoadHeatMap = new MenuButton("Load Heat Map");
    _strainLoadHeatMap->setCallback(this);
    _strainMenu->addItem(_strainLoadHeatMap);

    _eventMenu = new SubMenu("Events");
    _fpMenu->addItem(_eventMenu);

    _eventName = new MenuList();
    _eventName->setCallback(this);
    _eventName->setScrollingHint(MenuList::ONE_TO_ONE);
    _eventMenu->addItem(_eventName);

    _eventLoad = new MenuButton("Load");
    _eventLoad->setCallback(this);
    _eventMenu->addItem(_eventLoad);

    _eventLoadDisplay = new MenuButton("Load Display Set");
    _eventLoadDisplay->setCallback(this);
    _eventMenu->addItem(_eventLoadDisplay);

    _eventLoadAll = new MenuButton("Load All");
    _eventLoadAll->setCallback(this);
    _eventMenu->addItem(_eventLoadAll);

    _eventLoadMicrobe = new MenuButton("Load Microbe");
    _eventLoadMicrobe->setCallback(this);
    //_eventMenu->addItem(_eventLoadMicrobe);

    _eventDone = new MenuButton("Done");
    _eventDone->setCallback(this);

    _scatterMenu = new SubMenu("Scatter Plots");
    _fpMenu->addItem(_scatterMenu);

    _scatterPresetMenu = new SubMenu("Presets");
    _scatterMenu->addItem(_scatterPresetMenu);

    _scatterPresetDir = ConfigManager::getEntry("value","Plugin.FuturePatient.ScatterDir","");
    if(!_scatterPresetDir.empty())
    {
	loadScatterPresets();
    }

    _scatterMicrobeType = new MenuList();
    _scatterMicrobeType->setCallback(this);
    _scatterMenu->addItem(_scatterMicrobeType);

    _scatterFilterMenu = new SubMenu("Filters");

    _scatterPvalSort = new MenuCheckbox("P Val Enable",false);
    _scatterPvalSort->setCallback(this);
    _scatterFilterMenu->addItem(_scatterPvalSort);

    _scatterTvalSort = new MenuCheckbox("With T Val",false);
    _scatterTvalSort->setCallback(this);
    _scatterFilterMenu->addItem(_scatterTvalSort);

    _scatterTvalSortType = new MenuList();
    _scatterTvalSortType->setCallback(this);
    _scatterFilterMenu->addItem(_scatterTvalSortType);

    _scatterTvalSortType->setValues(tvaltype);

    sMicrobeBar = new MenuBar(osg::Vec4(1.0,1.0,1.0,1.0));
    _scatterFilterMenu->addItem(sMicrobeBar);

    _scatterAvgEnable = new MenuCheckbox("Average Enable",false);
    _scatterAvgEnable->setCallback(this);
    _scatterFilterMenu->addItem(_scatterAvgEnable);

    _scatterAvgValue = new MenuRangeValueCompact("Avg Threshold",0.0,1.0,0.0001);
    _scatterAvgValue->setCallback(this);
    _scatterFilterMenu->addItem(_scatterAvgValue);

    sMicrobeBar = new MenuBar(osg::Vec4(1.0,1.0,1.0,1.0));
    _scatterFilterMenu->addItem(sMicrobeBar);

    _scatterReqMaxEnable = new MenuCheckbox("Req Max Enable",false);
    _scatterReqMaxEnable->setCallback(this);
    _scatterFilterMenu->addItem(_scatterReqMaxEnable);

    _scatterReqMaxValue = new MenuRangeValueCompact("Value Threshold",0.0,1.0,0.0001);
    _scatterReqMaxValue->setCallback(this);
    _scatterFilterMenu->addItem(_scatterReqMaxValue);

    sMicrobeBar = new MenuBar(osg::Vec4(1.0,1.0,1.0,1.0));
    _scatterFilterMenu->addItem(sMicrobeBar);

    _scatterZerosEnable = new MenuCheckbox("Zeros Enable",false);
    _scatterZerosEnable->setCallback(this);
    _scatterFilterMenu->addItem(_scatterZerosEnable);

    _scatterZerosValue = new MenuRangeValueCompact("Zeros Threshold",0.0,1.0,0.25);
    _scatterZerosValue->setCallback(this);
    _scatterFilterMenu->addItem(_scatterZerosValue);

    std::vector<std::string> typeList;
    typeList.push_back("Species");
    typeList.push_back("Family");
    typeList.push_back("Genus");
    typeList.push_back("Phylum");
    _scatterMicrobeType->setValues(typeList);
    _scatterMicrobeType->setIndex(3);

    _scatterFirstLabel = new MenuText("Primary:",1.0,false);
    _scatterMenu->addItem(_scatterFirstLabel);

    _scatterFirstList = new MenuList();
    _scatterMenu->addItem(_scatterFirstList);

    _scatterFirstEntry = new MenuTextEntryItem("Manual Entry","",MenuItemGroup::ALIGN_CENTER);
    _scatterFirstEntry->setCallback(this);
    _scatterMenu->addItem(_scatterFirstEntry);

    _scatterSecondLabel = new MenuText("Secondary:",1.0,false);
    _scatterMenu->addItem(_scatterSecondLabel);

    _scatterSecondList = new MenuList();
    _scatterMenu->addItem(_scatterSecondList);

    _scatterSecondEntry = new MenuTextEntryItem("Manual Entry","",MenuItemGroup::ALIGN_CENTER);
    _scatterSecondEntry->setCallback(this);
    _scatterMenu->addItem(_scatterSecondEntry);

    _scatterLogCB = new MenuCheckbox("Log Scale",true);
    _scatterLogCB->setCallback(this);
    _scatterMenu->addItem(_scatterLogCB);

    _scatterLoad = new MenuButton("Load");
    _scatterLoad->setCallback(this);
    _scatterMenu->addItem(_scatterLoad);

    _scatterLoadAll = new MenuButton("Load All");
    _scatterLoadAll->setCallback(this);
    _scatterMenu->addItem(_scatterLoadAll);

    _scatterMenu->addItem(_scatterFilterMenu);
    
    _scatterPhenotypes = new MenuList();
    _scatterPhenotypes->setCallback(this);
    _scatterPhenotypes->setScrollingHint(MenuList::ONE_TO_ONE);
    _scatterMenu->addItem(_scatterPhenotypes);

    pheno.clear();
    pheno.push_back("Smarr");
    pheno.push_back("Crohns");
    pheno.push_back("UC");
    pheno.push_back("Healthy");

    _scatterPhenotypes->setValues(pheno);

    _scatterSecondPhenotype = new MenuList();
    _scatterSecondPhenotype->setCallback(this);
    _scatterSecondPhenotype->setScrollingHint(MenuList::ONE_TO_ONE);
    _scatterMenu->addItem(_scatterSecondPhenotype);

    pheno.clear();
    pheno.push_back("Smarr");
    pheno.push_back("Crohns");
    pheno.push_back("UC");
    pheno.push_back("Healthy");

    _scatterSecondPhenotype->setValues(pheno);

    _scatterSortResults = new MenuRangeValueCompact("Num Results",2,7,5);
    _scatterMenu->addItem(_scatterSortResults);

    _scatterLoadFilter = new MenuButton("Load w/ Filter");
    _scatterLoadFilter->setCallback(this);
    _scatterMenu->addItem(_scatterLoadFilter);

    bool cacheEnabled =  ConfigManager::getBool("value","Plugin.FuturePatient.Cache",true);

    _dbCache = new MenuCheckbox("DB Cache",cacheEnabled);
    _dbCache->setCallback(this);
    _fpMenu->addItem(_dbCache);

    _removeAllButton = new MenuButton("Remove All");
    _removeAllButton->setCallback(this);
    _fpMenu->addItem(_removeAllButton);

    _closeLayoutButton = new MenuButton("Close Layout");
    _closeLayoutButton->setCallback(this);

    _ssButton = new MenuButton("Sub shot");
    _ssButton->setCallback(this);
    //_fpMenu->addItem(_ssButton);

    PluginHelper::addRootMenuItem(_fpMenu);

    struct listField
    {
	char entry[256];
	int id;
    };

    struct listField * lfList = NULL;
    int listEntries = 0;
    int * sizes = NULL;
    listField ** groupLists = NULL;

    if(ComController::instance()->isMaster())
    {
	/*if(!_conn)
	{
	    _conn = new mysqlpp::Connection(false);
	    //if(!_conn->connect("futurepatient","palmsdev2.ucsd.edu","fpuser","FPp@ssw0rd"))
	    if(!_conn->connect("futurepatient","fiona-01.ucsd.edu","fpuser","p@ti3nt"))
	    {
		std::cerr << "Unable to connect to database." << std::endl;
		delete _conn;
		_conn = NULL;
	    }
	}*/

	if(!_dbm)
	{
	    std::string cacheDir = ConfigManager::getEntry("value","Plugin.FuturePatient.CacheDir","");
	    _dbm = new DBManager("futurepatient","fiona-01.ucsd.edu","fpuser","p@ti3nt",cacheDir,cacheEnabled);
	}

	if(_dbm)
	{
	    DBMQueryResult result;
	    _dbm->runQuery("select distinct Measurement.patient_id, Patient.last_name as name from Measurement inner join Patient on Measurement.patient_id = Patient.patient_id order by patient_id;",result);

	    listEntries = result.numRows();

	    if(listEntries)
	    {
		lfList = new struct listField[listEntries];
		sizes = new int[listEntries];
		groupLists = new listField*[listEntries];

		for(int i = 0; i < listEntries; i++)
		{
		    strncpy(lfList[i].entry,result(i,"name").c_str(),255);
		}

		for(int i = 0; i < listEntries; ++i)
		{
		    std::stringstream groupss;
		    groupss << "select distinct Measure.name from Measure inner join Measurement on Measure.measure_id = Measurement.measure_id and Measurement.patient_id = \"" << result(i,"patient_id") << "\" order by Measure.name;";

		    DBMQueryResult resultg;
		    _dbm->runQuery(groupss.str(),resultg);

		    sizes[i] = resultg.numRows();
		    if(resultg.numRows())
		    {
			groupLists[i] = new listField[resultg.numRows()];
		    }
		    else
		    {
			groupLists[i] = NULL;
		    }

		    for(int j = 0; j < resultg.numRows(); j++)
		    {
			strncpy(groupLists[i][j].entry,resultg(j,"name").c_str(),255);
		    }
		}
	    }
	}

	/*if(_conn)
	{
	    mysqlpp::Query q = _conn->query("select distinct Measurement.patient_id, Patient.last_name as name from Measurement inner join Patient on Measurement.patient_id = Patient.patient_id order by patient_id;");
	    mysqlpp::StoreQueryResult res = q.store();

	    listEntries = res.num_rows();

	    if(listEntries)
	    {
		lfList = new struct listField[listEntries];
		sizes = new int[listEntries];
		groupLists = new listField*[listEntries];

		for(int i = 0; i < listEntries; i++)
		{
		    strncpy(lfList[i].entry,res[i]["name"].c_str(),255);
		}

		for(int i = 0; i < listEntries; ++i)
		{
		    std::stringstream groupss;
		    groupss << "select distinct Measure.name from Measure inner join Measurement on Measure.measure_id = Measurement.measure_id and Measurement.patient_id = \"" << res[i]["patient_id"].c_str() << "\" order by Measure.name;";

		    mysqlpp::Query groupq = _conn->query(groupss.str().c_str());
		    mysqlpp::StoreQueryResult groupRes = groupq.store();

		    sizes[i] = groupRes.num_rows();
		    if(groupRes.num_rows())
		    {
			groupLists[i] = new listField[groupRes.num_rows()];
		    }
		    else
		    {
			groupLists[i] = NULL;
		    }

		    for(int j = 0; j < groupRes.num_rows(); j++)
		    {
			strncpy(groupLists[i][j].entry,groupRes[j]["name"].c_str(),255);
		    }
		}
	    }
	}*/

	ComController::instance()->sendSlaves(&listEntries,sizeof(int));
	if(listEntries)
	{
	    ComController::instance()->sendSlaves(lfList,sizeof(struct listField)*listEntries);
	    ComController::instance()->sendSlaves(sizes,sizeof(int)*listEntries);
	    for(int i = 0; i < listEntries; i++)
	    {
		if(sizes[i])
		{
		    ComController::instance()->sendSlaves(groupLists[i],sizeof(struct listField)*sizes[i]);
		}
	    }
	}
    }
    else
    {
	ComController::instance()->readMaster(&listEntries,sizeof(int));
	if(listEntries)
	{
	    lfList = new struct listField[listEntries];
	    ComController::instance()->readMaster(lfList,sizeof(struct listField)*listEntries);
	    sizes = new int[listEntries];
	    ComController::instance()->readMaster(sizes,sizeof(int)*listEntries);
	    groupLists = new listField*[listEntries];
	    for(int i = 0; i < listEntries; i++)
	    {
		if(sizes[i])
		{
		    groupLists[i] = new listField[sizes[i]];
		    ComController::instance()->readMaster(groupLists[i],sizeof(struct listField)*sizes[i]);
		}
		else
		{
		    groupLists[i] = NULL;
		}
	    }
	}
    }

    std::vector<std::string> stringlist;
    for(int i = 0; i < listEntries; i++)
    {
	stringlist.push_back(lfList[i].entry);

	_patientTestMap[lfList[i].entry] = std::vector<std::string>();
	for(int j = 0; j < sizes[i]; j++)
	{
	    _patientTestMap[lfList[i].entry].push_back(groupLists[i][j].entry);
	}
    }

    _chartPatientList->setValues(stringlist);

    if(_chartPatientList->getListSize())
    {
	_testList->setValues(_patientTestMap[_chartPatientList->getValue()]);
    }

    if(lfList)
    {
	delete[] lfList;
    }

    for(int i = 0; i < listEntries; i++)
    {
	if(groupLists[i])
	{
	    delete[] groupLists[i];
	}
    }

    if(listEntries)
    {
	delete[] sizes;
	delete[] groupLists;
    }

    _groupList = new MenuList();
    _groupList->setCallback(this);
    _groupLoadMenu->addItem(_groupList);

    _groupLoadButton = new MenuButton("Load");
    _groupLoadButton->setCallback(this);
    _groupLoadMenu->addItem(_groupLoadButton);

    lfList = NULL;
    listEntries = 0;
    sizes = NULL;
    groupLists = NULL;

    if(ComController::instance()->isMaster())
    {
	if(_dbm)
	{
	    DBMQueryResult result;

	    _dbm->runQuery("select display_name from Measure_Type order by display_name;",result);

	    listEntries = result.numRows();

	    if(listEntries)
	    {
		lfList = new struct listField[listEntries];

		for(int i = 0; i < listEntries; i++)
		{
		    strncpy(lfList[i].entry,result(i,"display_name").c_str(),255);
		}
	    }

	    if(listEntries)
	    {
		sizes = new int[listEntries];
		groupLists = new listField*[listEntries];

		for(int i = 0; i < listEntries; i++)
		{
		    std::stringstream groupss;
		    groupss << "select Measure.name from Measure inner join Measure_Type on Measure_Type.measure_type_id = Measure.measure_type_id where Measure_Type.display_name = \"" << result(i,"display_name") << "\";";

		    DBMQueryResult resultg;

		    _dbm->runQuery(groupss.str(),resultg);

		    sizes[i] = resultg.numRows();
		    if(resultg.numRows())
		    {
			groupLists[i] = new listField[resultg.numRows()];
		    }
		    else
		    {
			groupLists[i] = NULL;
		    }

		    for(int j = 0; j < resultg.numRows(); j++)
		    {
			strncpy(groupLists[i][j].entry,resultg(j,"name").c_str(),255);
		    }
		}
	    }
	}

	/*if(_conn)
	{
	    mysqlpp::Query q = _conn->query("select display_name from Measure_Type order by display_name;");
	    mysqlpp::StoreQueryResult res = q.store();

	    listEntries = res.num_rows();

	    if(listEntries)
	    {
		lfList = new struct listField[listEntries];

		for(int i = 0; i < listEntries; i++)
		{
		    strncpy(lfList[i].entry,res[i]["display_name"].c_str(),255);
		}
	    }

	    if(listEntries)
	    {
		sizes = new int[listEntries];
		groupLists = new listField*[listEntries];

		for(int i = 0; i < listEntries; i++)
		{
		    std::stringstream groupss;
		    groupss << "select Measure.name from Measure inner join Measure_Type on Measure_Type.measure_type_id = Measure.measure_type_id where Measure_Type.display_name = \"" << res[i]["display_name"].c_str() << "\";";

		    mysqlpp::Query groupq = _conn->query(groupss.str().c_str());
		    mysqlpp::StoreQueryResult groupRes = groupq.store();

		    sizes[i] = groupRes.num_rows();
		    if(groupRes.num_rows())
		    {
			groupLists[i] = new listField[groupRes.num_rows()];
		    }
		    else
		    {
			groupLists[i] = NULL;
		    }

		    for(int j = 0; j < groupRes.num_rows(); j++)
		    {
			strncpy(groupLists[i][j].entry,groupRes[j]["name"].c_str(),255);
		    }
		}
	    }
	}*/

	ComController::instance()->sendSlaves(&listEntries,sizeof(int));
	if(listEntries)
	{
	    ComController::instance()->sendSlaves(lfList,sizeof(struct listField)*listEntries);
	    ComController::instance()->sendSlaves(sizes,sizeof(int)*listEntries);
	    for(int i = 0; i < listEntries; i++)
	    {
		if(sizes[i])
		{
		    ComController::instance()->sendSlaves(groupLists[i],sizeof(struct listField)*sizes[i]);
		}
	    }
	}
    }
    else
    {
	ComController::instance()->readMaster(&listEntries,sizeof(int));
	if(listEntries)
	{
	    lfList = new struct listField[listEntries];
	    ComController::instance()->readMaster(lfList,sizeof(struct listField)*listEntries);
	    sizes = new int[listEntries];
	    ComController::instance()->readMaster(sizes,sizeof(int)*listEntries);
	    groupLists = new listField*[listEntries];
	    for(int i = 0; i < listEntries; i++)
	    {
		if(sizes[i])
		{
		    groupLists[i] = new listField[sizes[i]];
		    ComController::instance()->readMaster(groupLists[i],sizeof(struct listField)*sizes[i]);
		}
		else
		{
		    groupLists[i] = NULL;
		}
	    }
	}
    }

    stringlist.clear();
    for(int i = 0; i < listEntries; i++)
    {
	stringlist.push_back(lfList[i].entry);

	_groupTestMap[lfList[i].entry] = std::vector<std::string>();
	for(int j = 0; j < sizes[i]; j++)
	{
	    _groupTestMap[lfList[i].entry].push_back(groupLists[i][j].entry);
	}
    }

    _groupList->setValues(stringlist);

    if(lfList)
    {
	delete[] lfList;
    }

    for(int i = 0; i < listEntries; i++)
    {
	if(groupLists[i])
	{
	    delete[] groupLists[i];
	}
    }

    if(listEntries)
    {
	delete[] sizes;
	delete[] groupLists;
    }

    setupMicrobes();
    setupMicrobePatients();
    setupStrainMenu();
    
    if(_microbePatients->getListSize())
    {
	updateMicrobeTests(_microbePatients->getIndex() + 1);
    }

    lfList = NULL;
    listEntries = 0;

    if(ComController::instance()->isMaster())
    {
	if(_dbm)
	{
	    DBMQueryResult result;

	    _dbm->runQuery("select distinct name, group_id, type from Event where patient_id = \"1\" and group_id is not NULL order by type desc, name;",result);

	    listEntries = result.numRows();

	    if(listEntries)
	    {
		lfList = new struct listField[listEntries];

		for(int i = 0; i < listEntries; i++)
		{
		    strncpy(lfList[i].entry,result(i,"name").c_str(),255);
		    lfList[i].id = atoi(result(i,"group_id").c_str());
		}
	    }
	}

	ComController::instance()->sendSlaves(&listEntries,sizeof(int));
	if(listEntries)
	{
	    ComController::instance()->sendSlaves(lfList,sizeof(struct listField)*listEntries);
	}

	/*if(_conn)
	{
	    mysqlpp::Query q = _conn->query("select distinct name, type from Event where patient_id = \"1\" order by type desc, name;");
	    mysqlpp::StoreQueryResult res = q.store();

	    listEntries = res.num_rows();

	    if(listEntries)
	    {
		lfList = new struct listField[listEntries];

		for(int i = 0; i < listEntries; i++)
		{
		    strncpy(lfList[i].entry,res[i]["name"].c_str(),255);
		}
	    }
	}

	ComController::instance()->sendSlaves(&listEntries,sizeof(int));
	if(listEntries)
	{
	    ComController::instance()->sendSlaves(lfList,sizeof(struct listField)*listEntries);
	}*/
    }
    else
    {
	ComController::instance()->readMaster(&listEntries,sizeof(int));
	if(listEntries)
	{
	    lfList = new struct listField[listEntries];
	    ComController::instance()->readMaster(lfList,sizeof(struct listField)*listEntries);
	}
    }

    stringlist.clear();
    for(int i = 0; i < listEntries; i++)
    {
	//std::cerr << lfList[i].entry << " " << lfList[i].id << std::endl;
	stringlist.push_back(lfList[i].entry);
	_eventGroups.push_back(lfList[i].id);
    }

    _eventName->setValues(stringlist);

    if(lfList)
    {
	delete[] lfList;
    }

    _layoutDirectory = ConfigManager::getEntry("value","Plugin.FuturePatient.LayoutDir","");

    lfList = NULL;
    listEntries = 0;

    if(ComController::instance()->isMaster())
    {
	DIR * dir;

	if ((dir = opendir(_layoutDirectory.c_str())) == NULL)
	{
	    std::cerr << "Unable to open directory: " << _layoutDirectory << std::endl;
	}
	else
	{
	    dirent * entry;
	    struct stat st;
	    while ((entry = readdir(dir)) != NULL)
	    {
		std::string fullPath(_layoutDirectory + "/" + entry->d_name);
		stat(fullPath.c_str(), &st);
		if(!S_ISDIR(st.st_mode))
		{
		    listEntries++;
		}
	    }

	    if(listEntries)
	    {
		lfList = new listField[listEntries];
		int listIndex = 0;
		rewinddir(dir);
		while ((entry = readdir(dir)) != NULL)
		{
		    std::string fullPath(_layoutDirectory + "/" + entry->d_name);
		    stat(fullPath.c_str(), &st);
		    if(!S_ISDIR(st.st_mode))
		    {
			strncpy(lfList[listIndex].entry,entry->d_name,255);
			listIndex++;
		    }
		}
	    }
	}
	ComController::instance()->sendSlaves(&listEntries,sizeof(int));
	if(listEntries)
	{
	    ComController::instance()->sendSlaves(lfList,listEntries*sizeof(struct listField));
	}
    }
    else
    {
	ComController::instance()->readMaster(&listEntries,sizeof(int));
	if(listEntries)
	{
	    lfList = new listField[listEntries];
	    ComController::instance()->readMaster(lfList,listEntries*sizeof(struct listField));
	}
    }
    
    for(int i = 0; i < listEntries; ++i)
    {
	MenuButton * tempb = new MenuButton(lfList[i].entry);
	tempb->setCallback(this);
	_loadLayoutMenu->addItem(tempb);
	_loadLayoutButtons.push_back(tempb);
    }

    if(lfList)
    {
	delete[] lfList;
    }

    lfList = NULL;
    listEntries = 0;

    if(ComController::instance()->isMaster())
    {
	if(_dbm)
	{
	    DBMQueryResult result;

	    _dbm->runQuery("select distinct phylum from Microbes order by phylum;",result);

	    listEntries = result.numRows();

	    if(listEntries)
	    {
		lfList = new struct listField[listEntries];

		for(int i = 0; i < listEntries; i++)
		{
		    strncpy(lfList[i].entry,result(i,"phylum").c_str(),255);
		}
	    }
	}

	/*if(_conn)
	{
	    mysqlpp::Query q = _conn->query("select distinct phylum from Microbes order by phylum;");
	    mysqlpp::StoreQueryResult res = q.store();

	    listEntries = res.num_rows();

	    if(listEntries)
	    {
		lfList = new struct listField[listEntries];

		for(int i = 0; i < listEntries; i++)
		{
		    strncpy(lfList[i].entry,res[i]["phylum"].c_str(),255);
		}
	    }
	}*/

	ComController::instance()->sendSlaves(&listEntries,sizeof(int));
	if(listEntries)
	{
	    ComController::instance()->sendSlaves(lfList,sizeof(struct listField)*listEntries);
	}
    }
    else
    {
	ComController::instance()->readMaster(&listEntries,sizeof(int));
	if(listEntries)
	{
	    lfList = new struct listField[listEntries];
	    ComController::instance()->readMaster(lfList,sizeof(struct listField)*listEntries);
	}
    }

    stringlist.clear();
    for(int i = 0; i < listEntries; i++)
    {
	stringlist.push_back(lfList[i].entry);
    }

    _scatterPhylumList = stringlist;
    _scatterFirstList->setValues(stringlist);
    _scatterSecondList->setValues(stringlist);
    _scatterFirstEntry->setSearchList(_scatterPhylumList,5);
    _scatterSecondEntry->setSearchList(_scatterPhylumList,5);

    if(lfList)
    {
	delete[] lfList;
    } 

    if(ComController::instance()->isMaster())
    {
	_subImage = new osg::Image();
	_subImage->allocateImage(1920,1080,GL_RGBA,GL_RGBA,GL_FLOAT);
	_subImage->setInternalTextureFormat(4);

	_subDepthTex = new osg::Texture2D();
	_subDepthTex->setTextureSize(1920,1080);
	_subDepthTex->setInternalFormat(GL_DEPTH_COMPONENT);
	_subDepthTex->setFilter(osg::Texture2D::MIN_FILTER,osg::Texture2D::NEAREST);
	_subDepthTex->setFilter(osg::Texture2D::MAG_FILTER,osg::Texture2D::NEAREST);
	_subDepthTex->setResizeNonPowerOfTwoHint(false);
	_subDepthTex->setUseHardwareMipMapGeneration(false);


	_subCamera = new osg::Camera();
	_subCamera->setAllowEventFocus(false);
	_subCamera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	_subCamera->setClearColor(osg::Vec4(1.0,0,0,1.0));
	_subCamera->setRenderOrder(osg::Camera::PRE_RENDER);
	_subCamera->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);
	_subCamera->attach(osg::Camera::COLOR_BUFFER0, _subImage, 0, 0);
	_subCamera->attach(osg::Camera::DEPTH_BUFFER,_subDepthTex);
	_subCamera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);

	_subCamera->addChild((osg::Node*)SceneManager::instance()->getScene());
    }

    return true;
}

void FuturePatient::preFrame()
{
    int numCommands = 0;
    int * commands = NULL;
    if(ComController::instance()->isMaster())
    {
        if(_mls)
        {
            CVRSocket * con;
            if((con = _mls->accept()))
            {
                std::cerr << "Adding socket." << std::endl;
                con->setNoDelay(true);
                _socketList.push_back(con);
            }
        }

        std::vector<int> messageList;
        checkSockets(messageList);

        numCommands = messageList.size();

        ComController::instance()->sendSlaves(&numCommands, sizeof(int));

        if(numCommands)
        {
            commands = new int[numCommands];
            for(int i = 0; i < numCommands; i++)
            {
                commands[i] = messageList[i];
            }
            ComController::instance()->sendSlaves(commands,numCommands * sizeof(int));
        }
    }
    else
    {
        ComController::instance()->readMaster(&numCommands, sizeof(int));
        if(numCommands)
        {
            commands = new int[numCommands];
            ComController::instance()->readMaster(commands,numCommands * sizeof(int));
        }
    }

   if(numCommands)
    {
        std::stringstream filess;
        filess << "Preset" << commands[numCommands-1] << ".cfg";
        std::string file = filess.str();

        bool loaded = false;
        for(int i = 0; i < _loadLayoutButtons.size(); ++i)
        {
            if(_loadLayoutButtons[i]->getText() == file)
            {
                loaded = true;
                menuCallback(_loadLayoutButtons[i]);
                break;
            }
        }

        if(!loaded)
        {
            std::cerr << "Unable to find preset config: " << file << std::endl;
        }

        delete[] commands;
    }

    if(_layoutObject)
    {
	_layoutObject->perFrame();
    }

    if(_takeSubImage && !_subImageDone)
    {
	_subImageDone = true;
    }
    else if(_takeSubImage && _subImageDone)
    {
	_takeSubImage = false;
	_subImageDone = false;
	dynamic_cast<osg::Group*>(CVRViewer::instance()->getSceneData())->removeChild(_subCamera);
	//SceneManager::instance()->getScene()->removeChild(_subCamera);

	osgDB::writeImageFile(*_subImage.get(),"/home/aprudhom/testImage.tif");
    }
}

void FuturePatient::checkSockets(std::vector<int> & messageList)
{
    if(!_socketList.size())
    {
        return;
    }

    int maxfd = 0;

    fd_set socketsetR;
    FD_ZERO(&socketsetR);

    for(int i = 0; i < _socketList.size(); i++)
    {
        FD_SET((unsigned int)_socketList[i]->getSocketFD(),&socketsetR);
        if(_socketList[i]->getSocketFD() > maxfd)
        {
            maxfd = _socketList[i]->getSocketFD();
        }
    }

    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 0;

    select(maxfd+1,&socketsetR,NULL,NULL,&tv);

    for(std::vector<CVRSocket*>::iterator it = _socketList.begin(); it != _socketList.end(); )
    {
        if(FD_ISSET((*it)->getSocketFD(),&socketsetR))
        {
            if(!processSocketInput(*it,messageList))
            {
                std::cerr << "Removing socket." << std::endl;
                delete *it;
                it = _socketList.erase(it);
            }
            else
            {
                it++;
            }
        }
        else
        {
            it++;
        }
    }
}

bool FuturePatient::processSocketInput(CVRSocket * socket, std::vector<int> & messageList)
{
    /*char c;
    if(!socket->recv(&c,sizeof(char)))
    {
        return false;
    }

    std::cerr << "Char: " << (int)c << std::endl;*/
    int i;
    if(!socket->recv(&i,sizeof(int)))
    {
        return false;
    }

    //std::cerr << "int: " << i << std::endl;
    messageList.push_back(i);

    char resp[1024];
    memset(resp,'\0',1024);
    resp[0] = 'o';
    resp[1] = 'k';
    socket->send(resp,1024);

    return true;
}

void FuturePatient::menuCallback(MenuItem * item)
{
    if(item == _ssButton)
    {
	takeSubImage();
    }

    if(item == _loadButton)
    {
	loadGraph(_chartPatientList->getValue(),_testList->getValue());
    }

    if(item == _linearRegCB)
    {
	if(_layoutObject)
	{
	    _layoutObject->setChartLinearRegression(_linearRegCB->getValue());
	}
    }

    if(item == _groupLoadButton)
    {
	for(int i = 0; i < _groupTestMap[_groupList->getValue()].size(); i++)
	{
	    loadGraph("Smarr",_groupTestMap[_groupList->getValue()][i]);
	}
    }

    if(item == _chartPatientList)
    {
	if(_chartPatientList->getListSize())
	{
	    _testList->setValues(_patientTestMap[_chartPatientList->getValue()]);
	}
    }

    if(item == _inflammationButton)
    {
	checkLayout();

	menuCallback(_removeAllButton);

	if(_multiAddCB->getValue())
	{
	    _multiAddCB->setValue(false);
	    menuCallback(_multiAddCB);
	}

	_layoutObject->setSyncTime(false);
	loadGraph("Smarr","hs-CRP");
	loadGraph("Smarr","SIgA");
	loadGraph("Smarr","Lysozyme");
	loadGraph("Smarr","Lactoferrin");
	_layoutObject->setSyncTime(true);
	_layoutObject->setRows(4.0);
    }

    if(item == _big4MultiButton)
    {
	checkLayout();

	menuCallback(_removeAllButton);

	if(!_multiAddCB->getValue())
	{
	    _multiAddCB->setValue(true);
	    menuCallback(_multiAddCB);
	}

	_layoutObject->setSyncTime(false);
	loadGraph("Smarr","hs-CRP");
	loadGraph("Smarr","SIgA");
	loadGraph("Smarr","Lysozyme");
	loadGraph("Smarr","Lactoferrin");
	_layoutObject->setSyncTime(true);
	_layoutObject->setRows(4.0);
    }

    if(item == _cholesterolButton)
    {
	loadGraph("Smarr","Total Cholesterol");
	loadGraph("Smarr","LDL");
	loadGraph("Smarr","HDL");
	loadGraph("Smarr","TG");
	loadGraph("Smarr","TG/HDL");
	loadGraph("Smarr","Total LDL3+LDL-4");
    }

    if(item == _insGluButton)
    {
	loadGraph("Smarr","Fasting Glucose");
	loadGraph("Smarr","Insulin");
	loadGraph("Smarr","Hemoglobin a1c");
	loadGraph("Smarr","Homocysteine");
	loadGraph("Smarr","Vitamin D, 25-Hydroxy");
    }

    if(item == _inflammationImmuneButton)
    {
	loadGraph("Smarr","hs-CRP");
	loadGraph("Smarr","Lysozyme");
	loadGraph("Smarr","SIgA");
	loadGraph("Smarr","Lactoferrin");
	loadGraph("Smarr","Calprotectin");
	loadGraph("Smarr","WBC-");
	loadGraph("Smarr","NEU %");
    }

    if(item == _multiAddCB)
    {
	if(_multiObject)
	{
	    if(!_multiObject->getNumGraphs())
	    {
		delete _multiObject;
	    }
	    _multiObject = NULL;
	}
    }

    if(item == _loadAll)
    {
	GraphGlobals::setDeferUpdate(true);
	for(int i = 0; i < _patientTestMap["Smarr"].size(); i++)
	{
	    loadGraph("Smarr",_patientTestMap["Smarr"][i],_loadAllAvg->getValue());
	}
	GraphGlobals::setDeferUpdate(false);
	if(_layoutObject)
	{
	    _layoutObject->forceUpdate();
	    _layoutObject->setRows(7.0);
	    _layoutObject->setSyncTime(true);
	    _layoutObject->setAllGraphMinTime();
	}
    }

    if(item == _dbCache)
    {
	if(_dbm)
	{
	    _dbm->setUseCache(_dbCache->getValue());
	}
	return;
    }

    if(item == _removeAllButton)
    {
	if(_layoutObject)
	{
	    _layoutObject->removeAll();
	}
	if(_currentSBGraph && _microbeGraphType->getIndex() == 1)
	{
	    _currentSBGraph = NULL;
	    _microbeMenu->removeItem(_microbeDone);
	}
	if(_currentVBGraph && _microbeGraphType->getIndex() == 2)
	{
	    _currentVBGraph = NULL;
	    _microbeMenu->removeItem(_microbeDone);
	}

	if(_currentSymptomGraph)
	{
	    _currentSymptomGraph = NULL;
	    _eventMenu->removeItem(_eventDone);
	}

	menuCallback(_multiAddCB);
    }

    if(item == _closeLayoutButton)
    {
	menuCallback(_removeAllButton);

	if(_layoutObject)
	{
	    _layoutObject->detachFromScene();
	    PluginHelper::unregisterSceneObject(_layoutObject);

	    delete _layoutObject;
	    _layoutObject = NULL;
	}
	_fpMenu->removeItem(_closeLayoutButton);
    }

    if(item == _graphTypeList)
    {
	if(_layoutObject)
	{
	    _layoutObject->setChartDisplayType((GraphDisplayType)_graphTypeList->getIndex());
	}
	_graphTypeList->setIndex(0);
    }

    if(item == _microbeTable)
    {
	if(_microbePatients->getListSize())
	{
	    updateMicrobeTests(_microbePatients->getIndex() + 1);
	}

	menuCallback(_scatterMicrobeType);

	/*if(_microbeTable->getIndex() == 0)
	{
	    _sMicrobes->setValues(_microbeList);
	}
	else if(_microbeTable->getIndex() == 1)
	{
	    _sMicrobes->setValues(_microbeV2List);
	}*/
	if(_sMicrobeType->getValue() == "Microbe")
	{
	    _sMicrobes->setValues(_microbeTableList[_microbeTable->getIndex()]->microbeList);
	    _sMicrobeEntry->setSearchList(_microbeTableList[_microbeTable->getIndex()]->microbeList,5);
	}
	else if(_sMicrobeType->getValue() == "Family")
	{
	    _sMicrobes->setValues(_microbeTableList[_microbeTable->getIndex()]->familyList);
	    _sMicrobeEntry->setSearchList(_microbeTableList[_microbeTable->getIndex()]->familyList,5);
	}
	else if(_sMicrobeType->getValue() == "Genus")
	{
	    _sMicrobes->setValues(_microbeTableList[_microbeTable->getIndex()]->genusList);
	    _sMicrobeEntry->setSearchList(_microbeTableList[_microbeTable->getIndex()]->genusList,5);
	}
	else
	{
	    _sMicrobes->setValues(_scatterPhylumList);
	    _sMicrobeEntry->setSearchList(_scatterPhylumList,5);
	}
    }

    if(item == _scatterMicrobeType)
    {
	if(_scatterMicrobeType->getValue() == "Species")
	{
	    _scatterFirstList->setValues(_microbeTableList[_microbeTable->getIndex()]->microbeList);
	    _scatterSecondList->setValues(_microbeTableList[_microbeTable->getIndex()]->microbeList);
	    _scatterFirstEntry->setSearchList(_microbeTableList[_microbeTable->getIndex()]->microbeList,5);
	    _scatterSecondEntry->setSearchList(_microbeTableList[_microbeTable->getIndex()]->microbeList,5);
	}
	else if(_scatterMicrobeType->getValue() == "Family")
	{
	    _scatterFirstList->setValues(_microbeTableList[_microbeTable->getIndex()]->familyList);
	    _scatterSecondList->setValues(_microbeTableList[_microbeTable->getIndex()]->familyList);
	    _scatterFirstEntry->setSearchList(_microbeTableList[_microbeTable->getIndex()]->familyList,5);
	    _scatterSecondEntry->setSearchList(_microbeTableList[_microbeTable->getIndex()]->familyList,5);
	}
	else if(_scatterMicrobeType->getValue() == "Genus")
	{
	    _scatterFirstList->setValues(_microbeTableList[_microbeTable->getIndex()]->genusList);
	    _scatterSecondList->setValues(_microbeTableList[_microbeTable->getIndex()]->genusList);
	    _scatterFirstEntry->setSearchList(_microbeTableList[_microbeTable->getIndex()]->genusList,5);
	    _scatterSecondEntry->setSearchList(_microbeTableList[_microbeTable->getIndex()]->genusList,5);
	}
	else
	{
	    _scatterFirstList->setValues(_scatterPhylumList);
	    _scatterSecondList->setValues(_scatterPhylumList);
	    _scatterFirstEntry->setSearchList(_scatterPhylumList,5);
	    _scatterSecondEntry->setSearchList(_scatterPhylumList,5);
	}
    }

    if(item == _sMicrobeType)
    {
	if(_sMicrobeType->getValue() == "Microbe")
	{
	    _sMicrobes->setValues(_microbeTableList[_microbeTable->getIndex()]->microbeList);
	    _sMicrobeEntry->setSearchList(_microbeTableList[_microbeTable->getIndex()]->microbeList,5);
	}
	else if(_sMicrobeType->getValue() == "Family")
	{
	    _sMicrobes->setValues(_microbeTableList[_microbeTable->getIndex()]->familyList);
	    _sMicrobeEntry->setSearchList(_microbeTableList[_microbeTable->getIndex()]->familyList,5);
	}
	else if(_sMicrobeType->getValue() == "Genus")
	{
	    _sMicrobes->setValues(_microbeTableList[_microbeTable->getIndex()]->genusList);
	    _sMicrobeEntry->setSearchList(_microbeTableList[_microbeTable->getIndex()]->genusList,5);
	}
	else
	{
	    _sMicrobes->setValues(_scatterPhylumList);
	    _sMicrobeEntry->setSearchList(_scatterPhylumList,5);
	}
    }

    if(item == _microbePatients)
    {
	if(_microbePatients->getListSize())
	{
	    updateMicrobeTests(_microbePatients->getIndex() + 1);
	}
    }

    if(item == _microbeLoad && _microbePatients->getListSize() && _microbeTest->getListSize())
    {
	/*std::string tablesuffix;
	if(_microbeTable->getIndex() == 1)
	{
	    tablesuffix = "_V2";
	}*/
	// Bar Graph
	if(_microbeGraphType->getIndex() == 0)
	{
	    MicrobeGraphObject * mgo = new MicrobeGraphObject(_dbm, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);
	    if(mgo->setGraph(_microbePatients->getValue(), _microbePatients->getIndex()+1, _microbeTest->getValue(), _microbeTestTime[_microbeTest->getIndex()],(int)_microbeNumBars->getValue(),_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix,_microbeGrouping->getValue(),_microbeOrdering->getValue(),(MicrobeGraphType)(_microbeLevel->getIndex())))
	    {
		checkLayout();
		_layoutObject->addGraphObject(mgo);
	    }
	    else
	    {
		delete mgo;
	    }
	}
	// Stacked Bar Graph
	else if(_microbeGraphType->getIndex() == 1)
	{
	    if(!_currentSBGraph)
	    {
		_currentSBGraph = new MicrobeBarGraphObject(_dbm, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);
		_currentSBGraph->addGraph(_microbePatients->getValue(), _microbePatients->getIndex()+1, _microbeTest->getValue(),_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix);
		checkLayout();
		_layoutObject->addGraphObject(_currentSBGraph);
		_microbeMenu->addItem(_microbeDone);
	    }
	    else
	    {
		_currentSBGraph->addGraph(_microbePatients->getValue(), _microbePatients->getIndex()+1, _microbeTest->getValue(),_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix);
	    }
	}
	else if(_microbeGraphType->getIndex() == 2)
	{
	    if(!_currentVBGraph)
	    {
		_currentVBGraph = new MicrobeVerticalBarGraphObject(_dbm, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);

		switch((MicrobeGraphType)(_microbeLevel->getIndex()))
		{
		    case MGT_PHYLUM:
		    {
			_currentVBGraph->setNameList(_scatterPhylumList);
			break;
		    }
		    case MGT_FAMILY:
		    {
			_currentVBGraph->setNameList(_microbeTableList[_microbeTable->getIndex()]->familyListPO);
			break;
		    }
		    case MGT_GENUS:
		    {
			_currentVBGraph->setNameList(_microbeTableList[_microbeTable->getIndex()]->genusListPO);
			break;
		    }
		    default:
		    {
			_currentVBGraph->setNameList(_microbeTableList[_microbeTable->getIndex()]->microbeListPO);
			break;
		    }
		}
		_currentVBGraph->setGroupList(_scatterPhylumList);

		if(_currentVBGraph->addGraph(_microbePatients->getValue(), _microbePatients->getIndex()+1, _microbeTest->getValue(),_microbeTestTime[_microbeTest->getIndex()],_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix,(MicrobeGraphType)(_microbeLevel->getIndex())))
		{
		    checkLayout();
		    _layoutObject->addGraphObject(_currentVBGraph);
		    _microbeMenu->addItem(_microbeDone);
		}
		else
		{
		    delete _currentVBGraph;
		    _currentVBGraph = NULL;
		}
	    }
	    else
	    {
		_currentVBGraph->addGraph(_microbePatients->getValue(), _microbePatients->getIndex()+1, _microbeTest->getValue(),_microbeTestTime[_microbeTest->getIndex()],_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix,(MicrobeGraphType)(_microbeLevel->getIndex()));
	    }
	}
	// Graph with All
	else if(_microbeGraphType->getIndex() == 3)
	{
	    MicrobeGraphWAllObject * mgo = new MicrobeGraphWAllObject(_dbm, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);
	    if(mgo->setGraph(_microbePatients->getValue(), _microbePatients->getIndex()+1, _microbeTest->getValue(), _microbeTestTime[_microbeTest->getIndex()],(int)_microbeNumBars->getValue(),_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix,_microbeGrouping->getValue(),_microbeOrdering->getValue(),(MicrobeGraphType)(_microbeLevel->getIndex())))
	    {
		checkLayout();
		_layoutObject->addGraphObject(mgo);
	    }
	    else
	    {
		delete mgo;
	    }
	}
    }

    if(item == _microbeGraphType)
    {
	if(_currentSBGraph && _microbeGraphType->getIndex() != 1)
	{
	    _currentSBGraph = NULL;
	    _microbeMenu->removeItem(_microbeDone);
	}
	if(_currentVBGraph && _microbeGraphType->getIndex() != 2)
	{
	    _currentVBGraph = NULL;
	    _microbeMenu->removeItem(_microbeDone);
	}
	return;
    }

    if(item == _microbeDone)
    {
	// Stacked Bar Graph
	if(_microbeGraphType->getIndex() == 1)
	{
	    _currentSBGraph = NULL;
	    _microbeMenu->removeItem(_microbeDone);
	}
	if(_microbeGraphType->getIndex() == 2)
	{
	    _currentVBGraph = NULL;
	    _microbeMenu->removeItem(_microbeDone);
	}
	return;
    }

    if(item == _microbeLoadCrohnsAll)
    {
	/*std::string tablesuffix;
	if(_microbeTable->getIndex() == 1)
	{
	    tablesuffix = "_V2";
	}*/

	std::vector<std::pair<int,int> > rangeList;
	rangeList.push_back(std::pair<int,int>(237,241));

	struct timeval tstart,tend;
	gettimeofday(&tstart,NULL);

	GraphGlobals::setDeferUpdate(true);

	int rows = 0;
	int maxIndex = 0;

	for(int i = 0; i < rangeList.size(); ++i)
	{
	    if(rangeList[i].second < rangeList[i].first)
	    {
		continue;
	    }

	    rows += rangeList[i].second - rangeList[i].first + 1;

	    for(int j = rangeList[i].first; j <= rangeList[i].second; ++j)
	    {
		if(j > _microbePatients->getListSize())
		{
		    break;
		}
		std::map<int,std::vector<std::string> >::iterator it = _microbeTableList[_microbeTable->getIndex()]->testMap.find(j+1);
		if(it == _microbeTableList[_microbeTable->getIndex()]->testMap.end())
		{
		    continue;
		}

		if(it->second.size() > maxIndex)
		{
		    maxIndex = it->second.size();
		}
	    }
	}

	float bgLight = 0.9;
	float bgDark = 0.75;

	for(int i = 0; i < rangeList.size(); ++i)
	{
	    if(rangeList[i].second < rangeList[i].first)
	    {
		continue;
	    }

	    for(int j = rangeList[i].first; j <= rangeList[i].second; ++j)
	    {
		if(j > _microbePatients->getListSize())
		{
		    break;
		}
		std::map<int,std::vector<std::string> >::iterator it = _microbeTableList[_microbeTable->getIndex()]->testMap.find(j+1);
		if(it == _microbeTableList[_microbeTable->getIndex()]->testMap.end())
		{
		    continue;
		}

		float bgColor = ((float)(j - rangeList[i].first)) / ((float)(rows-1));
		bgColor = (1.0 - bgColor) * bgLight + bgColor * bgDark;

		for(int k = 0; k < it->second.size(); ++k)
		{
		    if(_microbeGraphType->getIndex() == 0)
		    {
			MicrobeGraphObject * mgo = new MicrobeGraphObject(_dbm, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);
			bool tb = mgo->setGraph(_microbePatients->getValue(j), j+1, it->second[k], _microbeTableList[_microbeTable->getIndex()]->testTimeMap[it->first][k],(int)_microbeNumBars->getValue(),_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix,_microbeGrouping->getValue(),_microbeOrdering->getValue(),(MicrobeGraphType)(_microbeLevel->getIndex()));
			if(tb)
			{
			    checkLayout();
			    _layoutObject->addGraphObject(mgo);
			    mgo->setBGColor(osg::Vec4(bgColor,bgColor,bgColor,1.0));
			}
			else
			{
			    delete mgo;
			}
		    }
		    else if(_microbeGraphType->getIndex() == 1)
		    {
			if(!_currentSBGraph)
			{
			    _currentSBGraph = new MicrobeBarGraphObject(_dbm, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);
			    _currentSBGraph->addGraph(_microbePatients->getValue(j), j+1, it->second[k],_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix);
			    checkLayout();
			    _layoutObject->addGraphObject(_currentSBGraph);
			    _microbeMenu->addItem(_microbeDone);
			}
			else
			{
			    _currentSBGraph->addGraph(_microbePatients->getValue(j), j+1, it->second[k], _microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix);
			}
		    }
		    else if(_microbeGraphType->getIndex() == 2)
		    {
			if(!_currentVBGraph)
			{
			    _currentVBGraph = new MicrobeVerticalBarGraphObject(_dbm, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);

			    switch((MicrobeGraphType)(_microbeLevel->getIndex()))
			    {
				case MGT_PHYLUM:
				    {
					_currentVBGraph->setNameList(_scatterPhylumList);
					break;
				    }
				case MGT_FAMILY:
				    {
					_currentVBGraph->setNameList(_microbeTableList[_microbeTable->getIndex()]->familyListPO);
					break;
				    }
				case MGT_GENUS:
				    {
					_currentVBGraph->setNameList(_microbeTableList[_microbeTable->getIndex()]->genusListPO);
					break;
				    }
				default:
				    {
					_currentVBGraph->setNameList(_microbeTableList[_microbeTable->getIndex()]->microbeListPO);
					break;
				    }
			    }
			    _currentVBGraph->setGroupList(_scatterPhylumList);

			    if(_currentVBGraph->addGraph(_microbePatients->getValue(j), j+1,it->second[k],_microbeTableList[_microbeTable->getIndex()]->testTimeMap[it->first][k],_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix,(MicrobeGraphType)(_microbeLevel->getIndex())))
			    {
				checkLayout();
				_layoutObject->addGraphObject(_currentVBGraph);
				_microbeMenu->addItem(_microbeDone);
			    }
			    else
			    {
				delete _currentVBGraph;
				_currentVBGraph = NULL;
			    }
			}
			else
			{
			    _currentVBGraph->addGraph(_microbePatients->getValue(j), j+1,it->second[k],_microbeTableList[_microbeTable->getIndex()]->testTimeMap[it->first][k],_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix,(MicrobeGraphType)(_microbeLevel->getIndex()));
			}
		    }
		}
	    }
	}

	if(_layoutObject)
	{
	    if(maxIndex > 0)
	    {
		_layoutObject->setRows((float)maxIndex);
	    }
	}

	gettimeofday(&tend,NULL);
	std::cerr << "Total load time: " << (tend.tv_sec - tstart.tv_sec) + ((tend.tv_usec - tstart.tv_usec) / 1000000.0) << std::endl;

	GraphGlobals::setDeferUpdate(false);
	if(_layoutObject)
	{
	    _layoutObject->forceUpdate();
	}
    }

    if(item == _microbeLoadCrohnsAll || item == _microbeLoadHealthyAll || item == _microbeLoadUCAll || item == _microbeLoadHealthy105All || item == _microbeLoadHealthy252All || item == _microbeLoadLarryAll)
    {
	/*std::string tablesuffix;
	if(_microbeTable->getIndex() == 1)
	{
	    tablesuffix = "_V2";
	}*/

	int microbesToLoad = 75;

	std::vector<std::pair<int,int> > rangeList;

	if(item == _microbeLoadLarryAll)
	{
	    rangeList.push_back(std::pair<int,int>(0,0));
	    microbesToLoad = (int)_microbeNumBars->getValue();
	}
	else if(item == _microbeLoadCrohnsAll)
	{
        //std::cerr << "crohns item\n";
	    //rangeList.push_back(std::pair<int,int>(44,58));
	    rangeList.push_back(std::pair<int,int>(338,342));
	}
	else if(item == _microbeLoadHealthyAll)
	{
	    rangeList.push_back(std::pair<int,int>(65,99));
	}
	else if(item == _microbeLoadHealthy105All)
	{
	    rangeList.push_back(std::pair<int,int>(65,99));
	    rangeList.push_back(std::pair<int,int>(118,236));
	}
	else if(item == _microbeLoadHealthy252All)
	{
	    rangeList.push_back(std::pair<int,int>(65,99));
	    rangeList.push_back(std::pair<int,int>(118,236));
	    rangeList.push_back(std::pair<int,int>(349,447));
	}
	else
	{
	    //rangeList.push_back(std::pair<int,int>(59,64));
	    rangeList.push_back(std::pair<int,int>(343,348));
	}

	struct timeval tstart,tend;
	gettimeofday(&tstart,NULL);

    // TODO this needs to be adjusted!!!!
	GraphGlobals::setDeferUpdate(true);
	for(int i = 0; i < rangeList.size(); ++i)
	{
	    int start = rangeList[i].first;
	    while(start <= rangeList[i].second)
	    {
		std::cerr << "Loading graph " << start << std::endl;
		updateMicrobeTests(start + 1);

		for(int j =0; j < _microbeTestTime.size(); ++j)
		{
		    if(_microbeGraphType->getIndex() == 0)
		    {
			//struct timeval loadstart,loadend;
			//gettimeofday(&loadstart,NULL);
			MicrobeGraphObject * mgo = new MicrobeGraphObject(_dbm, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);

			bool tb = mgo->setGraph(_microbePatients->getValue(start), start+1, _microbeTest->getValue(j), _microbeTestTime[j],microbesToLoad,_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix,_microbeGrouping->getValue(),_microbeOrdering->getValue(),(MicrobeGraphType)(_microbeLevel->getIndex()));
			if(tb)
			{
			    checkLayout();
			    _layoutObject->addGraphObject(mgo);
			}
			else
			{
			    delete mgo;
			}
			//gettimeofday(&loadend,NULL);
			//std::cerr << "load time: " << (loadend.tv_sec - loadstart.tv_sec) + ((loadend.tv_usec - loadstart.tv_usec) / 1000000.0) << std::endl;
		    }
		    else if(_microbeGraphType->getIndex() == 1)
		    {
			if(!_currentSBGraph)
			{
			    _currentSBGraph = new MicrobeBarGraphObject(_dbm, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);
			    _currentSBGraph->addGraph(_microbePatients->getValue(start), start+1, _microbeTest->getValue(j),_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix);
			    checkLayout();
			    _layoutObject->addGraphObject(_currentSBGraph);
			    _microbeMenu->addItem(_microbeDone);
			}
			else
			{
			    _currentSBGraph->addGraph(_microbePatients->getValue(start), start+1, _microbeTest->getValue(j), _microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix);
			}
		    }
		    else if(_microbeGraphType->getIndex() == 2)
		    {
			if(!_currentVBGraph)
			{
			    _currentVBGraph = new MicrobeVerticalBarGraphObject(_dbm, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);

			    switch((MicrobeGraphType)(_microbeLevel->getIndex()))
			    {
				case MGT_PHYLUM:
				    {
					_currentVBGraph->setNameList(_scatterPhylumList);
					break;
				    }
				case MGT_FAMILY:
				    {
					_currentVBGraph->setNameList(_microbeTableList[_microbeTable->getIndex()]->familyListPO);
					break;
				    }
				case MGT_GENUS:
				    {
					_currentVBGraph->setNameList(_microbeTableList[_microbeTable->getIndex()]->genusListPO);
					break;
				    }
				default:
				    {
					_currentVBGraph->setNameList(_microbeTableList[_microbeTable->getIndex()]->microbeListPO);
					break;
				    }
			    }
			    _currentVBGraph->setGroupList(_scatterPhylumList);

			    if(_currentVBGraph->addGraph(_microbePatients->getValue(start), start+1, _microbeTest->getValue(j),_microbeTestTime[j],_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix,(MicrobeGraphType)(_microbeLevel->getIndex())))
			    {
				checkLayout();
				_layoutObject->addGraphObject(_currentVBGraph);
				_microbeMenu->addItem(_microbeDone);
			    }
			    else
			    {
				delete _currentVBGraph;
				_currentVBGraph = NULL;
			    }
			}
			else
			{
			    _currentVBGraph->addGraph(_microbePatients->getValue(start), start+1, _microbeTest->getValue(j),_microbeTestTime[j],_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix,(MicrobeGraphType)(_microbeLevel->getIndex()));
			}
		    }
		}

		start++;
	    }
	}
	updateMicrobeTests(_microbePatients->getIndex() + 1);

	gettimeofday(&tend,NULL);
	std::cerr << "Total load time: " << (tend.tv_sec - tstart.tv_sec) + ((tend.tv_usec - tstart.tv_usec) / 1000000.0) << std::endl;

	GraphGlobals::setDeferUpdate(false);
	if(_layoutObject)
	{
	    _layoutObject->forceUpdate();
	}

	return;
    }

    if(item == _microbeLoadAverage || item == _microbeLoadHealthyAverage || item == _microbeLoadCrohnsAverage || item == _microbeLoadSmarrAverage || item == _microbeLoadSRSAverage || item == _microbeLoadSRXAverage)
    {
	/*std::string tablesuffix;
	if(_microbeTable->getIndex() == 1)
	{
	    tablesuffix = "_V2";
	}*/

	SpecialMicrobeGraphType mgt;
	if(item == _microbeLoadAverage)
	{
	    mgt = SMGT_AVERAGE;
	}
	else if(item == _microbeLoadHealthyAverage)
	{
	    mgt = SMGT_HEALTHY_AVERAGE;
	}
	else if(item == _microbeLoadCrohnsAverage)
	{
	    mgt = SMGT_CROHNS_AVERAGE;
	}
	else if(item == _microbeLoadSmarrAverage)
	{
	    mgt = SMGT_SMARR_AVERAGE;
	}
	else if(item == _microbeLoadSRSAverage)
	{
	    return;
	    mgt = SMGT_SRS_AVERAGE;
	}
	else if(item == _microbeLoadSRXAverage)
	{
	    return;
	    mgt = SMGT_SRX_AVERAGE;
	}

	// Bar Graph
	if(_microbeGraphType->getIndex() == 0)
	{
	    MicrobeGraphObject * mgo = new MicrobeGraphObject(_dbm, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);

	    if(mgo->setSpecialGraph(mgt,(int)_microbeNumBars->getValue(),_microbeRegionList->getValue(),_microbeGrouping->getValue(),_microbeOrdering->getValue(),(MicrobeGraphType) (_microbeLevel->getIndex())))
	    {
		//PluginHelper::registerSceneObject(mgo,"FuturePatient");
		//mgo->attachToScene();
		//_microbeGraphList.push_back(mgo);
		checkLayout();
		_layoutObject->addGraphObject(mgo);
	    }
	    else
	    {
		delete mgo;
	    }
	}
	else if(_microbeGraphType->getIndex() == 1)
	{
	    if(!_currentSBGraph)
	    {
		_currentSBGraph = new MicrobeBarGraphObject(_dbm, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);
		_currentSBGraph->addSpecialGraph(mgt,_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix);
		checkLayout();
		_layoutObject->addGraphObject(_currentSBGraph);
		_microbeMenu->addItem(_microbeDone);
	    }
	    else
	    {
		_currentSBGraph->addSpecialGraph(mgt,_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix);
	    }
	}
	else if(_microbeGraphType->getIndex() == 2)
	{
	    if(!_currentVBGraph)
	    {
		_currentVBGraph = new MicrobeVerticalBarGraphObject(_dbm, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);

		switch((MicrobeGraphType)(_microbeLevel->getIndex()))
		{
		    case MGT_PHYLUM:
		    {
			_currentVBGraph->setNameList(_scatterPhylumList);
			break;
		    }
		    case MGT_FAMILY:
		    {
			_currentVBGraph->setNameList(_microbeTableList[_microbeTable->getIndex()]->familyListPO);
			break;
		    }
		    case MGT_GENUS:
		    {
			_currentVBGraph->setNameList(_microbeTableList[_microbeTable->getIndex()]->genusListPO);
			break;
		    }
		    default:
		    {
			_currentVBGraph->setNameList(_microbeTableList[_microbeTable->getIndex()]->microbeListPO);
			break;
		    }
		}
		_currentVBGraph->setGroupList(_scatterPhylumList);

		if(_currentVBGraph->addSpecialGraph(mgt,_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix,_microbeRegionList->getValue(),(MicrobeGraphType) (_microbeLevel->getIndex())))
		{
		    checkLayout();
		    _layoutObject->addGraphObject(_currentVBGraph);
		    _microbeMenu->addItem(_microbeDone);
		}
		else
		{
		    delete _currentVBGraph;
		    _currentVBGraph = NULL;
		}
	    }
	    else
	    {
		_currentVBGraph->addSpecialGraph(mgt,_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix,_microbeRegionList->getValue(),(MicrobeGraphType) (_microbeLevel->getIndex()));
	    }
	}
	else if(_microbeGraphType->getIndex() == 3) 
	{
	    MicrobeGraphWAllObject * mgo = new MicrobeGraphWAllObject(_dbm, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);
	    if(mgo->setSpecialGraph(mgt,(int)_microbeNumBars->getValue(),_microbeRegionList->getValue(),_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix,_microbeGrouping->getValue(),_microbeOrdering->getValue(),(MicrobeGraphType)(_microbeLevel->getIndex())))
	    {
		checkLayout();
		_layoutObject->addGraphObject(mgo);
	    }
	    else
	    {
		delete mgo;
	    }

	}
    }

    if(item == _microbeLoadPointLine)
    {
	/*std::string tablesuffix;
	if(_microbeTable->getIndex() == 1)
	{
	    tablesuffix = "_V2";
	}*/

	MicrobePointLineObject * mplo = new MicrobePointLineObject(_dbm, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);
	if(mplo->setGraph(_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix,_microbePointLineExpand->getValue()))
	{
	    checkLayout();
	    _layoutObject->addGraphObject(mplo);
	}
	else
	{
	    delete mplo;
	}
    }

    if(item == _sMicrobeLogCB)
    {
	if(_layoutObject)
	{
	    _layoutObject->setSingleMicrobeLogScale(_sMicrobeLogCB->getValue());
	}
    }

    if(item == _sMicrobeStdDevCB)
    {
	if(_layoutObject)
	{
	    _layoutObject->setSingleMicrobeShowStdDev(_sMicrobeStdDevCB->getValue());
	}
    }

    if((item == _sMicrobeLoad || item == _sMicrobeEntry) && _sMicrobes->getListSize())
    {
	//std::string tablesuffix;
	int taxid = -1;
	std::string name;

	if(item == _sMicrobeLoad)
	{
	    name = _sMicrobes->getValue();
	    taxid = _microbeTableList[_microbeTable->getIndex()]->microbeIDList[_sMicrobes->getIndex()];
	}
	else
	{
	    std::string text = _sMicrobeEntry->getText();

	    //std::cerr << "Entry text: " << text << std::endl;
	    for(int i = 0; i < _sMicrobes->getListSize() ; ++i)
	    {
		if(text == _sMicrobes->getValue(i))
		{
		    name = _sMicrobes->getValue(i);
		    if(i < _microbeTableList[_microbeTable->getIndex()]->microbeIDList.size())
		    {
			taxid = _microbeTableList[_microbeTable->getIndex()]->microbeIDList[i];
		    }
		    else
		    {
			taxid = 0;
		    }

		    //std::cerr << "loading microbe: " << name << " id: " << taxid << std::endl;
		    break;
		}
	    }
	    _sMicrobeEntry->setText("");
	}

	if(taxid == -1)
	{
	    return;
	}

	/*if(_microbeTable->getIndex() == 0)
	{
	    taxid = _microbeIDList[_sMicrobes->getIndex()];
	}
	else if(_microbeTable->getIndex() == 1)
	{
	    taxid = _microbeV2IDList[_sMicrobes->getIndex()];
	    tablesuffix = "_V2";
	}*/


	if(_sMicrobeChartType->getValue() == "Bar")
	{
	    SingleMicrobeObject * smo = new SingleMicrobeObject(_dbm, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);
	    if(smo->setGraph(name,"",taxid,_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix,(MicrobeGraphType) (_sMicrobeType->getIndex()),_sMicrobeRankOrder->getValue(),_sMicrobeLabels->getValue(),_sMicrobeFirstTimeOnly->getValue(),_sMicrobeGroupPatients->getValue(),_sMicrobeGroup->getValue()))
	    {
		checkLayout();
		_layoutObject->addGraphObject(smo);
		smo->setLogScale(_sMicrobeLogCB->getValue());
		smo->setShowStdDev(_sMicrobeStdDevCB->getValue());
	    }
	    else
	    {
		delete smo;
	    }
	}
	else if(_sMicrobeChartType->getValue() == "Line")
	{
	    MicrobeLineGraphObject * mlgobject = new MicrobeLineGraphObject(_dbm, 1000.0, 1000.0, "MicrobeLineGraph", false, true, false, true, false);
	    if(mlgobject->addGraph("Smarr",name,(MicrobeGraphType) (_sMicrobeType->getIndex()),_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix))
	    {
		checkLayout();
		mlgobject->setLayoutDoesDelete(true);
		_layoutObject->addGraphObject(mlgobject);
	    }
	    else
	    {
		delete mlgobject;
	    }
	}
	return;
    }

    if(item == _sMicrobeLoadLines)
    {
	MicrobeLineGraphObject * mlgobject = new MicrobeLineGraphObject(_dbm, 1000.0, 1000.0, "MicrobeLineGraph", false, true, false, true, false);
	if(mlgobject->addGraph("Smarr",_sMicrobes->getValue(),(MicrobeGraphType) (_sMicrobeType->getIndex()),_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix))
	{
	    checkLayout();
	    mlgobject->setLayoutDoesDelete(true);
	    _layoutObject->addGraphObject(mlgobject);
	}
	else
	{
	    delete mlgobject;
	}

    }

    for(int j = 0; j < _sMicrobePresetList.size(); ++j)
    {
	if(item == _sMicrobePresetList[j] && _sMicrobes->getListSize())
	{
	    int index = -1;
	    for(int i = 0; i < _microbeTableList[_microbeTable->getIndex()]->microbeList.size(); ++i)
	    {
		if(_sMicrobePresetList[j]->getText() == _microbeTableList[_microbeTable->getIndex()]->microbeList[i])
		{
		    index = i;
		    break;
		}
	    }

	    if(index >= 0)
	    {
		if(_sMicrobeChartType->getValue() == "Bar")
		{
		    SingleMicrobeObject * smo = new SingleMicrobeObject(_dbm, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);
		    if(smo->setGraph(_sMicrobePresetList[j]->getText(),"",_microbeTableList[_microbeTable->getIndex()]->microbeIDList[index],_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix, MGT_SPECIES,_sMicrobeRankOrder->getValue(),_sMicrobeLabels->getValue(),_sMicrobeFirstTimeOnly->getValue(),_sMicrobeGroupPatients->getValue(),_sMicrobeGroup->getValue()))
		    {
			checkLayout();
			_layoutObject->addGraphObject(smo);
			smo->setLogScale(_sMicrobeLogCB->getValue());
			smo->setShowStdDev(_sMicrobeStdDevCB->getValue());
		    }
		    else
		    {
			delete smo;
		    }
		}
		else if(_sMicrobeChartType->getValue() == "Line")
		{
		    MicrobeLineGraphObject * mlgobject = new MicrobeLineGraphObject(_dbm, 1000.0, 1000.0, "MicrobeLineGraph", false, true, false, true, false);
		    if(mlgobject->addGraph("Smarr",_sMicrobePresetList[j]->getText(),MGT_SPECIES,_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix))
		    {
			checkLayout();
			mlgobject->setLayoutDoesDelete(true);
			_layoutObject->addGraphObject(mlgobject);
		    }
		    else
		    {
			delete mlgobject;
		    }
		}
	    }
	}
    }

    if(item == _sMicrobePhenotypeLoad)
    {
	loadPhenotype();
	return;
    }

    if(item == _otuLoad)
    {
	OtuGraphObject * ogo = new OtuGraphObject(_dbm, 1000.0, 1000.0, "OTU Graph", false, true, false, true);
	if(ogo->setGraph(_otuSampleList->getValue(),(int)_otuCount->getValue(),(MicrobeGraphType)_otuType->getIndex()))
	{
	    checkLayout();
	    _layoutObject->addGraphObject(ogo);
	}
	else
	{
	    delete ogo;
	}
	return;
    }

    if(item == _strainGroupList)
    {
	if(_strainGroupMap.find(_strainGroupList->getValue()) != _strainGroupMap.end())
	{
	    _strainList->setValues(_strainGroupMap[_strainGroupList->getValue()]);
	}
	return;
    }

    if(item == _strainLoadButton)
    {
	if(_strainGroupList->getListSize() && _strainList->getListSize())
	{
	    StrainGraphObject * sgo = new StrainGraphObject(_dbm, 1000.0, 1000.0, "Strain Graph", false, true, false, true);

	    if(sgo->setGraph(_strainList->getValue(),_strainIdMap[_strainList->getValue()],_strainLarryOnlyCB->getValue()))
	    {
		checkLayout();
		_layoutObject->addGraphObject(sgo);
	    }
	    else
	    {
		delete sgo;
	    }
	}

	return;
    }

    if(item == _strainLoadAllButton)
    {
	//TODO: implement another ordering and maybe limit
	if(_strainGroupList->getListSize() && _strainList->getListSize())
	{
	    std::map<std::string,std::vector<std::string> >::iterator it;
	    it = _strainGroupMap.find(_strainGroupList->getValue());
	    if(it != _strainGroupMap.end())
	    {
		std::cerr << "Loading " << it->second.size() << " strains." << std::endl;
		for(int i = 0; i < it->second.size(); ++i)
		{
		    std::cerr << "Loading strain " << i << " id: " << _strainIdMap[it->second[i]] << std::endl;
		    StrainGraphObject * sgo = new StrainGraphObject(_dbm, 1000.0, 1000.0, "Strain Graph", false, true, false, true);

		    if(sgo->setGraph(it->second[i],_strainIdMap[it->second[i]],_strainLarryOnlyCB->getValue()))
		    {
			checkLayout();
			_layoutObject->addGraphObject(sgo);
		    }
		    else
		    {
			delete sgo;
		    }
		}
	    }
	}
	return;
    }

    if(item == _strainLoadHeatMap)
    {
	if(_strainGroupList->getListSize() && _strainList->getListSize())
	{
	    std::map<std::string,std::vector<std::string> >::iterator it;
	    it = _strainGroupMap.find(_strainGroupList->getValue());
	    if(it != _strainGroupMap.end())
	    {
		std::cerr << "Loading " << it->second.size() << " strains." << std::endl;
		for(int i = 0; i < it->second.size(); ++i)
		{
		    std::cerr << "Loading strain " << i << " id: " << _strainIdMap[it->second[i]] << std::endl;
		    StrainHMObject * shmo = new StrainHMObject(_dbm, 1000.0, 1000.0, "Strain Heat Map", false, true, false, true);

		    if(shmo->setGraph(it->second[i],"Smarr",1,_strainIdMap[it->second[i]],osg::Vec4(1,0,0,1)))
		    {
			checkLayout();
			_layoutObject->addGraphObject(shmo);
		    }
		    else
		    {
			delete shmo;
		    }
		}
	    }
	}
    }

    if(item == _eventLoad && _eventName->getListSize())
    {
	if(!_currentSymptomGraph)
	{
	    _currentSymptomGraph = new SymptomGraphObject(_dbm, 1000.0, 1000.0, "Symptom Graph", false, true, false, true);
	    _currentSymptomGraph->addGraph(_eventName->getValue());
	    checkLayout();
	    _layoutObject->addGraphObject(_currentSymptomGraph);
	    _eventMenu->addItem(_eventDone);
	}
	else
	{
	    _currentSymptomGraph->addGraph(_eventName->getValue());
	}
    }

    if(item == _eventLoadDisplay && _eventName->getListSize())
    {
	bool addObject = false;
	if(!_currentSymptomGraph)
	{
	    _currentSymptomGraph = new SymptomGraphObject(_dbm, 1000.0, 1000.0, "Symptom Graph", false, true, false, true);
	    checkLayout();
	    addObject = true;
	}
	/*for(int i = 0; i < _eventName->getListSize(); ++i)
	{
	    if(_eventGroups[i] == 1 || _eventGroups[i] == 2 || _eventGroups[i] == 3 || _eventGroups[i] == 5)
	    {
		_currentSymptomGraph->addGraph(_eventName->getValue(i));
	    }
	}*/

	_currentSymptomGraph->addGraph("Blood");
	_currentSymptomGraph->addGraph("Flare");
	_currentSymptomGraph->addGraph("Malaise");
	_currentSymptomGraph->addPeripheral();
	_currentSymptomGraph->addGraph("Nasal");

	for(int i = 0; i < _eventName->getListSize(); ++i)
	{
	    if(_eventGroups[i] == 5)
	    {
		_currentSymptomGraph->addGraph(_eventName->getValue(i));
	    }
	}

	_currentSymptomGraph->addGraph("Stool");

	/*for(int i = 0; i < _eventName->getListSize(); ++i)
	{
	    if(_eventGroups[i] == 6)
	    {
		_currentSymptomGraph->addGraph(_eventName->getValue(i));
	    }
	}*/

	if(addObject)
	{
	    _layoutObject->addGraphObject(_currentSymptomGraph);
	    _eventMenu->addItem(_eventDone);
	}
    }

    if(item == _eventLoadAll && _eventName->getListSize())
    {
	bool addObject = false;
	if(!_currentSymptomGraph)
	{
	    _currentSymptomGraph = new SymptomGraphObject(_dbm, 1000.0, 1000.0, "Symptom Graph", false, true, false, true);
	    checkLayout();
	    addObject = true;
	}
	for(int i = 0; i < _eventName->getListSize(); ++i)
	{
	    _currentSymptomGraph->addGraph(_eventName->getValue(i));
	}

	if(addObject)
	{
	    _layoutObject->addGraphObject(_currentSymptomGraph);
	    _eventMenu->addItem(_eventDone);
	}
    }

    if(item == _eventLoadMicrobe)
    {
	if(!_currentSymptomGraph)
	{
	    _currentSymptomGraph = new SymptomGraphObject(_dbm, 1000.0, 1000.0, "Symptom Graph", false, true, false, true);
	    _currentSymptomGraph->addGraph("Microbe Test");
	    checkLayout();
	    _layoutObject->addGraphObject(_currentSymptomGraph);
	    _eventMenu->addItem(_eventDone);
	}
	else
	{
	    _currentSymptomGraph->addGraph("Microbe Test");
	}
    }

    if(item == _eventDone)
    {
	_currentSymptomGraph = NULL;
	_eventMenu->removeItem(_eventDone);
    }

    if(item == _scatterFirstEntry)
    {
	std::string entryText = _scatterFirstEntry->getText();

	for(int i = 0; i < _scatterFirstList->getListSize(); ++i)
	{
	    if(_scatterFirstList->getValue(i) == entryText)
	    {
		_scatterFirstList->setIndex(i);
		break;
	    }
	}

	_scatterFirstEntry->setText("");
    }

    if(item == _scatterSecondEntry)
    {
	std::string entryText = _scatterSecondEntry->getText();

	for(int i = 0; i < _scatterSecondList->getListSize(); ++i)
	{
	    if(_scatterSecondList->getValue(i) == entryText)
	    {
		_scatterSecondList->setIndex(i);
		break;
	    }
	}

	_scatterSecondEntry->setText("");
    }

    if(item == _scatterLogCB)
    {
	if(_layoutObject)
	{
	    _layoutObject->setScatterLogScale(_scatterLogCB->getValue());
	}
    }

    if(item == _scatterLoad)
    {
	if(_scatterFirstList->getListSize() && _scatterFirstList->getIndex() != _scatterSecondList->getIndex())
	{
	    MicrobeScatterGraphObject * msgo = new MicrobeScatterGraphObject(_dbm, 1000.0, 1000.0, "Scatter Plot", false, true, false, true);
	    if(msgo->setGraph(_scatterSecondList->getValue() + " vs " + _scatterFirstList->getValue(),_scatterFirstList->getValue(),_scatterSecondList->getValue(),(MicrobeGraphType)_scatterMicrobeType->getIndex(),_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix))
	    {
		checkLayout();
		_layoutObject->addGraphObject(msgo);
		msgo->setLogScale(_scatterLogCB->getValue());
	    }
	    else
	    {
		delete msgo;
	    }
	}
	return;
    }

    if(item == _scatterLoadAll)
    {
	if(_scatterFirstList->getListSize())
	{
	    for(int i = 0; i < _scatterSecondList->getListSize(); ++i)
	    {
		if(_scatterFirstList->getIndex() == i)
		{
		    continue;
		}
		MicrobeScatterGraphObject * msgo = new MicrobeScatterGraphObject(_dbm, 1000.0, 1000.0, "Scatter Plot", false, true, false, true);
		if(msgo->setGraph(_scatterSecondList->getValue(i) + " vs " + _scatterFirstList->getValue(),_scatterFirstList->getValue(),_scatterSecondList->getValue(i),(MicrobeGraphType)_scatterMicrobeType->getIndex(),_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix))
		{
		    checkLayout();
		    _layoutObject->addGraphObject(msgo);
		    msgo->setLogScale(_scatterLogCB->getValue());
		}
		else
		{
		    delete msgo;
		}
	    }
	}
	return;
    }

    if(item == _scatterLoadFilter)
    {
	loadScatter();
	return;
    }

    for(int k = 0; k < _scatterPresetButtons.size(); ++k)
    {
	if(item == _scatterPresetButtons[k])
	{
	    MicrobeGraphType mgt;
	    switch(_scatterMicrobeType->getIndex())
	    {
		case MGT_FAMILY:
		    {
			//std::cerr << "Family pheno load." << std::endl;
			mgt = MGT_FAMILY;
			break;
		    }
		case MGT_SPECIES:
		default:
		    {
			//std::cerr << "Species pheno load." << std::endl;
			mgt = MGT_SPECIES;
			break;
		    }
	    }

	    GraphGlobals::setDeferUpdate(true);
	    for(int i = 0; i < _scatterPresets[k].size(); ++i)
	    {
		for(int j = (_scatterPresets[k].size()-1); j > i; --j)
		{
		    MicrobeScatterGraphObject * msgo = new MicrobeScatterGraphObject(_dbm, 1000.0, 1000.0, "Scatter Plot", false, true, false, true);
		    if(msgo->setGraph(_scatterPresets[k][i] + " vs " + _scatterPresets[k][j],_scatterPresets[k][i],_scatterPresets[k][j],mgt,_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix))
		    {
			checkLayout();
			_layoutObject->addGraphObject(msgo);
			msgo->setLogScale(_scatterLogCB->getValue());
		    }
		    else
		    {
			delete msgo;
		    }
		}
	    }
	    GraphGlobals::setDeferUpdate(false);
	    if(_layoutObject)
	    {
		_layoutObject->forceUpdate();
	    }

	    return;
	}
    }

    if(item == _saveLayoutButton)
    {
	saveLayout();
	return;
    }

    for(int i = 0; i < _loadLayoutButtons.size(); ++i)
    {
	if(item == _loadLayoutButtons[i])
	{
	    loadLayout(_loadLayoutButtons[i]->getText());
	    return;
	}
    }
}

void FuturePatient::initOtuMenu()
{
    struct fileName
    {
	char name[256];
    };

    std::string otuDir = ConfigManager::getEntry("value","Plugin.FuturePatient.OtuDir","");

    struct fileName * names = NULL;
    int count = 0;

    if(ComController::instance()->isMaster())
    {
	DIR * dir;

	if ((dir = opendir(otuDir.c_str())) == NULL)
	{
	    std::cerr << "Unable to open directory: " << otuDir << std::endl;
	}
	else
	{
	    dirent * entry;
	    struct stat st;
	    while ((entry = readdir(dir)) != NULL)
	    {
		std::string fullPath(otuDir + "/" + entry->d_name);
		stat(fullPath.c_str(), &st);
		if(!S_ISDIR(st.st_mode))
		{
		    count++;
		}
	    }

	    if(count)
	    {
		names = new fileName[count];
		int listIndex = 0;
		rewinddir(dir);
		while ((entry = readdir(dir)) != NULL)
		{
		    std::string fullPath(otuDir + "/" + entry->d_name);
		    stat(fullPath.c_str(), &st);
		    if(!S_ISDIR(st.st_mode))
		    {
			strncpy(names[listIndex].name,entry->d_name,255);
			listIndex++;
		    }
		}
	    }
	}
	ComController::instance()->sendSlaves(&count,sizeof(int));
	if(count)
	{
	    ComController::instance()->sendSlaves(names,count*sizeof(struct fileName));
	}
    }
    else
    {
	ComController::instance()->readMaster(&count,sizeof(int));
	if(count)
	{
	    names = new fileName[count];
	    ComController::instance()->readMaster(names,count*sizeof(struct fileName));
	}
    }

    std::vector<std::string> fileList;
    for(int i = 0; i < count; ++i)
    {
	fileList.push_back(names[i].name);
    }
    std::sort(fileList.begin(),fileList.end());
    _otuSampleList->setValues(fileList);

    if(count > 0)
    {
	delete[] names;
    }
}

void FuturePatient::checkLayout()
{
    if(!_layoutObject)
    {
	float width, height;
	osg::Vec3 pos;
	width = ConfigManager::getFloat("width","Plugin.FuturePatient.Layout",1500.0);
	height = ConfigManager::getFloat("height","Plugin.FuturePatient.Layout",1000.0);
	pos = ConfigManager::getVec3("Plugin.FuturePatient.Layout");
	_layoutObject = new GraphLayoutObject(width,height,3,"FuturePatient",false,true,false,true,false);
	_layoutObject->setPosition(pos);
	PluginHelper::registerSceneObject(_layoutObject,"FuturePatient");
	_layoutObject->attachToScene();
	_fpMenu->addItem(_closeLayoutButton);
    }
}

void FuturePatient::loadGraph(std::string patient, std::string test, bool averageColor)
{
    checkLayout();

    std::string value = patient + test;
    if(!patient.empty() && !test.empty())
    {
	if(!_multiAddCB->getValue())
	{
	    //if(_graphObjectMap.find(value) == _graphObjectMap.end())
	    {
		GraphObject * gobject = new GraphObject(_dbm, 1000.0, 1000.0, "DataGraph", false, true, false, true, false);
		if(gobject->addGraph(patient,test,_requireRangeCB->getValue(),averageColor))
		{
		    //_graphObjectMap[value] = gobject;
		    gobject->setLayoutDoesDelete(true);
		    _layoutObject->addGraphObject(gobject);
		}
		else
		{
		    delete gobject;
		}
	    }

	    //if(_graphObjectMap.find(value) != _graphObjectMap.end())
	    //{
		//_layoutObject->addGraphObject(_graphObjectMap[value]);
	    //}
	}
	else
	{
	    if(!_multiObject)
	    {
		_multiObject = new GraphObject(_dbm, 1000.0, 1000.0, "DataGraph", false, true, false, true, false);
	    }

	    if(_multiObject->addGraph(patient,test,_requireRangeCB->getValue()))
	    {
		if(_multiObject->getNumGraphs() == 1)
		{
		    _multiObject->setLayoutDoesDelete(true);
		    _layoutObject->addGraphObject(_multiObject);
		}
	    }
	}
    }
}

void FuturePatient::setupMicrobes()
{
    struct Microbes
    {
	char name[512];
	char phylum[512];
	int taxid;
    };

    struct Families
    {
	char name[512];
	char phylum[512];
    };
    
    struct Genuses
    {
	char name[512];
	char phylum[512];
    };

    Microbes * microbes;
    int numMicrobes;

    Families * families;
    int numFamilies;
    
    Genuses * genuses;
    int numGenuses;

    for(int j = 0; j < _microbeTableList.size(); ++j)
    {
	microbes = NULL;
	numMicrobes = 0;

	if(ComController::instance()->isMaster())
	{
	    if(_dbm)
	    {
		std::stringstream qss;
		//qss << "select distinct taxonomy_id, species from Microbes" << _microbeTableList[j]->microbeSuffix << " order by species;";
		qss << "select distinct Microbe_Measurement" << _microbeTableList[j]->measureSuffix << ".taxonomy_id, Microbes" << _microbeTableList[j]->microbeSuffix << ".species, Microbes" <<_microbeTableList[j]->microbeSuffix << ".phylum from Microbe_Measurement" << _microbeTableList[j]->measureSuffix << " inner join Microbes" << _microbeTableList[j]->microbeSuffix << " on Microbe_Measurement" << _microbeTableList[j]->measureSuffix << ".taxonomy_id = Microbes" << _microbeTableList[j]->microbeSuffix << ".taxonomy_id order by Microbes" << _microbeTableList[j]->microbeSuffix << ".species;";

		DBMQueryResult result;

		_dbm->runQuery(qss.str(),result);

		numMicrobes = result.numRows();

		if(numMicrobes)
		{
		    microbes = new struct Microbes[numMicrobes];

		    for(int i = 0; i < numMicrobes; ++i)
		    {
			strncpy(microbes[i].name,result(i,"species").c_str(),511);
			strncpy(microbes[i].phylum,result(i,"phylum").c_str(),511);
			microbes[i].taxid = atoi(result(i,"taxonomy_id").c_str());
		    }
		}

		ComController::instance()->sendSlaves(&numMicrobes,sizeof(int));
		if(numMicrobes)
		{
		    ComController::instance()->sendSlaves(microbes,numMicrobes*sizeof(struct Microbes));
		}
	    }

	    /*if(_conn)
	    {
		std::stringstream qss;
		qss << "select distinct taxonomy_id, species from Microbes" << _microbeTableList[j]->microbeSuffix << " order by species;";
		mysqlpp::Query q = _conn->query(qss.str());
		mysqlpp::StoreQueryResult res = q.store();

		numMicrobes = res.num_rows();

		if(numMicrobes)
		{
		    microbes = new struct Microbes[numMicrobes];

		    for(int i = 0; i < numMicrobes; ++i)
		    {
			strncpy(microbes[i].name,res[i]["species"].c_str(),511);
			microbes[i].taxid = atoi(res[i]["taxonomy_id"].c_str());
		    }
		}

		ComController::instance()->sendSlaves(&numMicrobes,sizeof(int));
		if(numMicrobes)
		{
		    ComController::instance()->sendSlaves(microbes,numMicrobes*sizeof(struct Microbes));
		}
	    }*/
	}
	else
	{
	    ComController::instance()->readMaster(&numMicrobes,sizeof(int));
	    if(numMicrobes)
	    {
		microbes = new struct Microbes[numMicrobes];
		ComController::instance()->readMaster(microbes,numMicrobes*sizeof(struct Microbes));
	    }
	}

	for(int i = 0; i < numMicrobes; ++i)
	{
	    //std::cerr << microbes[i].name << std::endl;
	    _microbeTableList[j]->microbeList.push_back(microbes[i].name);
	    _microbeTableList[j]->microbeIDList.push_back(microbes[i].taxid);
	}

	std::map<std::string,std::vector<std::string> > mphylum;
	for(int i = 0; i < numMicrobes; ++i)
	{
	    mphylum[microbes[i].phylum].push_back(microbes[i].name);
	}

	for(std::map<std::string,std::vector<std::string> >::iterator it = mphylum.begin(); it != mphylum.end(); ++it)
	{
	    for(int i = 0; i < it->second.size(); ++i)
	    {
		_microbeTableList[j]->microbeListPO.push_back(it->second[i]);
	    }
	}

	if(microbes)
	{
	    delete[] microbes;
	}

	families = NULL;
	numFamilies = 0;

	if(ComController::instance()->isMaster())
	{
	    if(_dbm)
	    {
		std::stringstream qss;
		//qss << "select distinct family from Microbes" << _microbeTableList[j]->microbeSuffix << " order by family;";
		qss << "select distinct Microbe_Measurement" << _microbeTableList[j]->measureSuffix << ".taxonomy_id, Microbes" << _microbeTableList[j]->microbeSuffix << ".family, Microbes" << _microbeTableList[j]->microbeSuffix << ".phylum from Microbe_Measurement" << _microbeTableList[j]->measureSuffix << " inner join Microbes" << _microbeTableList[j]->microbeSuffix << " on Microbe_Measurement" << _microbeTableList[j]->measureSuffix << ".taxonomy_id = Microbes" << _microbeTableList[j]->microbeSuffix << ".taxonomy_id group by Microbes" << _microbeTableList[j]->microbeSuffix << ".family order by Microbes" << _microbeTableList[j]->microbeSuffix << ".family;";

		DBMQueryResult result;

		_dbm->runQuery(qss.str(),result);

		numFamilies = result.numRows();

		if(numFamilies)
		{
		    families = new struct Families[numFamilies];

		    for(int i = 0; i < numFamilies; ++i)
		    {
			strncpy(families[i].name,result(i,"family").c_str(),511);
			strncpy(families[i].phylum,result(i,"phylum").c_str(),511);
		    }
		}

		ComController::instance()->sendSlaves(&numFamilies,sizeof(int));
		if(numFamilies)
		{
		    ComController::instance()->sendSlaves(families,numFamilies*sizeof(struct Families));
		}
	    }

	    /*if(_conn)
	    {
		std::stringstream qss;
		qss << "select distinct family from Microbes" << _microbeTableList[j]->microbeSuffix << " order by family;";
		mysqlpp::Query q = _conn->query(qss.str());
		mysqlpp::StoreQueryResult res = q.store();

		numFamilies = res.num_rows();

		if(numFamilies)
		{
		    families = new struct Families[numFamilies];

		    for(int i = 0; i < numFamilies; ++i)
		    {
			strncpy(families[i].name,res[i]["family"].c_str(),511);
		    }
		}

		ComController::instance()->sendSlaves(&numFamilies,sizeof(int));
		if(numFamilies)
		{
		    ComController::instance()->sendSlaves(families,numFamilies*sizeof(struct Families));
		}
	    }*/
	}
	else
	{
	    ComController::instance()->readMaster(&numFamilies,sizeof(int));
	    if(numFamilies)
	    {
		families = new struct Families[numFamilies];
		ComController::instance()->readMaster(families,numFamilies*sizeof(struct Families));
	    }
	}

	for(int i = 0; i < numFamilies; ++i)
	{
	    _microbeTableList[j]->familyList.push_back(families[i].name);
	}

	mphylum.clear();
	for(int i = 0; i < numFamilies; ++i)
	{
	    mphylum[families[i].phylum].push_back(families[i].name);
	}

	for(std::map<std::string,std::vector<std::string> >::iterator it = mphylum.begin(); it != mphylum.end(); ++it)
	{
	    for(int i = 0; i < it->second.size(); ++i)
	    {
		_microbeTableList[j]->familyListPO.push_back(it->second[i]);
	    }
	}

	if(families)
	{
	    delete[] families;
	}


	genuses = NULL;
	numGenuses = 0;

	if(ComController::instance()->isMaster())
	{
	    if(_dbm)
	    {
		std::stringstream qss;
		//qss << "select distinct genus from Microbes" << _microbeTableList[j]->microbeSuffix << " order by genus;";
		qss << "select distinct Microbe_Measurement" << _microbeTableList[j]->measureSuffix << ".taxonomy_id, Microbes" << _microbeTableList[j]->microbeSuffix << ".genus, Microbes" << _microbeTableList[j]->microbeSuffix << ".phylum from Microbe_Measurement" << _microbeTableList[j]->measureSuffix << " inner join Microbes" << _microbeTableList[j]->microbeSuffix << " on Microbe_Measurement" << _microbeTableList[j]->measureSuffix << ".taxonomy_id = Microbes" << _microbeTableList[j]->microbeSuffix << ".taxonomy_id group by Microbes" << _microbeTableList[j]->microbeSuffix << ".genus order by Microbes" << _microbeTableList[j]->microbeSuffix << ".genus;";

		DBMQueryResult result;

		_dbm->runQuery(qss.str(),result);

		numGenuses = result.numRows();

		if(numGenuses)
		{
		    genuses = new struct Genuses[numGenuses];

		    for(int i = 0; i < numGenuses; ++i)
		    {
			strncpy(genuses[i].name,result(i,"genus").c_str(),511);
			strncpy(genuses[i].phylum,result(i,"phylum").c_str(),511);
		    }
		}

		ComController::instance()->sendSlaves(&numGenuses,sizeof(int));
		if(numGenuses)
		{
		    ComController::instance()->sendSlaves(genuses,numGenuses*sizeof(struct Genuses));
		}
	    }

	    /*if(_conn)
	    {
		std::stringstream qss;
		qss << "select distinct genus from Microbes" << _microbeTableList[j]->microbeSuffix << " order by genus;";
		mysqlpp::Query q = _conn->query(qss.str());
		mysqlpp::StoreQueryResult res = q.store();

		numGenuses = res.num_rows();

		if(numGenuses)
		{
		    genuses = new struct Genuses[numGenuses];

		    for(int i = 0; i < numGenuses; ++i)
		    {
			strncpy(genuses[i].name,res[i]["genus"].c_str(),511);
		    }
		}

		ComController::instance()->sendSlaves(&numGenuses,sizeof(int));
		if(numGenuses)
		{
		    ComController::instance()->sendSlaves(genuses,numGenuses*sizeof(struct Genuses));
		}
	    }*/
	}
	else
	{
	    ComController::instance()->readMaster(&numGenuses,sizeof(int));
	    if(numGenuses)
	    {
		genuses = new struct Genuses[numGenuses];
		ComController::instance()->readMaster(genuses,numGenuses*sizeof(struct Genuses));
	    }
	}

	for(int i = 0; i < numGenuses; ++i)
	{
	    _microbeTableList[j]->genusList.push_back(genuses[i].name);
	}

	mphylum.clear();
	for(int i = 0; i < numGenuses; ++i)
	{
	    mphylum[genuses[i].phylum].push_back(genuses[i].name);
	}

	for(std::map<std::string,std::vector<std::string> >::iterator it = mphylum.begin(); it != mphylum.end(); ++it)
	{
	    for(int i = 0; i < it->second.size(); ++i)
	    {
		_microbeTableList[j]->genusListPO.push_back(it->second[i]);
	    }
	}


	if(genuses)
	{
	    delete[] genuses;
	}

    }

    /*microbes = NULL;
    numMicrobes = 0;

    if(ComController::instance()->isMaster())
    {
	if(_conn)
	{
	    mysqlpp::Query q = _conn->query("select distinct taxonomy_id, species from Microbes_V2 order by species;");
	    mysqlpp::StoreQueryResult res = q.store();

	    numMicrobes = res.num_rows();

	    if(numMicrobes)
	    {
		microbes = new struct Microbes[numMicrobes];

		for(int i = 0; i < numMicrobes; ++i)
		{
		    strncpy(microbes[i].name,res[i]["species"].c_str(),511);
		    microbes[i].taxid = atoi(res[i]["taxonomy_id"].c_str());
		}
	    }

	    ComController::instance()->sendSlaves(&numMicrobes,sizeof(int));
	    if(numMicrobes)
	    {
		ComController::instance()->sendSlaves(microbes,numMicrobes*sizeof(struct Microbes));
	    }
	}
    }
    else
    {
	ComController::instance()->readMaster(&numMicrobes,sizeof(int));
	if(numMicrobes)
	{
	    microbes = new struct Microbes[numMicrobes];
	    ComController::instance()->readMaster(microbes,numMicrobes*sizeof(struct Microbes));
	}
    }

    for(int i = 0; i < numMicrobes; ++i)
    {
	_microbeV2List.push_back(microbes[i].name);
	_microbeV2IDList.push_back(microbes[i].taxid);
    }

    if(microbes)
    {
	delete[] microbes;
    }*/

    _sMicrobes->setValues(_microbeTableList[_microbeTable->getIndex()]->microbeList);
    _sMicrobeEntry->setSearchList(_microbeTableList[_microbeTable->getIndex()]->microbeList,5);
}

void FuturePatient::setupMicrobePatients()
{
    struct PatientName
    {
	char name[64];
    };

    PatientName * names = NULL;
    int numNames = 0;

    if(ComController::instance()->isMaster())
    {
	if(_dbm)
	{
	    DBMQueryResult result;

	    _dbm->runQuery("select last_name, patient_id from Patient order by patient_id;",result);

	    numNames = result.numRows();

	    if(numNames)
	    {
		names = new struct PatientName[numNames];

		for(int i = 0; i < numNames; ++i)
		{
		    strncpy(names[i].name,result(i,"last_name").c_str(),63);
		}
	    }
	}
	
	/*if(_conn)
	{
	    mysqlpp::Query q = _conn->query("select last_name, patient_id from Patient order by patient_id;");
	    mysqlpp::StoreQueryResult res = q.store();

	    numNames = res.num_rows();

	    if(numNames)
	    {
		names = new struct PatientName[numNames];

		for(int i = 0; i < numNames; ++i)
		{
		    strncpy(names[i].name,res[i]["last_name"].c_str(),63);
		}
	    }
	}*/

	ComController::instance()->sendSlaves(&numNames,sizeof(int));
	if(numNames)
	{
	    ComController::instance()->sendSlaves(names,numNames*sizeof(struct PatientName));
	}
    }
    else
    {
	ComController::instance()->readMaster(&numNames,sizeof(int));
	if(numNames)
	{
	    names = new struct PatientName[numNames];
	    ComController::instance()->readMaster(names,numNames*sizeof(struct PatientName));
	}
    }

    std::vector<std::string> nameVec;
    for(int i = 0; i < numNames; ++i)
    {
	nameVec.push_back(names[i].name);
    }

    _microbePatients->setValues(nameVec);

    if(names)
    {
	delete[] names;
    }

    struct TestLabel
    {
	int id;
	char label[256];
	time_t timestamp;
    };

    TestLabel * labels;
    int numTests;

    for(int i = 0; i < _microbeTableList.size(); ++i)
    {
	labels = NULL;
	numTests = 0;

	if(ComController::instance()->isMaster())
	{
	    if(_dbm)
	    {
		std::stringstream qss;
		qss << "select patient_id, timestamp, unix_timestamp(timestamp) as utimestamp from Microbe_Measurement" << _microbeTableList[i]->measureSuffix << " where seq_type = 'fast' group by patient_id, timestamp order by patient_id, timestamp;";

		DBMQueryResult result;

		_dbm->runQuery(qss.str(),result);

		numTests = result.numRows();

		if(numTests)
		{
		    labels = new struct TestLabel[numTests];

		    for(int j = 0; j < numTests; ++j)
		    {
			labels[j].id = atoi(result(j,"patient_id").c_str());
			strncpy(labels[j].label,result(j,"timestamp").c_str(),255);
			labels[j].timestamp = atol(result(j,"utimestamp").c_str());
		    }
		}
	    }
	    
	    /*if(_conn)
	    {
		std::stringstream qss;
		qss << "select patient_id, timestamp, unix_timestamp(timestamp) as utimestamp from Microbe_Measurement" << _microbeTableList[i]->measureSuffix << " group by patient_id, timestamp order by patient_id, timestamp;";

		mysqlpp::Query q = _conn->query(qss.str().c_str());
		mysqlpp::StoreQueryResult res = q.store();

		numTests = res.num_rows();

		if(numTests)
		{
		    labels = new struct TestLabel[numTests];

		    for(int j = 0; j < numTests; ++j)
		    {
			labels[j].id = atoi(res[j]["patient_id"].c_str());
			strncpy(labels[j].label,res[j]["timestamp"].c_str(),255);
			labels[j].timestamp = atol(res[j]["utimestamp"].c_str());
		    }
		}
	    }*/

	    ComController::instance()->sendSlaves(&numTests,sizeof(int));
	    if(numTests)
	    {
		ComController::instance()->sendSlaves(labels,numTests*sizeof(struct TestLabel));
	    }
	}
	else
	{
	    ComController::instance()->readMaster(&numTests,sizeof(int));
	    if(numTests)
	    {
		labels = new struct TestLabel[numTests];
		ComController::instance()->readMaster(labels,numTests*sizeof(struct TestLabel));
	    }
	}

	for(int j = 0; j < numTests; ++j)
	{
	    _microbeTableList[i]->testMap[labels[j].id].push_back(labels[j].label);
	    _microbeTableList[i]->testTimeMap[labels[j].id].push_back(labels[j].timestamp);
	    //_patientMicrobeTestMap[labels[j].id].push_back(labels[j].label);
	    //_patientMicrobeTestTimeMap[labels[j].id].push_back(labels[j].timestamp);
	}

	if(labels)
	{
	    delete[] labels;
	}

    }

    /*labels = NULL;
    numTests = 0;
    if(ComController::instance()->isMaster())
    {
	if(_conn)
	{
	    std::stringstream qss;
	    qss << "select patient_id, timestamp, unix_timestamp(timestamp) as utimestamp from Microbe_Measurement_V2 group by patient_id, timestamp order by patient_id, timestamp;";

	    mysqlpp::Query q = _conn->query(qss.str().c_str());
	    mysqlpp::StoreQueryResult res = q.store();

	    numTests = res.num_rows();

	    if(numTests)
	    {
		labels = new struct TestLabel[numTests];

		for(int j = 0; j < numTests; ++j)
		{
		    labels[j].id = atoi(res[j]["patient_id"].c_str());
		    strncpy(labels[j].label,res[j]["timestamp"].c_str(),255);
		    labels[j].timestamp = atol(res[j]["utimestamp"].c_str());
		}
	    }
	}

	ComController::instance()->sendSlaves(&numTests,sizeof(int));
	if(numTests)
	{
	    ComController::instance()->sendSlaves(labels,numTests*sizeof(struct TestLabel));
	}
    }
    else
    {
	ComController::instance()->readMaster(&numTests,sizeof(int));
	if(numTests)
	{
	    labels = new struct TestLabel[numTests];
	    ComController::instance()->readMaster(labels,numTests*sizeof(struct TestLabel));
	}
    }

    for(int j = 0; j < numTests; ++j)
    {
	_patientMicrobeV2TestMap[labels[j].id].push_back(labels[j].label);
	_patientMicrobeV2TestTimeMap[labels[j].id].push_back(labels[j].timestamp);
    }

    if(labels)
    {
	delete[] labels;
    }*/
}

void FuturePatient::setupStrainMenu()
{
    struct Data
    {
	char name[1024];
	int value;
    };

    Data * names = NULL;
    int numNames = 0;

    if(ComController::instance()->isMaster())
    {
	if(_dbm)
	{
	    DBMQueryResult result;

	    _dbm->runQuery("select distinct genus from TaxonomyId order by genus;",result);

	    numNames = result.numRows();

	    if(numNames)
	    {
		names = new struct Data[numNames];

		for(int i = 0; i < numNames; ++i)
		{
		    strncpy(names[i].name,result(i,"genus").c_str(),1000);
		}
	    }
	}

	/*if(_conn)
	{
	    mysqlpp::Query q = _conn->query("select distinct genus from TaxonomyId order by genus;");
	    mysqlpp::StoreQueryResult res = q.store();

	    numNames = res.num_rows();

	    if(numNames)
	    {
		names = new struct Data[numNames];

		for(int i = 0; i < numNames; ++i)
		{
		    strncpy(names[i].name,res[i]["genus"].c_str(),1000);
		}
	    }
	}*/

	ComController::instance()->sendSlaves(&numNames,sizeof(int));
	if(numNames)
	{
	    ComController::instance()->sendSlaves(names,numNames*sizeof(struct Data));
	}
    }
    else
    {
	ComController::instance()->readMaster(&numNames,sizeof(int));
	if(numNames)
	{
	    names = new struct Data[numNames];
	    ComController::instance()->readMaster(names,numNames*sizeof(struct Data));
	}
    }

    std::vector<std::string> nameVec;
    for(int i = 0; i < numNames; ++i)
    {
	nameVec.push_back(names[i].name);
    }

    _strainGroupList->setValues(nameVec);

    if(names)
    {
	delete[] names;
    }

    for(int i = 0; i < nameVec.size(); ++i)
    {
	Data * names = NULL;
	int numNames = 0;

	if(ComController::instance()->isMaster())
	{
	    if(_dbm)
	    {
		std::stringstream queryss;
		queryss << "select description, taxonomy_id from TaxonomyId where genus = '" << nameVec[i] << "' order by description;";

		DBMQueryResult result;

		_dbm->runQuery(queryss.str(),result);

		numNames = result.numRows();

		if(numNames)
		{
		    names = new struct Data[numNames];

		    for(int j = 0; j < numNames; ++j)
		    {
			strncpy(names[j].name,result(j,"description").c_str(),1000);
			names[j].value = atoi(result(j,"taxonomy_id").c_str());
		    }
		}
	    }

	    /*if(_conn)
	    {
		std::stringstream queryss;
		queryss << "select description, taxonomy_id from TaxonomyId where genus = '" << nameVec[i] << "' order by description;";
		mysqlpp::Query q = _conn->query(queryss.str().c_str());
		mysqlpp::StoreQueryResult res = q.store();

		numNames = res.num_rows();

		if(numNames)
		{
		    names = new struct Data[numNames];

		    for(int j = 0; j < numNames; ++j)
		    {
			strncpy(names[j].name,res[j]["description"].c_str(),1000);
			names[j].value = atoi(res[j]["taxonomy_id"].c_str());
		    }
		}
	    }*/

	    ComController::instance()->sendSlaves(&numNames,sizeof(int));
	    if(numNames)
	    {
		ComController::instance()->sendSlaves(names,numNames*sizeof(struct Data));
	    }
	}
	else
	{
	    ComController::instance()->readMaster(&numNames,sizeof(int));
	    if(numNames)
	    {
		names = new struct Data[numNames];
		ComController::instance()->readMaster(names,numNames*sizeof(struct Data));
	    }
	}

	for(int j = 0; j < numNames; ++j)
	{
	    _strainGroupMap[nameVec[i]].push_back(names[j].name);
	    _strainIdMap[names[j].name] = names[j].value;
	}

	if(names)
	{
	    delete[] names;
	}	
    }

    if(_strainGroupMap.find(_strainGroupList->getValue()) != _strainGroupMap.end())
    {
	_strainList->setValues(_strainGroupMap[_strainGroupList->getValue()]);
    }
}

void FuturePatient::updateMicrobeTests(int patientid)
{
    std::vector<std::string> emptyvec;
    _microbeTest->setValues(emptyvec);

    _microbeTestTime.clear();

    if(_microbeTableList[_microbeTable->getIndex()]->testMap.find(patientid) != _microbeTableList[_microbeTable->getIndex()]->testMap.end())
    {
	    _microbeTest->setValues(_microbeTableList[_microbeTable->getIndex()]->testMap[patientid]);
	    _microbeTestTime = _microbeTableList[_microbeTable->getIndex()]->testTimeMap[patientid];
    }
}

void FuturePatient::saveLayout()
{
    bool ok = true;
    char file[1024];
    if(ComController::instance()->isMaster())
    {
	time_t now;
	time(&now);

	struct tm timeInfo;
	timeInfo = *localtime(&now);
	strftime(file,1024,"%Y_%m_%d_%H_%M_%S.cfg",&timeInfo);

	std::string outFile = _layoutDirectory + "/" + file;

	std::cerr << "Trying to save layout file: " << outFile << std::endl;

	if(_layoutObject)
	{
	    std::ofstream outstream(outFile.c_str(),std::ios_base::out | std::ios_base::trunc);

	    if(!outstream.fail())
	    {
		outstream << ((int)SAVED_LAYOUT_VERSION) << std::endl;

		_layoutObject->dumpState(outstream);

		outstream.close();
	    }
	    else
	    {
		std::cerr << "Failed to open file for writing: " << outFile << std::endl;
		ok = false;
	    }
	}
	else
	{
	    ok = false;
	}
	ComController::instance()->sendSlaves(&ok,sizeof(bool));
	if(ok)
	{
	    ComController::instance()->sendSlaves(file,1024*sizeof(char));
	}
    }
    else
    {
	ComController::instance()->readMaster(&ok,sizeof(bool));
	if(ok)
	{
	    ComController::instance()->readMaster(file,1024*sizeof(char));
	}
    }

    if(ok)
    {
	MenuButton * button = new MenuButton(file);
	button->setCallback(this);
	_loadLayoutButtons.push_back(button);

	_loadLayoutMenu->addItem(button);
    }
}

void FuturePatient::loadLayout(const std::string & file)
{
    std::string fullPath = _layoutDirectory + "/" + file;

    std::cerr << "Trying to load layout file: " << fullPath << std::endl;

    checkLayout();

    menuCallback(_removeAllButton);

    std::ifstream instream(fullPath.c_str());

    if(instream.fail())
    {
	std::cerr << "Unable to open layout file." << std::endl;
	return;
    }

    int version;
    instream >> version;

    if(version != SAVED_LAYOUT_VERSION)
    {
	std::cerr << "Error loading layout, version too old." << std::endl;
	instream.close();
	return;
    }

    _layoutObject->loadState(instream);

    instream.close();
}

bool phenoDispSort(const std::pair<PhenoStats*,struct SortCriteria> & first, const std::pair<PhenoStats*,struct SortCriteria> & second)
{
    return first.second.primaryValue > second.second.primaryValue;
}

bool phenoDispSortRev(const std::pair<PhenoStats*,struct SortCriteria> & first, const std::pair<PhenoStats*,struct SortCriteria> & second)
{
    return first.second.primaryValue < second.second.primaryValue;
}

bool phenoDispSortWithT(const std::pair<PhenoStats*,struct SortCriteria> & first, const std::pair<PhenoStats*,struct SortCriteria> & second)
{
    if(first.second.primaryValue != second.second.primaryValue)
    {
	return first.second.primaryValue < second.second.primaryValue;
    }
    else
    {
	//if(first.second.primaryValue == 0.0)
	//{
	//    std::cerr << "Tie breaker first t: " << first.second.secondaryValue << " second t: " << second.second.secondaryValue << std::endl;
	//}
	return first.second.secondaryValue > second.second.secondaryValue;
    }
}

void FuturePatient::loadScatter()
{
    std::vector<std::pair<PhenoStats*,SortCriteria> > displayList = createListWithFilters((MicrobeGraphType)_scatterMicrobeType->getIndex(),_scatterPhenotypes->getValue(),_scatterSecondPhenotype->getValue(),_scatterPvalSort->getValue(),_scatterTvalSort->getValue(),_scatterTvalSortType->getValue(),_scatterAvgEnable->getValue(),_scatterAvgValue->getValue(),_scatterReqMaxEnable->getValue(),_scatterReqMaxValue->getValue(),_scatterZerosEnable->getValue(),_scatterZerosValue->getValue());

    std::cerr << "Got " << displayList.size() << " graphs in display list." << std::endl;

    if(displayList.size() < 2)
    {
	return;
    }

    MicrobeGraphType mgt;
    switch(_scatterMicrobeType->getIndex())
    {
	case MGT_PHYLUM:
	{
	    mgt = MGT_PHYLUM;
	    break;
	}
	case MGT_FAMILY:
	{
	    //std::cerr << "Family pheno load." << std::endl;
	    mgt = MGT_FAMILY;
	    break;
	}
	case MGT_GENUS:
	{
	    mgt = MGT_GENUS;
	    break;
	}
	case MGT_SPECIES:
	default:
	{
	    //std::cerr << "Species pheno load." << std::endl;
	    mgt = MGT_SPECIES;
	    break;
	}
    }

    int graphs = std::min((int)displayList.size(),((int)_scatterSortResults->getValue()));

    GraphGlobals::setDeferUpdate(true);
    for(int i = 0; i < graphs; ++i)
    {
	for(int j = (graphs-1); j > i; --j)
	{
	    MicrobeScatterGraphObject * msgo = new MicrobeScatterGraphObject(_dbm, 1000.0, 1000.0, "Scatter Plot", false, true, false, true);
	    if(msgo->setGraph(displayList[i].first->name + " vs " + displayList[j].first->name,displayList[i].first->name,displayList[j].first->name,mgt,_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix))
	    {
		checkLayout();
		_layoutObject->addGraphObject(msgo);
		msgo->setLogScale(_scatterLogCB->getValue());
	    }
	    else
	    {
		delete msgo;
	    }
	}
    }
    GraphGlobals::setDeferUpdate(false);
    if(_layoutObject)
    {
	_layoutObject->forceUpdate();
    }
}

void FuturePatient::loadPhenotype()
{
    std::vector<std::pair<PhenoStats*,SortCriteria> > displayList = createListWithFilters((MicrobeGraphType)_sMicrobeType->getIndex(),_sMicrobePhenotypes->getValue(),_sMicrobeSecondPhenotype->getValue(),_sMicrobePvalSort->getValue(),_sMicrobeTvalSort->getValue(),_sMicrobeTvalSortType->getValue(),_sMicrobeAvgEnable->getValue(),_sMicrobeAvgValue->getValue(),_sMicrobeReqMaxEnable->getValue(),_sMicrobeReqMaxValue->getValue(),_sMicrobeZerosEnable->getValue(),_sMicrobeZerosValue->getValue());

    std::cerr << "Got " << displayList.size() << " graphs in display list." << std::endl;

    if(!displayList.size())
    {
	return;
    }

    MicrobeGraphType mgt;
    switch(_sMicrobeType->getIndex())
    {
	case MGT_PHYLUM:
	{
	    mgt = MGT_PHYLUM;
	    break;
	}
	case MGT_FAMILY:
	{
	    //std::cerr << "Family pheno load." << std::endl;
	    mgt = MGT_FAMILY;
	    break;
	}
	case MGT_GENUS:
	{
	    mgt = MGT_GENUS;
	    break;
	}
	case MGT_SPECIES:
	default:
	{
	    //std::cerr << "Species pheno load." << std::endl;
	    mgt = MGT_SPECIES;
	    break;
	}
    }

    GraphGlobals::setDeferUpdate(true);
    for(int i = 0; i < displayList.size() && i < ((int)_sMicrobeSortResults->getValue()); ++i)
    {
	SingleMicrobeObject * smo = new SingleMicrobeObject(_dbm, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);

	std::string titleSuffix;

	if(_sMicrobePvalSort->getValue())
	{
	    std::stringstream ss;
	    ss << " PVal: " << displayList[i].second.primaryValue;

	    if(_sMicrobeTvalSort->getValue())
	    {
		ss << " Min TVal: " << displayList[i].second.secondaryValue;
	    }

	    titleSuffix = ss.str();
	}

	if(smo->setGraph(displayList[i].first->name,titleSuffix,displayList[i].first->taxid,_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix, mgt,  _sMicrobeRankOrder->getValue(),_sMicrobeLabels->getValue(),_sMicrobeFirstTimeOnly->getValue(),_sMicrobeGroupPatients->getValue(),_sMicrobeGroup->getValue()))
	{
	    checkLayout();
	    _layoutObject->addGraphObject(smo);
	    smo->setLogScale(_sMicrobeLogCB->getValue());
	    smo->setShowStdDev(_sMicrobeStdDevCB->getValue());
	}
	else
	{
	    delete smo;
	}	
    }
    GraphGlobals::setDeferUpdate(false);
    if(_layoutObject)
    {
	_layoutObject->forceUpdate();
    }
}

std::vector<std::pair<PhenoStats*,SortCriteria> > FuturePatient::createListWithFilters(MicrobeGraphType type, std::string phenotype, std::string tphenotype, bool pvalSort, bool tvalSort, std::string tvalSortType, bool averageThresh, float avgVal, bool reqMax, float reqMaxVal, bool zeroLimit, float zeroVal)
{
    if(!_microbeTableList[_microbeTable->getIndex()]->statsMap.size())
    {
	initPhenoStats(_microbeTableList[_microbeTable->getIndex()]->statsMap,_microbeTableList[_microbeTable->getIndex()]->genusStatsMap,_microbeTableList[_microbeTable->getIndex()]->familyStatsMap,_microbeTableList[_microbeTable->getIndex()]->phylumStatsMap,_microbeTableList[_microbeTable->getIndex()]->microbeSuffix,_microbeTableList[_microbeTable->getIndex()]->measureSuffix);
    }

    std::vector<std::pair<PhenoStats*,SortCriteria> > displayList;

    if(tvalSort && tvalSortType == "Group Pair T" && phenotype == tphenotype)
    {
	return displayList;
    }

    std::map<std::string,std::map<std::string,struct PhenoStats > > * statsMapp;

    MicrobeGraphType mgt;
    switch(type)
    {
	case MGT_PHYLUM:
	{
	    mgt = MGT_PHYLUM;
	    statsMapp = &_microbeTableList[_microbeTable->getIndex()]->phylumStatsMap;
	    break;
	}
	case MGT_FAMILY:
	{
	    //std::cerr << "Family pheno load." << std::endl;
	    mgt = MGT_FAMILY;
	    statsMapp = &_microbeTableList[_microbeTable->getIndex()]->familyStatsMap;
	    break;
	}
	case MGT_GENUS:
	{
	    mgt = MGT_GENUS;
	    statsMapp = &_microbeTableList[_microbeTable->getIndex()]->genusStatsMap;
	    break;
	}
	case MGT_SPECIES:
	default:
	{
	    //std::cerr << "Species pheno load." << std::endl;
	    mgt = MGT_SPECIES;
	    statsMapp = &_microbeTableList[_microbeTable->getIndex()]->statsMap;
	    break;
	}
    }
    

    for(std::map<std::string,std::map<std::string,struct PhenoStats > >::iterator it = statsMapp->begin(); it != statsMapp->end(); ++it)
    {
	if(averageThresh || reqMax || zeroLimit)
	{
	    float avg = 0.0;
	    int count = 0;
	    int zeros = 0;
	    float maxVal = 0.0;
	    for(std::map<std::string,struct PhenoStats >::iterator itt = it->second.begin(); itt != it->second.end(); ++itt)
	    {
		for(int i = 0; i < itt->second.values.size(); ++i)
		{
		    avg += itt->second.values[i];
		    count++;
		    if(itt->second.values[i] == 0.0)
		    {
			zeros++;
		    }
		    if(itt->second.values[i] > maxVal)
		    {
			maxVal = itt->second.values[i];
		    }
		}
	    }
	    avg /= ((float)count);
	    float zeroRatio = ((float)zeros) / ((float)count);

	    if(averageThresh && avg < avgVal)
	    {
		continue;
	    }

	    if(reqMax && maxVal < reqMaxVal)
	    {
		continue;
	    }

	    if(zeroLimit && zeroRatio > zeroVal)
	    {
		continue;
	    }
	}

	if(!pvalSort)
	{
	    struct PhenoStats * ps = NULL;

	    for(std::map<std::string,struct PhenoStats >::iterator itt = it->second.begin(); itt != it->second.end(); ++itt)
	    {
		if(itt->first == phenotype)
		{
		    ps = &itt->second;
		    break;
		}
	    }

	    if(!ps)
	    {
		continue;
	    }

	    float refMax = ps->avg + ps->stdev;
	    float refMin = ps->avg - ps->stdev;

	    bool addGraph = true;
	    float minGap = FLT_MAX;
	    for(std::map<std::string,struct PhenoStats >::iterator itt = it->second.begin(); itt != it->second.end(); ++itt)
	    {
		if(itt->first == phenotype)
		{
		    continue;
		}

		float localMax = itt->second.avg + itt->second.stdev;
		float localMin = itt->second.avg - itt->second.stdev;
		if(refMax < localMin)
		{
		    minGap = std::min(minGap,localMin-refMax);
		}
		else if(refMin > localMax)
		{
		    minGap = std::min(minGap,refMin-localMax);
		}
		else
		{
		    addGraph = false;
		    break;
		}
	    }
	    if(addGraph)
	    {
		struct SortCriteria sc;
		sc.primaryValue = minGap;
		displayList.push_back(std::pair<PhenoStats*,struct SortCriteria>(ps,sc));
	    }
	}
	else
	{
	    int groupIndex = 0;
	    std::vector<float> inVal;
	    std::vector<int> inGrp;

	    for(std::map<std::string,struct PhenoStats >::iterator itt = it->second.begin(); itt != it->second.end(); ++itt)
	    {
		for(int i = 0; i < itt->second.values.size(); ++i)
		{
		    inGrp.push_back(groupIndex);
		    inVal.push_back(itt->second.values[i]);
		}
		groupIndex++;
	    }

	    std::map<std::string,struct PhenoStats >::iterator baseStats;

	    baseStats = it->second.find(phenotype);

	    float tval = 0.0;
	    int count = 0;

	    if(baseStats != it->second.end())
	    {
		if(tvalSortType == "Group Avg Dev")
		{
		    float avgavg = 0;
		    float acount = 0;

		    for(std::map<std::string,struct PhenoStats >::iterator itt = it->second.begin(); itt != it->second.end(); ++itt)
		    {
			avgavg += itt->second.avg;
			acount += 1.0;
		    }

		    if(acount > 0.0)
		    {
			avgavg /= acount;

			float dev = 0.0;
			for(std::map<std::string,struct PhenoStats >::iterator itt = it->second.begin(); itt != it->second.end(); ++itt)
			{
			    dev += pow(avgavg - itt->second.avg,2.0);
			}
			dev /+ acount;
			tval = sqrt(dev);
		    }
		}
		else
		{
		    for(std::map<std::string,struct PhenoStats >::iterator itt = it->second.begin(); itt != it->second.end(); ++itt)
		    {
			if(itt->first == phenotype)
			{
			    continue;
			}
			if(tvalSortType == "Group Pair T" && itt->first != tphenotype)
			{
			    continue;
			}

			float tempTval = fabs(baseStats->second.avg - itt->second.avg);
			float denom = ((baseStats->second.stdev*baseStats->second.stdev) / baseStats->second.values.size() + (itt->second.stdev*itt->second.stdev) / itt->second.values.size());

			if(denom <= 0.0)
			{
			    continue;
			}

			tempTval /= sqrt(denom);

			if(tvalSortType == "Avg Group T")
			{
			    tval += tempTval;
			    count++;
			}
			else
			{
			    if(tval == 0.0 || tempTval < tval)
			    {
				tval = tempTval;
			    }
			}
		    }
		}
		if(tval == 0.0)
		{
		    tval = FLT_MAX;
		}
	    }

	    if(tvalSortType == "Avg Group T")
	    {
		tval /= ((float)count);
	    }

	    if(inVal.size())
	    {
		Matrix valMat(inVal.size(),1);
		Matrix grpMat(inVal.size(),1);

		for(int i = 0; i < inVal.size(); ++i)
		{
		    valMat(i) = inVal[i];
		    grpMat(i) = inGrp[i];
		}

		octave_value_list mList;
		mList(0) = valMat;
		mList(1) = grpMat;

		octave_value_list out = feval("anova",mList);

		float pval = out(0).float_value();
		if(std::isnan(pval))
		{
		}
		else
		{
		    struct SortCriteria sc;
		    sc.primaryValue = pval;
		    sc.secondaryValue = tval;
		    displayList.push_back(std::pair<PhenoStats*,struct SortCriteria>(&it->second.begin()->second,sc));
		}
	    }
	}
    }
    
    if(!pvalSort)
    {
	std::sort(displayList.begin(),displayList.end(),phenoDispSort);
    }
    else if(!tvalSort)
    {
	std::sort(displayList.begin(),displayList.end(),phenoDispSortRev);
    }
    else
    {
	std::sort(displayList.begin(),displayList.end(),phenoDispSortWithT);
    }

    return displayList;
}

void FuturePatient::initPhenoStats(std::map<std::string,std::map<std::string,struct PhenoStats > > & statMap, std::map<std::string,std::map<std::string,struct PhenoStats > > & genusStatMap, std::map<std::string,std::map<std::string,struct PhenoStats > > & familyStatMap, std::map<std::string,std::map<std::string,struct PhenoStats > > & phylumStatMap, std::string microbeSuffix, std::string measureSuffix)
{
    struct entry
    {
	char name[512];
	char genus[512];
	char family[512];
	char phylum[512];
	char patientName[512];
	int taxid;
	float value;
	time_t timestamp;
    };

    struct entry * entries = NULL;
    int numEntries[4];
    numEntries[0] = numEntries[1] = numEntries[2] = numEntries[3] = 0;
    int totalEntries = 0;

    std::string measurementTable = "Microbe_Measurement";
    measurementTable += measureSuffix;

    std::string microbesTable = "Microbes";
    microbesTable += microbeSuffix;

    std::stringstream queryhss, querycss, queryuss, querylss;
    queryhss << "select " << microbesTable << ".species, " << microbesTable << ".genus, " << microbesTable << ".family, " << microbesTable << ".phylum, t.last_name, t.taxonomy_id, t.timestamp, t.value from (select Patient.last_name, " << measurementTable << ".taxonomy_id, unix_timestamp(" << measurementTable << ".timestamp) as timestamp, " << measurementTable << ".value from " << measurementTable << " inner join Patient on Patient.patient_id = " << measurementTable << ".patient_id where Patient.p_condition = \"healthy\" and Patient.region = \"US\")t inner join " << microbesTable << " on t.taxonomy_id = " << microbesTable << ".taxonomy_id;";

    querycss << "select " << microbesTable << ".species, " << microbesTable << ".genus, " << microbesTable << ".family, " << microbesTable << ".phylum, t.last_name, t.taxonomy_id, t.timestamp, t.value from (select Patient.last_name, " << measurementTable << ".taxonomy_id, unix_timestamp(" << measurementTable << ".timestamp) as timestamp, " << measurementTable << ".value from " << measurementTable << " inner join Patient on Patient.patient_id = " << measurementTable << ".patient_id where Patient.p_condition = \"crohn's disease\" and Patient.region = \"US\")t inner join " << microbesTable << " on t.taxonomy_id = " << microbesTable << ".taxonomy_id;";

    queryuss << "select " << microbesTable << ".species, " << microbesTable << ".genus, " << microbesTable << ".family, " << microbesTable << ".phylum, t.last_name, t.taxonomy_id, t.timestamp, t.value from (select Patient.last_name, " << measurementTable << ".taxonomy_id, unix_timestamp(" << measurementTable << ".timestamp) as timestamp, " << measurementTable << ".value from " << measurementTable << " inner join Patient on Patient.patient_id = " << measurementTable << ".patient_id where Patient.p_condition = \"ulcerous colitis\" and Patient.region = \"US\")t inner join " << microbesTable << " on t.taxonomy_id = " << microbesTable << ".taxonomy_id;";

    querylss << "select " << microbesTable << ".species, " << microbesTable << ".genus, " << microbesTable << ".family, " << microbesTable << ".phylum, t.last_name, t.taxonomy_id, t.timestamp, t.value from (select Patient.last_name, " << measurementTable << ".taxonomy_id, unix_timestamp(" << measurementTable << ".timestamp) as timestamp, " << measurementTable << ".value from " << measurementTable << " inner join Patient on Patient.patient_id = " << measurementTable << ".patient_id where Patient.p_condition = \"Larry\")t inner join " << microbesTable << " on t.taxonomy_id = " << microbesTable << ".taxonomy_id;";

    //std::cerr << "Larry query: " << querylss.str() << std::endl;

    if(ComController::instance()->isMaster())
    {
	if(_dbm)
	{
	    DBMQueryResult res[4];

	    _dbm->runQuery(queryhss.str(),res[0]);
	    _dbm->runQuery(querycss.str(),res[1]);
	    _dbm->runQuery(queryuss.str(),res[2]);
	    _dbm->runQuery(querylss.str(),res[3]);

	    for(int i = 0; i < 4; ++i)
	    {
		numEntries[i] = res[i].numRows();
	    }

	    totalEntries += numEntries[0] + numEntries[1] + numEntries[2] + numEntries[3];

	    if(totalEntries)
	    {
		entries = new struct entry[totalEntries];
		int entryIndex = 0;

		for(int i = 0; i < 4; ++i)
		{
		    for(int j = 0; j < numEntries[i]; ++j)
		    {
			entries[entryIndex+j].name[511] = '\0';
			strncpy(entries[entryIndex+j].name,res[i](j,"species").c_str(),511);
			entries[entryIndex+j].genus[511] = '\0';
			strncpy(entries[entryIndex+j].genus,res[i](j,"genus").c_str(),511);
			entries[entryIndex+j].family[511] = '\0';
			strncpy(entries[entryIndex+j].family,res[i](j,"family").c_str(),511);
			entries[entryIndex+j].phylum[511] = '\0';
			strncpy(entries[entryIndex+j].phylum,res[i](j,"phylum").c_str(),511);
			entries[entryIndex+j].patientName[511] = '\0';
			strncpy(entries[entryIndex+j].patientName,res[i](j,"last_name").c_str(),511);
			entries[entryIndex+j].taxid = atoi(res[i](j,"taxonomy_id").c_str());
			entries[entryIndex+j].value = atof(res[i](j,"value").c_str());
			entries[entryIndex+j].timestamp = atol(res[i](j,"timestamp").c_str());
		    }
		    entryIndex += numEntries[i];
		}
	    }
	}

	/*if(_conn)
	{
	    mysqlpp::StoreQueryResult res[4];

	    mysqlpp::Query queryh = _conn->query(queryhss.str().c_str());
	    res[0] = queryh.store();

	    mysqlpp::Query queryc = _conn->query(querycss.str().c_str());
	    res[1] = queryc.store();

	    mysqlpp::Query queryu = _conn->query(queryuss.str().c_str());
	    res[2] = queryu.store();

	    mysqlpp::Query queryl = _conn->query(querylss.str().c_str());
	    res[3] = queryl.store();

	    for(int i = 0; i < 4; ++i)
	    {
		numEntries[i] = res[i].num_rows();
	    }

	    totalEntries += numEntries[0] + numEntries[1] + numEntries[2] + numEntries[3];

	    if(totalEntries)
	    {
		entries = new struct entry[totalEntries];
		int entryIndex = 0;

		for(int i = 0; i < 4; ++i)
		{
		    for(int j = 0; j < numEntries[i]; ++j)
		    {
			entries[entryIndex+j].name[511] = '\0';
			strncpy(entries[entryIndex+j].name,res[i][j]["species"].c_str(),511);
			entries[entryIndex+j].family[511] = '\0';
			strncpy(entries[entryIndex+j].family,res[i][j]["family"].c_str(),511);
			entries[entryIndex+j].patientName[511] = '\0';
			strncpy(entries[entryIndex+j].patientName,res[i][j]["last_name"].c_str(),511);
			entries[entryIndex+j].taxid = atoi(res[i][j]["taxonomy_id"].c_str());
			entries[entryIndex+j].value = atof(res[i][j]["value"].c_str());
		    }
		    entryIndex += numEntries[i];
		}
	    }
	}*/

	ComController::instance()->sendSlaves(&totalEntries,sizeof(int));
	ComController::instance()->sendSlaves(numEntries,4*sizeof(int));
	if(totalEntries)
	{
	    ComController::instance()->sendSlaves(entries,totalEntries*sizeof(struct entry));
	}
    }
    else
    {
	ComController::instance()->readMaster(&totalEntries,sizeof(int));
	ComController::instance()->readMaster(numEntries,4*sizeof(int));
	if(totalEntries)
	{
	    entries = new struct entry[totalEntries];
	    ComController::instance()->readMaster(entries,totalEntries*sizeof(struct entry));
	}
    }

    std::vector<std::string> groupLabels;
    groupLabels.push_back("Healthy");
    groupLabels.push_back("Crohns");
    groupLabels.push_back("UC");
    groupLabels.push_back("Smarr");

    int entryIndex = 0;
    for(int i = 0; i < 4; ++i)
    {
	std::cerr << "NumEntries " << i << ": " << numEntries[i] << std::endl;
	std::map<std::string,int> countMap;
	std::map<std::string, std::map<std::string,float> > genusMap;
	std::map<std::string, std::map<std::string,float> > familyMap;
	std::map<std::string, std::map<std::string,float> > phylumMap;
	for(int j = 0; j < numEntries[i]; ++j)
	{
	    int index = entryIndex + j;
	    statMap[entries[index].name][groupLabels[i]].avg += entries[index].value;
	    countMap[entries[index].name]++;
	    statMap[entries[index].name][groupLabels[i]].taxid = entries[index].taxid;
	    statMap[entries[index].name][groupLabels[i]].name = entries[index].name;
	    statMap[entries[index].name][groupLabels[i]].values.push_back(entries[index].value);

	    struct tm timetm = *localtime(&entries[index].timestamp);
	    char timestr[256];
	    timestr[255] = '\0';
	    strftime(timestr, 255, "%F", &timetm);

	    std::string pname = std::string(entries[index].patientName) + " - " + timestr;

	    genusMap[entries[index].genus][pname] += entries[index].value;
	    familyMap[entries[index].family][pname] += entries[index].value;
	    phylumMap[entries[index].phylum][pname] += entries[index].value;
	}

	for(std::map<std::string,int>::iterator it = countMap.begin(); it != countMap.end(); ++it)
	{
	    statMap[it->first][groupLabels[i]].avg /= ((float)it->second);
	}

	for(std::map<std::string, std::map<std::string,float> >::iterator it = genusMap.begin(); it != genusMap.end(); ++it)
	{
	    genusStatMap[it->first][groupLabels[i]].name = it->first;
	    for(std::map<std::string,float>::iterator itt = it->second.begin(); itt != it->second.end(); ++itt)
	    {
		genusStatMap[it->first][groupLabels[i]].avg += itt->second;
		genusStatMap[it->first][groupLabels[i]].values.push_back(itt->second);
	    }
	    genusStatMap[it->first][groupLabels[i]].avg /= ((float)it->second.size());
	    //std::cerr << "Gen: " << it->first << " group label: " << groupLabels[i] << " avg: " << genusStatMap[it->first][groupLabels[i]].avg << " size: " << it->second.size() << std::endl;

	    for(int j = 0; j < genusStatMap[it->first][groupLabels[i]].values.size(); ++j)
	    {
		float val = genusStatMap[it->first][groupLabels[i]].values[j] - genusStatMap[it->first][groupLabels[i]].avg;
		val *= val;
		genusStatMap[it->first][groupLabels[i]].stdev += val;
	    }
	    genusStatMap[it->first][groupLabels[i]].stdev = sqrt(genusStatMap[it->first][groupLabels[i]].stdev / ((float)it->second.size()));
	}

	for(std::map<std::string, std::map<std::string,float> >::iterator it = familyMap.begin(); it != familyMap.end(); ++it)
	{
	    familyStatMap[it->first][groupLabels[i]].name = it->first;
	    for(std::map<std::string,float>::iterator itt = it->second.begin(); itt != it->second.end(); ++itt)
	    {
		familyStatMap[it->first][groupLabels[i]].avg += itt->second;
		familyStatMap[it->first][groupLabels[i]].values.push_back(itt->second);
	    }
	    familyStatMap[it->first][groupLabels[i]].avg /= ((float)it->second.size());
	    //std::cerr << "Fam: " << it->first << " group label: " << groupLabels[i] << " avg: " << familyStatMap[it->first][groupLabels[i]].avg << " size: " << it->second.size() << std::endl;

	    for(int j = 0; j < familyStatMap[it->first][groupLabels[i]].values.size(); ++j)
	    {
		float val = familyStatMap[it->first][groupLabels[i]].values[j] - familyStatMap[it->first][groupLabels[i]].avg;
		val *= val;
		familyStatMap[it->first][groupLabels[i]].stdev += val;
	    }
	    familyStatMap[it->first][groupLabels[i]].stdev = sqrt(familyStatMap[it->first][groupLabels[i]].stdev / ((float)it->second.size()));
	}

	for(std::map<std::string, std::map<std::string,float> >::iterator it = phylumMap.begin(); it != phylumMap.end(); ++it)
	{
	    phylumStatMap[it->first][groupLabels[i]].name = it->first;
	    for(std::map<std::string,float>::iterator itt = it->second.begin(); itt != it->second.end(); ++itt)
	    {
		phylumStatMap[it->first][groupLabels[i]].avg += itt->second;
		phylumStatMap[it->first][groupLabels[i]].values.push_back(itt->second);
	    }
	    phylumStatMap[it->first][groupLabels[i]].avg /= ((float)it->second.size());
	    //std::cerr << "Phy: " << it->first << " group label: " << groupLabels[i] << " avg: " << phylumStatMap[it->first][groupLabels[i]].avg << " size: " << it->second.size() << std::endl;

	    for(int j = 0; j < phylumStatMap[it->first][groupLabels[i]].values.size(); ++j)
	    {
		float val = phylumStatMap[it->first][groupLabels[i]].values[j] - phylumStatMap[it->first][groupLabels[i]].avg;
		val *= val;
		phylumStatMap[it->first][groupLabels[i]].stdev += val;
	    }
	    phylumStatMap[it->first][groupLabels[i]].stdev = sqrt(phylumStatMap[it->first][groupLabels[i]].stdev / ((float)it->second.size()));
	}

	for(int j = 0; j < numEntries[i]; ++j)
	{
	    int index = entryIndex + j;
	    float val = entries[index].value - statMap[entries[index].name][groupLabels[i]].avg;
	    val *= val;
	    statMap[entries[index].name][groupLabels[i]].stdev += val;
	}

	for(std::map<std::string,int>::iterator it = countMap.begin(); it != countMap.end(); ++it)
	{
	    statMap[it->first][groupLabels[i]].stdev = sqrt(statMap[it->first][groupLabels[i]].stdev / ((float)it->second));
	}

	entryIndex += numEntries[i];
    }

    if(entries)
    {
	delete[] entries;
    }
}

void FuturePatient::loadScatterPresets()
{
    DIR * dir = opendir(_scatterPresetDir.c_str());
    if(!dir)
    {
	std::cerr << "Unable to open scatter preset director: " << _scatterPresetDir << std::endl;
	return;
    }

    struct dirent * ent;
    struct stat fstat;
    while((ent = readdir(dir)))
    {
	std::string path = _scatterPresetDir + "/" + ent->d_name;

	if(stat(path.c_str(),&fstat) || S_ISDIR(fstat.st_mode))
	{
	    continue;
	}

	std::vector<std::string> entryList;

	std::ifstream infile;
	infile.open(path.c_str());
	while(infile.good())
	{
	    std::string entry;
	    std::getline(infile,entry);
	    if(!entry.empty())
	    {
		//std::cerr << "Entry: " << entry << std::endl;
		entryList.push_back(entry);
	    }
	}
	infile.close();

	if(entryList.size())
	{
	    MenuButton * button = new MenuButton(ent->d_name,false);
	    button->setCallback(this);
	    _scatterPresetMenu->addItem(button);
	    _scatterPresetButtons.push_back(button);
	    _scatterPresets.push_back(entryList);
	}
    }
    closedir(dir);
}

void FuturePatient::takeSubImage()
{
    if(_takeSubImage)
    {
	return;
    }

    if(ComController::instance()->isMaster())
    {
	_takeSubImage = true;
	_subImageDone = false;

	dynamic_cast<osg::Group*>(CVRViewer::instance()->getSceneData())->addChild(_subCamera);

	//SceneManager::instance()->getScene()->addChild(_subCamera);

	osg::Vec3 center(2000,0,0);
	center = center * SceneManager::instance()->getTiledWallTransform();
	setSubImageParams(center,1500,1500);

    }
}

void FuturePatient::setSubImageParams(osg::Vec3 pos, float width, float height)
{
    osg::Matrix centerTrans;
    centerTrans.makeTranslate(-pos);
    osg::Vec3 camPos = PluginHelper::getHeadMat(0).getTrans();
    camPos = camPos * centerTrans;
    osg::Matrix camTrans;
    camTrans.makeTranslate(-camPos);
    
    osg::Matrix view = centerTrans * camTrans * osg::Matrix::lookAt(osg::Vec3(0,0,0),osg::Vec3(0,1,0),osg::Vec3(0,0,1));

    float top, bottom, left, right;
    float screenDist = -camPos.y();

    top = ScreenBase::getNear() * (height / 2.0 - camPos.z()) / screenDist;
    bottom = ScreenBase::getNear() * (-height / 2.0 - camPos.z()) / screenDist;
    right = ScreenBase::getNear() * (width / 2.0 - camPos.x()) / screenDist;
    left = ScreenBase::getNear() * (-width / 2.0 - camPos.x()) / screenDist;

    osg::Matrix proj;
    proj.makeFrustum(left,right,bottom,top,ScreenBase::getNear(),ScreenBase::getFar());

    _subCamera->setViewMatrix(view);
    _subCamera->setProjectionMatrix(proj);

    CVRViewer::Contexts contexts;
    CVRViewer::instance()->getContexts(contexts);

    for(CVRViewer::Contexts::iterator citr = contexts.begin(); citr != contexts.end(); ++citr)
    {
	osgViewer::GraphicsWindow* gw = dynamic_cast<osgViewer::GraphicsWindow*>(*citr);
	if(gw)
	{
	    osg::GraphicsContext::Cameras& cameras =
                                        gw->getCameras();

	    for(osg::GraphicsContext::Cameras::iterator citr =
		    cameras.begin(); citr != cameras.end();
		    ++citr)
	    {
		std::cerr << "Found camera" << std::endl;
		//_subCamera->setViewMatrix((*citr)->getViewMatrix());
		//_subCamera->setProjectionMatrix((*citr)->getProjectionMatrix());
	    }
	}
    }
}
