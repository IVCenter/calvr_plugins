#include "FuturePatient.h"
#include "DataGraph.h"
#include "StrainGraphObject.h"
#include "StrainHMObject.h"

#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/ComController.h>
#include <cvrConfig/ConfigManager.h>

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <sstream>
#include <fstream>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <mysql++/mysql++.h>

#include <sys/time.h>

#define SAVED_LAYOUT_VERSION 3

using namespace cvr;

CVRPLUGIN(FuturePatient)

mysqlpp::Connection * FuturePatient::_conn = NULL;

FuturePatient::FuturePatient()
{
    _layoutObject = NULL;
    _multiObject = NULL;
    _currentSBGraph = NULL;
    _currentSymptomGraph = NULL;
    _mls = NULL;
}

FuturePatient::~FuturePatient()
{
    if(_mls)
    {
        delete _mls;
    }
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
    _chartMenu->addItem(_testList);

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

    _microbeLoadAverage = new MenuButton("UC Average");
    _microbeLoadAverage->setCallback(this);
    _microbeSpecialMenu->addItem(_microbeLoadAverage);

    _microbeLoadHealthyAverage = new MenuButton("Healthy Average");
    _microbeLoadHealthyAverage->setCallback(this);
    _microbeSpecialMenu->addItem(_microbeLoadHealthyAverage);

    _microbeLoadCrohnsAverage = new MenuButton("Crohns Average");
    _microbeLoadCrohnsAverage->setCallback(this);
    _microbeSpecialMenu->addItem(_microbeLoadCrohnsAverage);

    _microbeLoadSRSAverage = new MenuButton("SRS Average");
    _microbeLoadSRSAverage->setCallback(this);
    //_microbeSpecialMenu->addItem(_microbeLoadSRSAverage);

    _microbeLoadSRXAverage = new MenuButton("SRX Average");
    _microbeLoadSRXAverage->setCallback(this);
    //_microbeSpecialMenu->addItem(_microbeLoadSRXAverage);

    _microbeLoadCrohnsAll = new MenuButton("Crohns All");
    _microbeLoadCrohnsAll->setCallback(this);
    _microbeSpecialMenu->addItem(_microbeLoadCrohnsAll);

    _microbeLoadHealthyAll = new MenuButton("Healthy All");
    _microbeLoadHealthyAll->setCallback(this);
    _microbeSpecialMenu->addItem(_microbeLoadHealthyAll);

    _microbeLoadHealthy105All = new MenuButton("Healthy 119");
    _microbeLoadHealthy105All->setCallback(this);
    _microbeSpecialMenu->addItem(_microbeLoadHealthy105All);

    _microbeLoadUCAll = new MenuButton("UC All");
    _microbeLoadUCAll->setCallback(this);
    _microbeSpecialMenu->addItem(_microbeLoadUCAll);

    _microbeLoadPointLine = new MenuButton("Point Line Graph");
    _microbeLoadPointLine->setCallback(this);
    _microbeSpecialMenu->addItem(_microbeLoadPointLine);

    _microbeGraphType = new MenuList();
    _microbeGraphType->setCallback(this);
    _microbeMenu->addItem(_microbeGraphType);

    std::vector<std::string> mGraphTypes;
    mGraphTypes.push_back("Bar Graph");
    mGraphTypes.push_back("Stacked Bar Graph");
    _microbeGraphType->setValues(mGraphTypes);

    _microbePatients = new MenuList();
    _microbePatients->setCallback(this);
    _microbeMenu->addItem(_microbePatients);

    _microbeTest = new MenuList();
    _microbeTest->setCallback(this);
    _microbeMenu->addItem(_microbeTest);
    _microbeTest->setSensitivity(2.0);

    //_microbeNumBars = new MenuRangeValueCompact("Microbes",1,100,25);
    _microbeNumBars = new MenuRangeValueCompact("Microbes",1,2330,25);
    _microbeNumBars->setCallback(this);
    _microbeMenu->addItem(_microbeNumBars);

    _microbeOrdering = new MenuCheckbox("LS Ordering", true);
    _microbeOrdering->setCallback(this);
    _microbeMenu->addItem(_microbeOrdering);

    _microbeGrouping = new MenuCheckbox("Group",true);
    _microbeGrouping->setCallback(this);
    _microbeMenu->addItem(_microbeGrouping);

    _microbeLoad = new MenuButton("Load");
    _microbeLoad->setCallback(this);
    _microbeMenu->addItem(_microbeLoad);

    _microbeDone = new MenuButton("Done");
    _microbeDone->setCallback(this);

    _strainMenu = new SubMenu("Strains");
    _fpMenu->addItem(_strainMenu);

    _strainGroupList = new MenuList();
    _strainGroupList->setCallback(this);
    _strainMenu->addItem(_strainGroupList);

    _strainList = new MenuList();
    _strainList->setCallback(this);
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
    _eventMenu->addItem(_eventName);

    _eventLoad = new MenuButton("Load");
    _eventLoad->setCallback(this);
    _eventMenu->addItem(_eventLoad);

    _eventLoadAll = new MenuButton("Load All");
    _eventLoadAll->setCallback(this);
    _eventMenu->addItem(_eventLoadAll);

    _eventDone = new MenuButton("Done");
    _eventDone->setCallback(this);

    _scatterMenu = new SubMenu("Scatter Plots");
    _fpMenu->addItem(_scatterMenu);

    _scatterFirstLabel = new MenuText("Primary Phylum:",1.0,false);
    _scatterMenu->addItem(_scatterFirstLabel);

    _scatterFirstList = new MenuList();
    _scatterMenu->addItem(_scatterFirstList);

    _scatterSecondLabel = new MenuText("Secondary Phylum:",1.0,false);
    _scatterMenu->addItem(_scatterSecondLabel);

    _scatterSecondList = new MenuList();
    _scatterMenu->addItem(_scatterSecondList);

    _scatterLoad = new MenuButton("Load");
    _scatterLoad->setCallback(this);
    _scatterMenu->addItem(_scatterLoad);

    _scatterLoadAll = new MenuButton("Load All");
    _scatterLoadAll->setCallback(this);
    _scatterMenu->addItem(_scatterLoadAll);

    _removeAllButton = new MenuButton("Remove All");
    _removeAllButton->setCallback(this);
    _fpMenu->addItem(_removeAllButton);

    _closeLayoutButton = new MenuButton("Close Layout");
    _closeLayoutButton->setCallback(this);

    PluginHelper::addRootMenuItem(_fpMenu);

    struct listField
    {
	char entry[256];
    };

    struct listField * lfList = NULL;
    int listEntries = 0;
    int * sizes = NULL;
    listField ** groupLists = NULL;

    if(ComController::instance()->isMaster())
    {
	if(!_conn)
	{
	    _conn = new mysqlpp::Connection(false);
	    if(!_conn->connect("futurepatient","palmsdev2.ucsd.edu","fpuser","FPp@ssw0rd"))
	    {
		std::cerr << "Unable to connect to database." << std::endl;
		delete _conn;
		_conn = NULL;
	    }
	}

	if(_conn)
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
	}

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
	if(_conn)
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
	}

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
	if(_conn)
	{
	    mysqlpp::Query q = _conn->query("select distinct name from Event where patient_id = \"1\" order by name;");
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
	if(_conn)
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
	}

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

    _scatterFirstList->setValues(stringlist);
    _scatterSecondList->setValues(stringlist);

    if(lfList)
    {
	delete[] lfList;
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
    if(item == _loadButton)
    {
	loadGraph(_chartPatientList->getValue(),_testList->getValue());
	/*std::string value = _testList->getValue();
	if(!value.empty())
	{
	    if(_graphObjectMap.find(value) == _graphObjectMap.end())
	    {
		GraphObject * gobject = new GraphObject(_conn, 1000.0, 1000.0, "DataGraph", false, true, false, true, false);
		if(gobject->addGraph(value))
		{
		    _graphObjectMap[value] = gobject;
		}
		else
		{
		    delete gobject;
		}
	    }

	    if(_graphObjectMap.find(value) != _graphObjectMap.end())
	    {
		if(!_layoutObject)
		{
		    float width, height;
		    osg::Vec3 pos;
		    width = ConfigManager::getFloat("width","Plugin.FuturePatient.Layout",1500.0);
		    height = ConfigManager::getFloat("height","Plugin.FuturePatient.Layout",1000.0);
		    pos = ConfigManager::getVec3("Plugin.FuturePatient.Layout");
		    _layoutObject = new GraphLayoutObject(width,height,3,"GraphLayout",false,true,false,true,false);
		    _layoutObject->setPosition(pos);
		    PluginHelper::registerSceneObject(_layoutObject,"FuturePatient");
		    _layoutObject->attachToScene();
		}

		_layoutObject->addGraphObject(_graphObjectMap[value]);
	    }
	}*/
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
	    loadGraph("Smarr",_patientTestMap["Smarr"][i],true);
	}
	GraphGlobals::setDeferUpdate(false);
	if(_layoutObject)
	{
	    _layoutObject->forceUpdate();
	}
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

    if(item == _microbePatients)
    {
	if(_microbePatients->getListSize())
	{
	    updateMicrobeTests(_microbePatients->getIndex() + 1);
	}
    }

    if(item == _microbeLoad && _microbePatients->getListSize() && _microbeTest->getListSize())
    {
	// Bar Graph
	if(_microbeGraphType->getIndex() == 0)
	{
	    MicrobeGraphObject * mgo = new MicrobeGraphObject(_conn, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);
	    if(mgo->setGraph(_microbePatients->getValue(), _microbePatients->getIndex()+1, _microbeTest->getValue(), _microbeTestTime[_microbeTest->getIndex()],(int)_microbeNumBars->getValue(),_microbeGrouping->getValue(),_microbeOrdering->getValue()))
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
		_currentSBGraph = new MicrobeBarGraphObject(_conn, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);
		_currentSBGraph->addGraph(_microbePatients->getValue(), _microbePatients->getIndex()+1, _microbeTest->getValue());
		checkLayout();
		_layoutObject->addGraphObject(_currentSBGraph);
		_microbeMenu->addItem(_microbeDone);
	    }
	    else
	    {
		_currentSBGraph->addGraph(_microbePatients->getValue(), _microbePatients->getIndex()+1, _microbeTest->getValue());
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
	return;
    }

    if(item == _microbeLoadCrohnsAll)
    {
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
		std::map<int,std::vector<std::string> >::iterator it = _patientMicrobeTestMap.find(j+1);
		if(it == _patientMicrobeTestMap.end())
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
		std::map<int,std::vector<std::string> >::iterator it = _patientMicrobeTestMap.find(j+1);
		if(it == _patientMicrobeTestMap.end())
		{
		    continue;
		}

		float bgColor = ((float)(j - rangeList[i].first)) / ((float)(rows-1));
		bgColor = (1.0 - bgColor) * bgLight + bgColor * bgDark;

		for(int k = 0; k < it->second.size(); ++k)
		{
		    if(_microbeGraphType->getIndex() == 0)
		    {
			MicrobeGraphObject * mgo = new MicrobeGraphObject(_conn, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);
			bool tb = mgo->setGraph(_microbePatients->getValue(j), j+1, it->second[k], _patientMicrobeTestTimeMap[it->first][k],(int)_microbeNumBars->getValue(),_microbeGrouping->getValue(),_microbeOrdering->getValue());
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
			    _currentSBGraph = new MicrobeBarGraphObject(_conn, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);
			    _currentSBGraph->addGraph(_microbePatients->getValue(j), j+1, it->second[k]);
			    checkLayout();
			    _layoutObject->addGraphObject(_currentSBGraph);
			    _microbeMenu->addItem(_microbeDone);
			}
			else
			{
			    _currentSBGraph->addGraph(_microbePatients->getValue(j), j+1, it->second[k]);
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

    if(item == _microbeLoadHealthyAll || item == _microbeLoadUCAll || item == _microbeLoadHealthy105All)
    {
	std::vector<std::pair<int,int> > rangeList;

	if(item == _microbeLoadCrohnsAll)
	{
	    rangeList.push_back(std::pair<int,int>(44,58));
	}
	else if(item == _microbeLoadHealthyAll)
	{
	    rangeList.push_back(std::pair<int,int>(65,99));
	}
	else if(item == _microbeLoadHealthy105All)
	{
	    rangeList.push_back(std::pair<int,int>(118,236));
	}
	else
	{
	    rangeList.push_back(std::pair<int,int>(59,64));
	}

	struct timeval tstart,tend;
	gettimeofday(&tstart,NULL);

	GraphGlobals::setDeferUpdate(true);
	for(int i = 0; i < rangeList.size(); ++i)
	{
	    int start = rangeList[i].first;
	    while(start <= rangeList[i].second)
	    {
		std::cerr << "Loading graph " << start << std::endl;
		updateMicrobeTests(start + 1);
		if(_microbeGraphType->getIndex() == 0)
		{
		    MicrobeGraphObject * mgo = new MicrobeGraphObject(_conn, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);
		    bool tb = mgo->setGraph(_microbePatients->getValue(start), start+1, _microbeTest->getValue(0), _microbeTestTime[0],(int)_microbeNumBars->getValue(),_microbeGrouping->getValue(),_microbeOrdering->getValue());
		    if(tb)
		    {
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
			_currentSBGraph = new MicrobeBarGraphObject(_conn, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);
			_currentSBGraph->addGraph(_microbePatients->getValue(start), start+1, _microbeTest->getValue(0));
			checkLayout();
			_layoutObject->addGraphObject(_currentSBGraph);
			_microbeMenu->addItem(_microbeDone);
		    }
		    else
		    {
			_currentSBGraph->addGraph(_microbePatients->getValue(start), start+1, _microbeTest->getValue(0));
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

    if(item == _microbeLoadAverage || item == _microbeLoadHealthyAverage || item == _microbeLoadCrohnsAverage || item == _microbeLoadSRSAverage || item == _microbeLoadSRXAverage)
    {
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
	else if(item == _microbeLoadSRSAverage)
	{
	    mgt = SMGT_SRS_AVERAGE;
	}
	else if(item == _microbeLoadSRXAverage)
	{
	    mgt = SMGT_SRX_AVERAGE;
	}

	// Bar Graph
	if(_microbeGraphType->getIndex() == 0)
	{
	    MicrobeGraphObject * mgo = new MicrobeGraphObject(_conn, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);

	    if(mgo->setSpecialGraph(mgt,(int)_microbeNumBars->getValue(),_microbeGrouping->getValue(),_microbeOrdering->getValue()))
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
		_currentSBGraph = new MicrobeBarGraphObject(_conn, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);
		_currentSBGraph->addSpecialGraph(mgt);
		checkLayout();
		_layoutObject->addGraphObject(_currentSBGraph);
		_microbeMenu->addItem(_microbeDone);
	    }
	    else
	    {
		_currentSBGraph->addSpecialGraph(mgt);
	    }
	}
    }

    if(item == _microbeLoadPointLine)
    {
	MicrobePointLineObject * mplo = new MicrobePointLineObject(_conn, 1000.0, 1000.0, "Microbe Graph", false, true, false, true);
	if(mplo->setGraph())
	{
	    checkLayout();
	    _layoutObject->addGraphObject(mplo);
	}
	else
	{
	    delete mplo;
	}
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
	    StrainGraphObject * sgo = new StrainGraphObject(_conn, 1000.0, 1000.0, "Strain Graph", false, true, false, true);

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
		    StrainGraphObject * sgo = new StrainGraphObject(_conn, 1000.0, 1000.0, "Strain Graph", false, true, false, true);

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
		    StrainHMObject * shmo = new StrainHMObject(_conn, 1000.0, 1000.0, "Strain Heat Map", false, true, false, true);

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
	    _currentSymptomGraph = new SymptomGraphObject(_conn, 1000.0, 1000.0, "Symptom Graph", false, true, false, true);
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

    if(item == _eventLoadAll && _eventName->getListSize())
    {
	bool addObject = false;
	if(!_currentSymptomGraph)
	{
	    _currentSymptomGraph = new SymptomGraphObject(_conn, 1000.0, 1000.0, "Symptom Graph", false, true, false, true);
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

    if(item == _eventDone)
    {
	_currentSymptomGraph = NULL;
	_eventMenu->removeItem(_eventDone);
    }

    if(item == _scatterLoad)
    {
	if(_scatterFirstList->getListSize() && _scatterFirstList->getIndex() != _scatterSecondList->getIndex())
	{
	    MicrobeScatterGraphObject * msgo = new MicrobeScatterGraphObject(_conn, 1000.0, 1000.0, "Scatter Plot", false, true, false, true);
	    if(msgo->setGraph(_scatterSecondList->getValue() + " vs " + _scatterFirstList->getValue(),_scatterFirstList->getValue(),_scatterSecondList->getValue()))
	    {
		checkLayout();
		_layoutObject->addGraphObject(msgo);
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
		MicrobeScatterGraphObject * msgo = new MicrobeScatterGraphObject(_conn, 1000.0, 1000.0, "Scatter Plot", false, true, false, true);
		if(msgo->setGraph(_scatterSecondList->getValue(i) + " vs " + _scatterFirstList->getValue(),_scatterFirstList->getValue(),_scatterSecondList->getValue(i)))
		{
		    checkLayout();
		    _layoutObject->addGraphObject(msgo);
		}
		else
		{
		    delete msgo;
		}
	    }
	}
	return;
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
		GraphObject * gobject = new GraphObject(_conn, 1000.0, 1000.0, "DataGraph", false, true, false, true, false);
		if(gobject->addGraph(patient,test,averageColor))
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
		_multiObject = new GraphObject(_conn, 1000.0, 1000.0, "DataGraph", false, true, false, true, false);
	    }

	    if(_multiObject->addGraph(patient,test))
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

/*void FuturePatient::makeGraph(std::string name)
{
    mysqlpp::Connection conn(false);
    if(!conn.connect("futurepatient","palmsdev2.ucsd.edu","fpuser","FPp@ssw0rd"))
    {
	std::cerr << "Unable to connect to database." << std::endl;
	return;
    }

    std::stringstream qss;
    qss << "select Measurement.timestamp, unix_timestamp(Measurement.timestamp) as utime, Measurement.value from Measurement  inner join Measure  on Measurement.measure_id = Measure.measure_id  where Measure.name = \"" << name << "\" order by utime;";

    mysqlpp::Query query = conn.query(qss.str().c_str());
    mysqlpp::StoreQueryResult res;
    res = query.store();
    //std::cerr << "Num Rows: " << res.num_rows() << std::endl;
    if(!res.num_rows())
    {
	std::cerr << "Empty query result." << std::endl;
	return;
    }

    std::stringstream mss;
    mss << "select * from Measure where name = \"" << name << "\";";

    mysqlpp::Query metaQuery = conn.query(mss.str().c_str());
    mysqlpp::StoreQueryResult metaRes = metaQuery.store();

    if(!metaRes.num_rows())
    {
	std::cerr << "Meta Data query result empty." << std::endl;
	return;
    }

    osg::Vec3Array * points = new osg::Vec3Array(res.num_rows());
    osg::Vec4Array * colors = new osg::Vec4Array(res.num_rows());

    bool hasGoodRange = false;
    float goodLow, goodHigh;

    if(strcmp(metaRes[0]["good_low"].c_str(),"NULL") && metaRes[0]["good_high"].c_str())
    {
	hasGoodRange = true;
	goodLow = atof(metaRes[0]["good_low"].c_str());
	goodHigh = atof(metaRes[0]["good_high"].c_str());
    }

    //find min/max values
    time_t mint, maxt;
    mint = maxt = atol(res[0]["utime"].c_str());
    float minval,maxval;
    minval = maxval = atof(res[0]["value"].c_str());
    for(int i = 1; i < res.num_rows(); i++)
    {
	time_t time = atol(res[i]["utime"].c_str());
	float value = atof(res[i]["value"].c_str());

	if(time < mint)
	{
	    mint = time;
	}
	if(time > maxt)
	{
	    maxt = time;
	}
	if(value < minval)
	{
	    minval = value;
	}
	if(value > maxval)
	{
	    maxval = value;
	}
    }

    //std::cerr << "Mintime: " << mint << " Maxtime: " << maxt << " MinVal: " << minval << " Maxval: " << maxval << std::endl;

    for(int i = 0; i < res.num_rows(); i++)
    {
	time_t time = atol(res[i]["utime"].c_str());
	float value = atof(res[i]["value"].c_str());
	points->at(i) = osg::Vec3((time-mint) / (double)(maxt-mint),0,(value-minval) / (maxval-minval));
	if(hasGoodRange)
	{
	    if(value < goodLow || value > goodHigh)
	    {
		colors->at(i) = osg::Vec4(1.0,0,0,1.0);
	    }
	    else
	    {
		colors->at(i) = osg::Vec4(0,1.0,0,1.0);
	    }
	}
	else
	{
	    colors->at(i) = osg::Vec4(0,0,1.0,1.0);
	}
    }

    DataGraph * dg = new DataGraph();
    dg->addGraph(metaRes[0]["display_name"].c_str(), points, POINTS_WITH_LINES, "Time", metaRes[0]["units"].c_str(), osg::Vec4(0,1.0,0,1.0),colors);
    dg->setZDataRange(metaRes[0]["display_name"].c_str(),minval,maxval);
    dg->setXDataRangeTimestamp(metaRes[0]["display_name"].c_str(),mint,maxt);
    PluginHelper::getObjectsRoot()->addChild(dg->getGraphRoot());
}*/

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
	if(_conn)
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
	}

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

    TestLabel * labels = NULL;
    int numTests = 0;
    if(ComController::instance()->isMaster())
    {
	if(_conn)
	{
	    std::stringstream qss;
	    qss << "select patient_id, timestamp, unix_timestamp(timestamp) as utimestamp from Microbe_Measurement group by patient_id, timestamp order by patient_id, timestamp;";

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
	_patientMicrobeTestMap[labels[j].id].push_back(labels[j].label);
	_patientMicrobeTestTimeMap[labels[j].id].push_back(labels[j].timestamp);
    }

    if(labels)
    {
	delete[] labels;
    }
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
	if(_conn)
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
	}

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
	    if(_conn)
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
	    }

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
    //struct timeval start,end;
    //gettimeofday(&start,NULL);
    //std::cerr << "Update Microbe Tests Patient: " << patientid << std::endl;
    
    if(_patientMicrobeTestMap.find(patientid) != _patientMicrobeTestMap.end())
    {
	_microbeTest->setValues(_patientMicrobeTestMap[patientid]);
	_microbeTestTime = _patientMicrobeTestTimeMap[patientid];
    }

    /*struct TestLabel
    {
	char label[256];
	time_t timestamp;
    };

    TestLabel * labels = NULL;
    int numTests = 0;

    _microbeTestTime.clear();

    if(ComController::instance()->isMaster())
    {
	if(_conn)
	{
	    std::stringstream qss;
	    qss << "select distinct timestamp, unix_timestamp(timestamp) as utimestamp from Microbe_Measurement where patient_id = \"" << patientid << "\" order by timestamp;";

	    mysqlpp::Query q = _conn->query(qss.str().c_str());
	    mysqlpp::StoreQueryResult res = q.store();

	    numTests = res.num_rows();

	    if(numTests)
	    {
		labels = new struct TestLabel[numTests];

		for(int i = 0; i < numTests; ++i)
		{
		    strncpy(labels[i].label,res[i]["timestamp"].c_str(),255);
		    labels[i].timestamp = atol(res[i]["utimestamp"].c_str());
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

    std::vector<std::string> labelVec;

    for(int i = 0; i < numTests; ++i)
    {
	labelVec.push_back(labels[i].label);
	_microbeTestTime.push_back(labels[i].timestamp);
    }

    _microbeTest->setValues(labelVec);*/
    //gettimeofday(&end,NULL);
    //std::cerr << "menu update: " << (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec) / 1000000.0) << std::endl;
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
