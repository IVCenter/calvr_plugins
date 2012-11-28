#include "FuturePatient.h"
#include "DataGraph.h"

#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/ComController.h>
#include <cvrConfig/ConfigManager.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

#include <mysql++/mysql++.h>

using namespace cvr;

CVRPLUGIN(FuturePatient)

FuturePatient::FuturePatient()
{
    _conn = NULL;
    _layoutObject = NULL;
    _multiObject = NULL;
    _currentSBGraph = NULL;
}

FuturePatient::~FuturePatient()
{
}

bool FuturePatient::init()
{
    /*DataGraph * dg = new DataGraph();
    srand(0);
    osg::Vec3Array * points = new osg::Vec3Array();
    for(int i = 0; i < 20; i++)
    {
	osg::Vec3 point(i / 20.0,0,(rand() % 1000)/1000.0);
	points->push_back(point);
	//std::cerr << "Point x: " << point.x() << " y: " << point.y() << " z: " << point.z() << std::endl;
    }

    dg->addGraph("TestGraph", points, POINTS_WITH_LINES, "X - Axis", "Z - Axis", osg::Vec4(0,1.0,0,1.0));
    dg->setZDataRange("TestGraph",-100,100);

    struct tm timestart,timeend;
    timestart.tm_year = 2007 - 1900;
    timestart.tm_mon = 3;
    timestart.tm_mday = 12;
    timestart.tm_hour = 14;
    timestart.tm_min = 20;
    timestart.tm_sec = 1;

    timeend.tm_year = 2008 - 1900;
    timeend.tm_mon = 5;
    timeend.tm_mday = 1;
    timeend.tm_hour = 20;
    timeend.tm_min = 4;
    timeend.tm_sec = 58;

    time_t tstart, tend;
    tstart = mktime(&timestart);
    tend = mktime(&timeend);

    dg->setXDataRangeTimestamp("TestGraph",tstart,tend);

    timestart.tm_year = 2003 - 1900;
    timeend.tm_year = 2007 - 1900;
    tstart = mktime(&timestart);
    tend = mktime(&timeend);*/

    //dg->setXDisplayRangeTimestamp(tstart,tend);

    //dg->setXDisplayRange(0.25,0.8);
    //PluginHelper::getObjectsRoot()->addChild(dg->getGraphRoot());

    //makeGraph("SIga");
    
    _fpMenu = new SubMenu("FuturePatient");

    _presetMenu = new SubMenu("Presets");
    _fpMenu->addItem(_presetMenu);

    _inflammationButton = new MenuButton("Big 4");
    _inflammationButton->setCallback(this);
    _presetMenu->addItem(_inflammationButton);

    _loadAll = new MenuButton("All");
    _loadAll->setCallback(this);
    _presetMenu->addItem(_loadAll);

    _groupLoadMenu = new SubMenu("Group Load");
    _fpMenu->addItem(_groupLoadMenu);

    _testList = new MenuList();
    _testList->setCallback(this);
    _fpMenu->addItem(_testList);

    _loadButton = new MenuButton("Load");
    _loadButton->setCallback(this);
    _fpMenu->addItem(_loadButton);

    _removeAllButton = new MenuButton("Remove All");
    _removeAllButton->setCallback(this);
    _fpMenu->addItem(_removeAllButton);

    _multiAddCB = new MenuCheckbox("Multi Add", false);
    _multiAddCB->setCallback(this);
    _fpMenu->addItem(_multiAddCB);

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

    //_microbeNumBars = new MenuRangeValueCompact("Microbes",1,100,25);
    _microbeNumBars = new MenuRangeValueCompact("Microbes",1,2330,25);
    _microbeNumBars->setCallback(this);
    _microbeMenu->addItem(_microbeNumBars);

    _microbeLoad = new MenuButton("Load");
    _microbeLoad->setCallback(this);
    _microbeMenu->addItem(_microbeLoad);

    _microbeDone = new MenuButton("Done");
    _microbeDone->setCallback(this);

    PluginHelper::addRootMenuItem(_fpMenu);

    struct listField
    {
	char entry[256];
    };

    struct listField * lfList = NULL;
    int listEntries = 0;

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
	    mysqlpp::Query q = _conn->query("select distinct name from Measure order by name;");
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

    std::vector<std::string> stringlist;
    for(int i = 0; i < listEntries; i++)
    {
	stringlist.push_back(lfList[i].entry);
    }

    _testList->setValues(stringlist);

    if(lfList)
    {
	delete[] lfList;
    }

    _groupList = new MenuList();
    _groupList->setCallback(this);
    _groupLoadMenu->addItem(_groupList);

    _groupLoadButton = new MenuButton("Load");
    _groupLoadButton->setCallback(this);
    _groupLoadMenu->addItem(_groupLoadButton);

    lfList = NULL;
    listEntries = 0;
    int * sizes = NULL;
    listField ** groupLists = NULL;

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
    
    if(_microbePatients->getListSize())
    {
	updateMicrobeTests(_microbePatients->getIndex() + 1);
    }

    /*if(_conn)
    {
	GraphObject * gobject = new GraphObject(_conn, 1000.0, 1000.0, "DataGraph", false, true, false, true, false);
	gobject->addGraph("LDL");
	PluginHelper::registerSceneObject(gobject,"FuturePatient");
	gobject->attachToScene();
    }*/

    return true;
}

void FuturePatient::preFrame()
{
    if(_layoutObject)
    {
	_layoutObject->perFrame();
    }
}

void FuturePatient::menuCallback(MenuItem * item)
{
    if(item == _loadButton)
    {
	loadGraph(_testList->getValue());
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
	    loadGraph(_groupTestMap[_groupList->getValue()][i]);
	}
    }

    if(item == _inflammationButton)
    {
	loadGraph("hs-CRP");
	loadGraph("SIgA");
	loadGraph("Lysozyme");
	loadGraph("Lactoferrin");
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
	for(int i = 0; i < _testList->getListSize(); i++)
	{
	    loadGraph(_testList->getValue(i));
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
	menuCallback(_multiAddCB);
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
	    if(mgo->setGraph(_microbePatients->getValue(), _microbePatients->getIndex()+1, _microbeTest->getValue(),(int)_microbeNumBars->getValue()))
	    {
		//PluginHelper::registerSceneObject(mgo,"FuturePatient");
		//mgo->attachToScene();
		//_microbeGraphList.push_back(mgo);
		checkLayout();
		_layoutObject->addMicrobeGraphObject(mgo);
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
		//PluginHelper::registerSceneObject(_currentSBGraph,"FuturePatient");
		//_currentSBGraph->attachToScene();
		checkLayout();
		_layoutObject->addMicrobeBarGraphObject(_currentSBGraph);
		_microbeMenu->addItem(_microbeDone);
	    }
	    _currentSBGraph->addGraph(_microbePatients->getValue(), _microbePatients->getIndex()+1, _microbeTest->getValue());
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

	    if(mgo->setSpecialGraph(mgt,(int)_microbeNumBars->getValue()))
	    {
		//PluginHelper::registerSceneObject(mgo,"FuturePatient");
		//mgo->attachToScene();
		//_microbeGraphList.push_back(mgo);
		checkLayout();
		_layoutObject->addMicrobeGraphObject(mgo);
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
		checkLayout();
		_layoutObject->addMicrobeBarGraphObject(_currentSBGraph);
		_microbeMenu->addItem(_microbeDone);
	    }
	    _currentSBGraph->addSpecialGraph(mgt);
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
	_layoutObject = new GraphLayoutObject(width,height,3,"GraphLayout",false,true,false,true,false);
	_layoutObject->setPosition(pos);
	PluginHelper::registerSceneObject(_layoutObject,"FuturePatient");
	_layoutObject->attachToScene();
    }
}

void FuturePatient::loadGraph(std::string name)
{
    checkLayout();

    std::string value = name;
    if(!value.empty())
    {
	if(!_multiAddCB->getValue())
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
		_layoutObject->addGraphObject(_graphObjectMap[value]);
	    }
	}
	else
	{
	    if(!_multiObject)
	    {
		_multiObject = new GraphObject(_conn, 1000.0, 1000.0, "DataGraph", false, true, false, true, false);
	    }

	    if(_multiObject->addGraph(value))
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
}

void FuturePatient::updateMicrobeTests(int patientid)
{
    //std::cerr << "Update Microbe Tests Patient: " << patientid << std::endl;
    struct TestLabel
    {
	char label[256];
    };

    TestLabel * labels = NULL;
    int numTests = 0;

    if(ComController::instance()->isMaster())
    {
	if(_conn)
	{
	    std::stringstream qss;
	    qss << "select distinct timestamp from Microbe_Measurement where patient_id = \"" << patientid << "\" order by timestamp;";

	    mysqlpp::Query q = _conn->query(qss.str().c_str());
	    mysqlpp::StoreQueryResult res = q.store();

	    numTests = res.num_rows();

	    if(numTests)
	    {
		labels = new struct TestLabel[numTests];

		for(int i = 0; i < numTests; ++i)
		{
		    strncpy(labels[i].label,res[i]["timestamp"].c_str(),255);
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
    }

    _microbeTest->setValues(labelVec);
}
