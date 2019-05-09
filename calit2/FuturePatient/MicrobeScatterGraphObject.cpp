#include "MicrobeScatterGraphObject.h"
#include "GraphLayoutObject.h"
#include "ColorGenerator.h"

#include <cvrKernel/ComController.h>
#include <cvrConfig/ConfigManager.h>
#include <cvrInput/TrackingManager.h>
#include <cvrUtil/OsgMath.h>

#include <iostream>
#include <sstream>
#include <cstring>

// can be removed later
#include <sys/time.h>

bool MicrobeScatterGraphObject::_dataInit = false;
std::vector<std::vector<struct MicrobeScatterGraphObject::DataEntry> > MicrobeScatterGraphObject::_data;
std::map<std::string,int> MicrobeScatterGraphObject::_phylumIndexMap;

using namespace cvr;

MicrobeScatterGraphObject::MicrobeScatterGraphObject(DBManager * dbm, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : LayoutTypeObject(name,navigation,movable,clip,contextMenu,showBounds), SelectableObject()
{
    _graph = new GroupedScatterPlot(width,height);
    _dbm = dbm;

    _desktopMode = ConfigManager::getBool("Plugin.FuturePatient.DesktopMode",false);

    setBoundsCalcMode(SceneObject::MANUAL);
    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);

    makeSelect();
    updateSelect();

    if(contextMenu)
    {
	_selectCB = new MenuCheckbox("Selected",false);
	_selectCB->setCallback(this);
	addMenuItem(_selectCB);
    }
}

MicrobeScatterGraphObject::~MicrobeScatterGraphObject()
{
}

bool MicrobeScatterGraphObject::setGraph(std::string title, std::string primaryPhylum, std::string secondaryPhylum, MicrobeGraphType type, std::string microbeTableSuffix, std::string measureTableSuffix)
{
    std::string measurementTable = "Microbe_Measurement";
    measurementTable += measureTableSuffix;

    std::string microbesTable = "Microbes";
    microbesTable += microbeTableSuffix;

    struct microbeData
    {
	char name[512];
	char condition[512];
	int id;
	time_t timestamp;
	float value;
    };

    std::stringstream queryss;

    switch( type )
    {
        case MGT_SPECIES:
	default:
        {
	    //queryss << "select Patient.last_name, Patient.p_condition, Patient.patient_id, unix_timestamp(" << measurementTable << ".timestamp) as timestamp, " << measurementTable << ".value from " << measurementTable << " inner join Patient on Patient.patient_id = " << measurementTable << ".patient_id where " << measurementTable << ".taxonomy_id = " << taxid << " and Patient.region = \"US\" order by p_condition, last_name, timestamp;";
	    queryss << "select Patient.p_condition, Patient.last_name, t.patient_id, unix_timestamp(t.timestamp) as timestamp, t.value from (select " << measurementTable << ".patient_id, " << measurementTable << ".timestamp, " << measurementTable << ".value as value from "<< measurementTable << " inner join " << microbesTable << " on " << measurementTable << ".taxonomy_id = " << microbesTable << ".taxonomy_id where " << microbesTable << ".species = \"" << primaryPhylum << "\")t inner join Patient on t.patient_id = Patient.patient_id where Patient.region = \"US\" order by Patient.p_condition, Patient.last_name, t.timestamp;";
            break;
        }
    
        case MGT_FAMILY:
        {
	    queryss << "select Patient.p_condition, Patient.last_name, t.patient_id, unix_timestamp(t.timestamp) as timestamp, t.value from (select " << microbesTable << ".family, " << measurementTable << ".patient_id, " << measurementTable << ".timestamp, sum(" << measurementTable << ".value) as value from "<< measurementTable << " inner join " << microbesTable << " on " << measurementTable << ".taxonomy_id = " << microbesTable << ".taxonomy_id where " << microbesTable << ".family = \"" << primaryPhylum << "\" group by " << measurementTable << ".patient_id, " << measurementTable << ".timestamp)t inner join Patient on t.patient_id = Patient.patient_id where Patient.region = \"US\" order by Patient.p_condition, Patient.last_name, t.timestamp;";
	        //queryss << "select Patient.last_name, Patient.p_condition, Patient.patient_id, unix_timestamp(" << measurementTable << ".timestamp) as timestamp, " << microbesTable << ".family, sum(" << measurementTable << ".value) as value from " << measurementTable << " inner join Patient on Patient.patient_id = " << measurementTable << ".patient_id inner join " << microbesTable << " on " << microbesTable << ".taxonomy_id = " << measurementTable << ".taxonomy_id where " << microbesTable << ".family = \"" << microbe << "\" and Patient.region = \"US\" group by patient_id, timestamp order by p_condition, last_name, timestamp;";
            break;
        }

        case MGT_GENUS:
        {
	    queryss << "select Patient.p_condition, Patient.last_name, t.patient_id, unix_timestamp(t.timestamp) as timestamp, t.value from (select " << microbesTable << ".genus, " << measurementTable << ".patient_id, " << measurementTable << ".timestamp, sum(" << measurementTable << ".value) as value from "<< measurementTable << " inner join " << microbesTable << " on " << measurementTable << ".taxonomy_id = " << microbesTable << ".taxonomy_id where " << microbesTable << ".genus = \"" << primaryPhylum << "\" group by " << measurementTable << ".patient_id, " << measurementTable << ".timestamp)t inner join Patient on t.patient_id = Patient.patient_id where Patient.region = \"US\" order by Patient.p_condition, Patient.last_name, t.timestamp;";
	        //queryss << "select Patient.last_name, Patient.p_condition, Patient.patient_id, unix_timestamp(" << measurementTable << ".timestamp) as timestamp, " << microbesTable << ".genus, sum(" << measurementTable << ".value) as value from " << measurementTable << " inner join Patient on Patient.patient_id = " << measurementTable << ".patient_id inner join " << microbesTable << " on " << microbesTable << ".taxonomy_id = " << measurementTable << ".taxonomy_id where " << microbesTable << ".genus = \"" << microbe << "\" and Patient.region = \"US\" group by patient_id, timestamp order by p_condition, last_name, timestamp;";

            break;
        }

	case MGT_PHYLUM:
	{
	    queryss << "select Patient.p_condition, Patient.last_name, t.patient_id, unix_timestamp(t.timestamp) as timestamp, t.value from (select " << microbesTable << ".genus, " << measurementTable << ".patient_id, " << measurementTable << ".timestamp, sum(" << measurementTable << ".value) as value from "<< measurementTable << " inner join " << microbesTable << " on " << measurementTable << ".taxonomy_id = " << microbesTable << ".taxonomy_id where " << microbesTable << ".phylum = \"" << primaryPhylum << "\" group by " << measurementTable << ".patient_id, " << measurementTable << ".timestamp)t inner join Patient on t.patient_id = Patient.patient_id where Patient.region = \"US\" order by Patient.p_condition, Patient.last_name, t.timestamp;";
	}

        /*default:
        {
	        queryss << "select Patient.last_name, Patient.p_condition, Patient.patient_id, unix_timestamp(" << measurementTable << ".timestamp) as timestamp, " << measurementTable << ".value from " << measurementTable << " inner join Patient on Patient.patient_id = " << measurementTable << ".patient_id where " << measurementTable << ".taxonomy_id = " << taxid << " and Patient.region = \"US\" order by p_condition, last_name, timestamp;";

            break;
        }*/
    }

    //std::cerr << "Query: " << queryss.str() << std::endl;

    struct microbeData * data = NULL;
    int dataSize = 0;

    if(ComController::instance()->isMaster())
    {
	if(_dbm)
	{
	    DBMQueryResult result;

	    _dbm->runQuery(queryss.str(),result);

	    dataSize = result.numRows();

	    if(dataSize)
	    {
		data = new struct microbeData[dataSize];
		for(int i = 0; i < dataSize; ++i)
		{
		    data[i].name[511] = '\0';
		    strncpy(data[i].name,result(i,"last_name").c_str(),511);
		    data[i].condition[511] = '\0';
		    strncpy(data[i].condition,result(i,"p_condition").c_str(),511);
		    data[i].id = atoi(result(i,"patient_id").c_str());
		    data[i].timestamp = atol(result(i,"timestamp").c_str());
		    data[i].value = atof(result(i,"value").c_str());
		}
	    }
	}

	/*if(_conn)
	{
	    mysqlpp::Query query = _conn->query(queryss.str().c_str());
	    mysqlpp::StoreQueryResult res = query.store();

	    dataSize = res.num_rows();

	    if(dataSize)
	    {
		data = new struct microbeData[dataSize];
		for(int i = 0; i < dataSize; ++i)
		{
		    data[i].name[511] = '\0';
		    strncpy(data[i].name,res[i]["last_name"].c_str(),511);
		    data[i].condition[511] = '\0';
		    strncpy(data[i].condition,res[i]["p_condition"].c_str(),511);
		    data[i].id = atoi(res[i]["patient_id"].c_str());
		    data[i].timestamp = atol(res[i]["timestamp"].c_str());
		    data[i].value = atof(res[i]["value"].c_str());
		}
	    }
	}*/

	ComController::instance()->sendSlaves(&dataSize,sizeof(int));
	if(dataSize)
	{
	    ComController::instance()->sendSlaves(data,dataSize*sizeof(struct microbeData));
	}
    }
    else
    {
	ComController::instance()->readMaster(&dataSize,sizeof(int));
	if(dataSize)
	{
	    data = new struct microbeData[dataSize];
	    ComController::instance()->readMaster(data,dataSize*sizeof(struct microbeData));
	}
    }

    //std::cerr << "Data size: " << dataSize << std::endl;

    std::map<std::string, std::vector<std::pair<std::string, std::pair<float,float> > > > dataMap;

    for(int i = 0; i < dataSize; ++i)
    {
	std::string condition = data[i].condition;
	//std::cerr << "Condition: " << condition << " value: " << data[i].value << std::endl;
	if(condition == "CD" || condition == "crohn's disease" || condition == "healthy" || condition == "Larry" || condition == "UC" || condition == "ulcerous colitis")
	{
	    //if(data[i].value > 0.0)
	    {
		char timestamp[512];
		timestamp[511] = '\0';
		strftime(timestamp,511,"%F",localtime(&data[i].timestamp));

		std::string group;

		std::string name = data[i].name;
		name = name + " - " + timestamp;

		if(condition == "CD" || condition == "crohn's disease")
		{
		    group = "Crohns";
		}
		else if(condition == "UC" || condition == "ulcerous colitis")
		{
		    group = "UC";
		}

		if(condition == "healthy")
		{
		    group = "Healthy";
		}
		else if(condition == "Larry")
		{
		    //group = "Smarr";
		    group = "LS";
		}

		dataMap[group].push_back(std::pair<std::string,std::pair<float,float> >(name,std::pair<float,float>(data[i].value,-1.0)));
	    }
	}
    }

    if(data)
    {
	delete[] data;
    }

    if(!dataSize)
    {
	std::cerr << "No entries for " << primaryPhylum << std::endl;
	return false;
    }

    std::stringstream queryss2;

    switch( type )
    {
        case MGT_SPECIES:
	default:
        {
	    //queryss << "select Patient.last_name, Patient.p_condition, Patient.patient_id, unix_timestamp(" << measurementTable << ".timestamp) as timestamp, " << measurementTable << ".value from " << measurementTable << " inner join Patient on Patient.patient_id = " << measurementTable << ".patient_id where " << measurementTable << ".taxonomy_id = " << taxid << " and Patient.region = \"US\" order by p_condition, last_name, timestamp;";
	    queryss2 << "select Patient.p_condition, Patient.last_name, t.patient_id, unix_timestamp(t.timestamp) as timestamp, t.value from (select " << measurementTable << ".patient_id, " << measurementTable << ".timestamp, " << measurementTable << ".value as value from "<< measurementTable << " inner join " << microbesTable << " on " << measurementTable << ".taxonomy_id = " << microbesTable << ".taxonomy_id where " << microbesTable << ".species = \"" << secondaryPhylum << "\")t inner join Patient on t.patient_id = Patient.patient_id where Patient.region = \"US\" order by Patient.p_condition, Patient.last_name, t.timestamp;";
            break;
        }
    
        case MGT_FAMILY:
        {
	    queryss2 << "select Patient.p_condition, Patient.last_name, t.patient_id, unix_timestamp(t.timestamp) as timestamp, t.value from (select " << microbesTable << ".family, " << measurementTable << ".patient_id, " << measurementTable << ".timestamp, sum(" << measurementTable << ".value) as value from "<< measurementTable << " inner join " << microbesTable << " on " << measurementTable << ".taxonomy_id = " << microbesTable << ".taxonomy_id where " << microbesTable << ".family = \"" << secondaryPhylum << "\" group by " << measurementTable << ".patient_id, " << measurementTable << ".timestamp)t inner join Patient on t.patient_id = Patient.patient_id where Patient.region = \"US\" order by Patient.p_condition, Patient.last_name, t.timestamp;";
	        //queryss << "select Patient.last_name, Patient.p_condition, Patient.patient_id, unix_timestamp(" << measurementTable << ".timestamp) as timestamp, " << microbesTable << ".family, sum(" << measurementTable << ".value) as value from " << measurementTable << " inner join Patient on Patient.patient_id = " << measurementTable << ".patient_id inner join " << microbesTable << " on " << microbesTable << ".taxonomy_id = " << measurementTable << ".taxonomy_id where " << microbesTable << ".family = \"" << microbe << "\" and Patient.region = \"US\" group by patient_id, timestamp order by p_condition, last_name, timestamp;";
            break;
        }

        case MGT_GENUS:
        {
	    queryss2 << "select Patient.p_condition, Patient.last_name, t.patient_id, unix_timestamp(t.timestamp) as timestamp, t.value from (select " << microbesTable << ".genus, " << measurementTable << ".patient_id, " << measurementTable << ".timestamp, sum(" << measurementTable << ".value) as value from "<< measurementTable << " inner join " << microbesTable << " on " << measurementTable << ".taxonomy_id = " << microbesTable << ".taxonomy_id where " << microbesTable << ".genus = \"" << secondaryPhylum << "\" group by " << measurementTable << ".patient_id, " << measurementTable << ".timestamp)t inner join Patient on t.patient_id = Patient.patient_id where Patient.region = \"US\" order by Patient.p_condition, Patient.last_name, t.timestamp;";
	        //queryss << "select Patient.last_name, Patient.p_condition, Patient.patient_id, unix_timestamp(" << measurementTable << ".timestamp) as timestamp, " << microbesTable << ".genus, sum(" << measurementTable << ".value) as value from " << measurementTable << " inner join Patient on Patient.patient_id = " << measurementTable << ".patient_id inner join " << microbesTable << " on " << microbesTable << ".taxonomy_id = " << measurementTable << ".taxonomy_id where " << microbesTable << ".genus = \"" << microbe << "\" and Patient.region = \"US\" group by patient_id, timestamp order by p_condition, last_name, timestamp;";

            break;
        }

	case MGT_PHYLUM:
	{
	    queryss2 << "select Patient.p_condition, Patient.last_name, t.patient_id, unix_timestamp(t.timestamp) as timestamp, t.value from (select " << microbesTable << ".genus, " << measurementTable << ".patient_id, " << measurementTable << ".timestamp, sum(" << measurementTable << ".value) as value from "<< measurementTable << " inner join " << microbesTable << " on " << measurementTable << ".taxonomy_id = " << microbesTable << ".taxonomy_id where " << microbesTable << ".phylum = \"" << secondaryPhylum << "\" group by " << measurementTable << ".patient_id, " << measurementTable << ".timestamp)t inner join Patient on t.patient_id = Patient.patient_id where Patient.region = \"US\" order by Patient.p_condition, Patient.last_name, t.timestamp;";
	}

        /*default:
        {
	        queryss << "select Patient.last_name, Patient.p_condition, Patient.patient_id, unix_timestamp(" << measurementTable << ".timestamp) as timestamp, " << measurementTable << ".value from " << measurementTable << " inner join Patient on Patient.patient_id = " << measurementTable << ".patient_id where " << measurementTable << ".taxonomy_id = " << taxid << " and Patient.region = \"US\" order by p_condition, last_name, timestamp;";

            break;
        }*/
    }

    //std::cerr << "Query: " << queryss2.str() << std::endl;

    data = NULL;
    dataSize = 0;

    if(ComController::instance()->isMaster())
    {
	if(_dbm)
	{
	    DBMQueryResult result;

	    _dbm->runQuery(queryss2.str(),result);

	    dataSize = result.numRows();

	    if(dataSize)
	    {
		data = new struct microbeData[dataSize];
		for(int i = 0; i < dataSize; ++i)
		{
		    data[i].name[511] = '\0';
		    strncpy(data[i].name,result(i,"last_name").c_str(),511);
		    data[i].condition[511] = '\0';
		    strncpy(data[i].condition,result(i,"p_condition").c_str(),511);
		    data[i].id = atoi(result(i,"patient_id").c_str());
		    data[i].timestamp = atol(result(i,"timestamp").c_str());
		    data[i].value = atof(result(i,"value").c_str());
		}
	    }
	}

	/*if(_conn)
	{
	    mysqlpp::Query query = _conn->query(queryss.str().c_str());
	    mysqlpp::StoreQueryResult res = query.store();

	    dataSize = res.num_rows();

	    if(dataSize)
	    {
		data = new struct microbeData[dataSize];
		for(int i = 0; i < dataSize; ++i)
		{
		    data[i].name[511] = '\0';
		    strncpy(data[i].name,res[i]["last_name"].c_str(),511);
		    data[i].condition[511] = '\0';
		    strncpy(data[i].condition,res[i]["p_condition"].c_str(),511);
		    data[i].id = atoi(res[i]["patient_id"].c_str());
		    data[i].timestamp = atol(res[i]["timestamp"].c_str());
		    data[i].value = atof(res[i]["value"].c_str());
		}
	    }
	}*/

	ComController::instance()->sendSlaves(&dataSize,sizeof(int));
	if(dataSize)
	{
	    ComController::instance()->sendSlaves(data,dataSize*sizeof(struct microbeData));
	}
    }
    else
    {
	ComController::instance()->readMaster(&dataSize,sizeof(int));
	if(dataSize)
	{
	    data = new struct microbeData[dataSize];
	    ComController::instance()->readMaster(data,dataSize*sizeof(struct microbeData));
	}
    }

    //std::cerr << "Data size: " << dataSize << std::endl;

    for(int i = 0; i < dataSize; ++i)
    {
	std::string condition = data[i].condition;
	//std::cerr << "Condition: " << condition << " value: " << data[i].value << std::endl;
	if(condition == "CD" || condition == "crohn's disease" || condition == "healthy" || condition == "Larry" || condition == "UC" || condition == "ulcerous colitis")
	{
	    //if(data[i].value > 0.0)
	    {
		char timestamp[512];
		timestamp[511] = '\0';
		strftime(timestamp,511,"%F",localtime(&data[i].timestamp));

		std::string group;

		std::string name = data[i].name;
		name = name + " - " + timestamp;

		if(condition == "CD" || condition == "crohn's disease")
		{
		    group = "Crohns";
		}
		else if(condition == "UC" || condition == "ulcerous colitis")
		{
		    group = "UC";
		}

		if(condition == "healthy")
		{
		    group = "Healthy";
		}
		else if(condition == "Larry")
		{
		    //group = "Smarr";
		    group = "LS";
		}

		if(dataMap.find(group) != dataMap.end())
		{
		    for(int j = 0; j < dataMap[group].size(); ++j)
		    {
			if(dataMap[group][j].first == name)
			{
			    dataMap[group][j].second.second = data[i].value;
			}
		    }
		}
	    }
	}
    }

    if(data)
    {
	delete[] data;
    }

    if(!dataSize)
    {
	std::cerr << "No entries for " << secondaryPhylum << std::endl;
	return false;
    }

    _graph->setLabels(title,primaryPhylum,secondaryPhylum);
    _graph->setAxisTypes(GSP_LOG,GSP_LOG);

    _graph->setColorMapping(GraphGlobals::getDefaultPhylumColor(),GraphGlobals::getPatientColorMap());

    bool graphValid = false;

    int groupIndex = 0;
    for(std::map<std::string, std::vector<std::pair<std::string, std::pair<float,float> > > >::iterator it = dataMap.begin(); it != dataMap.end(); ++it)
    {
	//std::cerr << "Group: " << groupIndex << " name: " << it->first << std::endl;

	std::vector<std::pair<float,float> > dataList;
	std::vector<std::string> labels;
	for(int i = 0; i < it->second.size(); ++i)
	{
	    //if(it->second[i].second.first > 0.0 && it->second[i].second.second > 0.0)
	    {
		//std::cerr << "adding data name: " << it->second[i].first << " first: " << it->second[i].second.first << " second: " << it->second[i].second.second << std::endl;
		labels.push_back(it->second[i].first);
		dataList.push_back(std::pair<float,float>(it->second[i].second.first,it->second[i].second.second));
	    }
	}

	if(dataList.size())
	{
	    if(_graph->addGroup(groupIndex,it->first,dataList,labels))
	    {
		graphValid = true;
	    }
	}

	groupIndex++;
    }
    
    if(graphValid)
    {
	addChild(_graph->getRootNode());
    }

    return graphValid;

    /*if(!_dataInit)
    {
	initData();
    }

    if(!_dataInit)
    {
	return false;
    }

    if(_phylumIndexMap.find(primaryPhylum) == _phylumIndexMap.end() || _phylumIndexMap.find(secondaryPhylum) == _phylumIndexMap.end())
    {
	return false;
    }

    std::vector<std::string> groupLabels;
    groupLabels.push_back("Smarr");
    groupLabels.push_back("Crohns");
    groupLabels.push_back("UC");
    groupLabels.push_back("Healthy");

    std::vector<std::string> matchList;
    matchList.push_back("Smarr");
    matchList.push_back("CD-");
    matchList.push_back("UC-");
    matchList.push_back("HE-");

    _graph->setLabels(title,primaryPhylum,secondaryPhylum);
    _graph->setAxisTypes(GSP_LOG,GSP_LOG);

    for(int i = 0; i < groupLabels.size(); ++i)
    {
	std::map<std::string,std::pair<float,float> > groupDataMap;

	int index = _phylumIndexMap[primaryPhylum];

	const char * format = "%F";
	char buffer[256];

	for(int j = 0; j < _data[index].size(); ++j)
	{
	    if(!strncmp((char*)_data[index][j].name.c_str(),matchList[i].c_str(),matchList[i].size()))
	    {
		strftime(buffer,256,format,localtime(&_data[index][j].timestamp));
		std::string label = _data[index][j].name + " - " + buffer;
		groupDataMap[label].first = _data[index][j].value;
	    }
	}
	
	index = _phylumIndexMap[secondaryPhylum];

	for(int j = 0; j < _data[index].size(); ++j)
	{
	    if(!strncmp((char*)_data[index][j].name.c_str(),matchList[i].c_str(),matchList[i].size()))
	    {
		strftime(buffer,256,format,localtime(&_data[index][j].timestamp));
		std::string label = _data[index][j].name + " - " + buffer;
		groupDataMap[label].second = _data[index][j].value;
	    }
	}

	std::vector<std::pair<float,float> > dataList;
	std::vector<std::string> labels;
	for(std::map<std::string,std::pair<float,float> >::iterator it = groupDataMap.begin(); it != groupDataMap.end(); ++it)
	{
	    if(it->second.first > 0.0 && it->second.second > 0.0)
	    {
		labels.push_back(it->first);
		dataList.push_back(it->second);
	    }
	}

	_graph->addGroup(i,groupLabels[i],dataList,labels);
    }

    addChild(_graph->getRootNode());

    return true;*/
}

void MicrobeScatterGraphObject::objectAdded()
{
    GraphLayoutObject * layout = dynamic_cast<GraphLayoutObject*>(_parent);
    if(layout && layout->getPatientKeyObject())
    {
	bool addKey = !layout->getPatientKeyObject()->hasRef();
	layout->getPatientKeyObject()->ref(this);

	if(addKey)
	{
	    layout->addLineObject(layout->getPatientKeyObject());
	}
    }
}

void MicrobeScatterGraphObject::objectRemoved()
{
    GraphLayoutObject * layout = dynamic_cast<GraphLayoutObject*>(_parent);
    if(layout && layout->getPatientKeyObject())
    {
	layout->getPatientKeyObject()->unref(this);
    }
}

void MicrobeScatterGraphObject::setGraphSize(float width, float height)
{
    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);

    updateSelect();

    _graph->setDisplaySize(width,height);
}

void MicrobeScatterGraphObject::selectPatients(std::map<std::string,std::vector<std::string> > & selectMap)
{
    _graph->selectPoints(selectMap);
}

float MicrobeScatterGraphObject::getGraphXMaxValue()
{
    return _graph->getFirstMax();
}

float MicrobeScatterGraphObject::getGraphXMinValue()
{
    return _graph->getFirstMin();
}

float MicrobeScatterGraphObject::getGraphZMaxValue()
{
    return _graph->getSecondMax();
}

float MicrobeScatterGraphObject::getGraphZMinValue()
{
    return _graph->getSecondMin();
}

float MicrobeScatterGraphObject::getGraphXDisplayRangeMax()
{
    return _graph->getFirstDisplayMax();
}

float MicrobeScatterGraphObject::getGraphXDisplayRangeMin()
{
    return _graph->getFirstDisplayMin();
}

float MicrobeScatterGraphObject::getGraphZDisplayRangeMax()
{
    return _graph->getSecondDisplayMax();
}

float MicrobeScatterGraphObject::getGraphZDisplayRangeMin()
{
    return _graph->getSecondDisplayMin();
}

void MicrobeScatterGraphObject::setGraphXDisplayRange(float min, float max)
{
    _graph->setFirstDisplayRange(min,max);
}

void MicrobeScatterGraphObject::setGraphZDisplayRange(float min, float max)
{
    _graph->setSecondDisplayRange(min,max);
}

void MicrobeScatterGraphObject::resetGraphDisplayRange()
{
    _graph->resetDisplayRange();
}

bool MicrobeScatterGraphObject::processEvent(cvr::InteractionEvent * ie)
{
    if(ie->asTrackedButtonEvent() && ie->asTrackedButtonEvent()->getButton() == 0 && (ie->getInteraction() == BUTTON_DOWN || ie->getInteraction() == BUTTON_DOUBLE_CLICK))
    {
	GraphLayoutObject * layout = dynamic_cast<GraphLayoutObject*>(_parent);
	if(!layout)
	{
	    return false;
	}

	std::string patientGroup;
	std::vector<std::string> selectedPatients;

	bool clickUsed = false;
	if(_graph->processClick(patientGroup,selectedPatients))
	{
	    clickUsed = true;
	}

	layout->selectPatients(patientGroup,selectedPatients);
	if(clickUsed)
	{
	    return true;
	}
    }

    return FPTiledWallSceneObject::processEvent(ie);
}

void MicrobeScatterGraphObject::updateCallback(int handID, const osg::Matrix & mat)
{
    if((_desktopMode && TrackingManager::instance()->getHandTrackerType(handID) == TrackerBase::MOUSE) ||
	(!_desktopMode && TrackingManager::instance()->getHandTrackerType(handID) == TrackerBase::POINTER))
    {
	osg::Vec3 start, end(0,1000,0);
	start = start * mat * getWorldToObjectMatrix();
	end = end * mat * getWorldToObjectMatrix();

	osg::Vec3 planePoint;
	osg::Vec3 planeNormal(0,-1,0);
	osg::Vec3 intersect;
	float w;

	if(linePlaneIntersectionRef(start,end,planePoint,planeNormal,intersect,w))
	{
	    _graph->setHover(intersect);
	}
    }
}

void MicrobeScatterGraphObject::leaveCallback(int handID)
{
    if((_desktopMode && TrackingManager::instance()->getHandTrackerType(handID) == TrackerBase::MOUSE) ||
	(!_desktopMode && TrackingManager::instance()->getHandTrackerType(handID) == TrackerBase::POINTER))
    {
	_graph->clearHoverText();
    }
}

void MicrobeScatterGraphObject::menuCallback(MenuItem * item)
{
    if(item == _selectCB)
    {
	if(_selectCB->getValue())
	{
	    addChild(_selectGeode);
	}
	else
	{
	    removeChild(_selectGeode);
	}
	return;
    }

    FPTiledWallSceneObject::menuCallback(item);
}

void MicrobeScatterGraphObject::setLogScale(bool logScale)
{
    if(logScale)
    {
	_graph->setAxisTypes(GSP_LOG,GSP_LOG);
    }
    else
    {
	_graph->setAxisTypes(GSP_LINEAR,GSP_LINEAR);
    }
}

void MicrobeScatterGraphObject::initData()
{
    int numPhylum = 0;
    struct phylumEntry
    {
	char name[1024];
    };
    struct phylumEntry * phylums = NULL;

    int * orderSizes = NULL;
    int orderTotal = 0;
    struct orderEntry
    {
	char name[1024];
	time_t timestamp;
    };
    struct orderEntry * order = NULL;

    float * data = NULL;

    if(ComController::instance()->isMaster())
    {
	if(_dbm)
	{
	    std::string sizesQuery = "SELECT q.phylum, count(q.value) as count from (SELECT Microbes.phylum, Patient.last_name, unix_timestamp(Microbe_Measurement.timestamp) as timestamp, sum(Microbe_Measurement.value) as value FROM Microbe_Measurement INNER JOIN Patient ON Microbe_Measurement.patient_id = Patient.patient_id INNER JOIN Microbes ON Microbe_Measurement.taxonomy_id = Microbes.taxonomy_id GROUP BY Patient.last_name, Microbe_Measurement.timestamp, Microbes.phylum ORDER BY Microbes.phylum, Patient.last_name, Microbe_Measurement.timestamp)q group by q.phylum order by q.phylum;";
	    std::string orderQuery = "SELECT q.phylum, q.last_name, q.timestamp from (SELECT Microbes.phylum, Patient.last_name, unix_timestamp(Microbe_Measurement.timestamp) as timestamp, sum(Microbe_Measurement.value) as value FROM Microbe_Measurement INNER JOIN Patient ON Microbe_Measurement.patient_id = Patient.patient_id INNER JOIN Microbes ON Microbe_Measurement.taxonomy_id = Microbes.taxonomy_id GROUP BY Patient.last_name, Microbe_Measurement.timestamp, Microbes.phylum ORDER BY Microbes.phylum, Patient.last_name, Microbe_Measurement.timestamp)q order by q.phylum, q.last_name, q.timestamp;";
	    std::string query = "SELECT Microbes.phylum, Patient.last_name, unix_timestamp(Microbe_Measurement.timestamp) as timestamp, sum(Microbe_Measurement.value) as value FROM Microbe_Measurement INNER JOIN Patient ON Microbe_Measurement.patient_id = Patient.patient_id INNER JOIN Microbes ON Microbe_Measurement.taxonomy_id = Microbes.taxonomy_id GROUP BY Patient.last_name, Microbe_Measurement.timestamp, Microbes.phylum ORDER BY Microbes.phylum, Patient.last_name, Microbe_Measurement.timestamp;";

	    std::cerr << "Starting first query." << std::endl;

	    DBMQueryResult sresult;

	    _dbm->runQuery(sizesQuery,sresult);

	    numPhylum = sresult.numRows();

	    std::cerr << "Num Phylum: " << numPhylum << std::endl;

	    if(numPhylum)
	    {
		phylums = new struct phylumEntry[numPhylum];
		orderSizes = new int[numPhylum];
		for(int i = 0; i < numPhylum; ++i)
		{
		    strncpy(phylums[i].name,sresult(i,"phylum").c_str(),1023);
		    orderSizes[i] = atoi(sresult(i,"count").c_str());
		    orderTotal += orderSizes[i];
		}

		std::cerr << "OrderTotal: " << orderTotal << std::endl;

                DBMQueryResult oresult;

		_dbm->runQuery(orderQuery,oresult);

		std::cerr << "Second done." << std::endl;

		if(oresult.numRows() == orderTotal)
		{
		    order = new struct orderEntry[orderTotal];
		    for(int i = 0; i < orderTotal; ++i)
		    {
			strncpy(order[i].name,oresult(i,"last_name").c_str(),1023);
			order[i].timestamp = atol(oresult(i,"timestamp").c_str());
		    }
		}
		else
		{
		    std::cerr << "Number of order rows different than expected. Wanted: " << orderTotal << " Got: " << oresult.numRows() << std::endl;
		    orderTotal = 0;
		}

		std::cerr << "Last started." << std::endl;

		DBMQueryResult vresult;

		_dbm->runQuery(query,vresult);

		std::cerr << "Last done." << std::endl;

		if(vresult.numRows() == orderTotal)
		{
		    data = new float[orderTotal];
		    for(int i = 0; i < orderTotal; ++i)
		    {
			data[i] = atof(vresult(i,"value").c_str());
		    }
		}
		else
		{
		    std::cerr << "Number of values different than expected. Wanted: " << orderTotal << " Got: " << vresult.numRows() << std::endl;
		    orderTotal = 0;
		}
	    }
	    else
	    {
		std::cerr << "No Phylum found." << std::endl;
	    }
	}

	ComController::instance()->sendSlaves(&numPhylum,sizeof(int));
	ComController::instance()->sendSlaves(&orderTotal,sizeof(int));

	if(numPhylum)
	{
	    ComController::instance()->sendSlaves(phylums,sizeof(struct phylumEntry)*numPhylum);
	    ComController::instance()->sendSlaves(orderSizes,sizeof(int)*numPhylum);

	    if(orderTotal)
	    {
		ComController::instance()->sendSlaves(order,sizeof(struct orderEntry)*orderTotal);
		ComController::instance()->sendSlaves(data,sizeof(float)*orderTotal);
	    }
	}
    }
    else
    {
	ComController::instance()->readMaster(&numPhylum,sizeof(int));
	ComController::instance()->readMaster(&orderTotal,sizeof(int));

	if(numPhylum)
	{
	    phylums = new struct phylumEntry[numPhylum];
	    orderSizes = new int[numPhylum];
	    ComController::instance()->readMaster(phylums,sizeof(struct phylumEntry)*numPhylum);
	    ComController::instance()->readMaster(orderSizes,sizeof(int)*numPhylum);

	    if(orderTotal)
	    {
		order = new struct orderEntry[orderTotal];
		data = new float[orderTotal];
		ComController::instance()->readMaster(order,sizeof(struct orderEntry)*orderTotal);
		ComController::instance()->readMaster(data,sizeof(float)*orderTotal);
	    }
	}
    }

    std::cerr << "Loading data." << std::endl;

    if(numPhylum && orderTotal)
    {
	int offset = 0;
	for(int i = 0; i < numPhylum; ++i)
	{
	    _phylumIndexMap[phylums[i].name] = i;
	    _data.push_back(std::vector<struct DataEntry>());
	    for(int j = 0; j < orderSizes[i]; ++j)
	    {
		struct DataEntry de;
		de.name = order[offset+j].name;
		de.timestamp = order[offset+j].timestamp;
		de.value = data[offset+j];
		_data.back().push_back(de);
	    }
	    offset += orderSizes[i];
	}

	_dataInit = true;
    }

    std::cerr << "Done." << std::endl;

    if(phylums)
    {
	delete[] phylums;
    }
    if(orderSizes)
    {
	delete[] orderSizes;
    }
    if(order)
    {
	delete[] order;
    }
    if(data)
    {
	delete[] data;
    }
}

void MicrobeScatterGraphObject::makeSelect()
{
    _selectGeode = new osg::Geode();
    _selectGeom = new osg::Geometry();
    _selectGeode->addDrawable(_selectGeom);
    _selectGeode->setCullingActive(false);

    osg::Vec3Array * verts = new osg::Vec3Array(16);
    osg::Vec4Array * colors = new osg::Vec4Array();

    colors->push_back(osg::Vec4(1.0,0.0,0.0,0.66));

    _selectGeom->setVertexArray(verts);
    _selectGeom->setColorArray(colors);
    _selectGeom->setColorBinding(osg::Geometry::BIND_OVERALL);
    _selectGeom->setUseDisplayList(false);
    _selectGeom->setUseVertexBufferObjects(true);

    _selectGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,16));

    osg::StateSet * stateset = _selectGeode->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    stateset->setMode(GL_BLEND,osg::StateAttribute::ON);
    stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
}

void MicrobeScatterGraphObject::updateSelect()
{
    if(!_selectGeom)
    {
	return;
    }

    osg::Vec3Array * verts = dynamic_cast<osg::Vec3Array*>(_selectGeom->getVertexArray());
    if(!verts)
    {
	return;
    }

    osg::BoundingBox bb = getOrComputeBoundingBox();

    osg::Vec3 ul(bb.xMin(),bb.yMin(),bb.zMax());
    osg::Vec3 ur(bb.xMax(),bb.yMin(),bb.zMax());
    osg::Vec3 ll(bb.xMin(),bb.yMin(),bb.zMin());
    osg::Vec3 lr(bb.xMax(),bb.yMin(),bb.zMin());

    float offset = std::min(bb.xMax()-bb.xMin(),bb.zMax()-bb.zMin())*0.015;

    // left
    verts->at(0) = ul;
    verts->at(1) = ll;
    verts->at(2) = ll + osg::Vec3(offset,0,0);
    verts->at(3) = ul + osg::Vec3(offset,0,0);

    // bottom
    verts->at(4) = ll + osg::Vec3(0,0,offset);
    verts->at(5) = ll;
    verts->at(6) = lr;
    verts->at(7) = lr + osg::Vec3(0,0,offset);

    // right
    verts->at(8) = ur - osg::Vec3(offset,0,0);
    verts->at(9) = lr - osg::Vec3(offset,0,0);
    verts->at(10) = lr;
    verts->at(11) = ur;

    // top
    verts->at(12) = ul;
    verts->at(13) = ul - osg::Vec3(0,0,offset);
    verts->at(14) = ur - osg::Vec3(0,0,offset);
    verts->at(15) = ur;

    verts->dirty();
    _selectGeom->getBoundingBox();
}
