#include "MicrobePointLineObject.h"
#include "GraphGlobals.h"
#include "GraphLayoutObject.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrInput/TrackingManager.h>
#include <cvrKernel/ComController.h>
#include <cvrUtil/OsgMath.h>

#include <iostream>
#include <sstream>

using namespace cvr;

MicrobePointLineObject::MicrobePointLineObject(DBManager * dbm, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : LayoutTypeObject(name,navigation,movable,clip,contextMenu,showBounds)
{
    _dbm = dbm;

    setBoundsCalcMode(SceneObject::MANUAL);
    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);

    _desktopMode = ConfigManager::getBool("Plugin.FuturePatient.DesktopMode",false);
    
    _graph = new PointLineGraph(width,height);
    _graph->setAxisType(PLG_LOG);
    _graph->setColorMapping(GraphGlobals::getPatientColorMap());
}

MicrobePointLineObject::~MicrobePointLineObject()
{
}

enum patType
{
    SMARR=0,
    CROHNS,
    UC,
    HEALTHY
};

struct patientData
{
    patType type;
    char name[512];
    time_t timestamp;
    float data[7];
};

bool MicrobePointLineObject::setGraph(std::string microbeTableSuffix, std::string measureTableSuffix, bool expandAxis)
{
    std::vector<std::string> phylumOrder;
    // larry order
    /*phylumOrder.push_back("Bacteroidetes");
    phylumOrder.push_back("Firmicutes");
    phylumOrder.push_back("Verrucomicrobia");
    phylumOrder.push_back("Proteobacteria");
    phylumOrder.push_back("Actinobacteria");
    phylumOrder.push_back("Fusobacteria");
    phylumOrder.push_back("Euryarchaeota");*/

    phylumOrder.push_back("Bacteroidetes");
    phylumOrder.push_back("Firmicutes");
    phylumOrder.push_back("Actinobacteria");
    phylumOrder.push_back("Proteobacteria");
    phylumOrder.push_back("Verrucomicrobia");
    phylumOrder.push_back("Euryarchaeota");
    phylumOrder.push_back("Fusobacteria");

    int numPatients = 0;
    std::vector<patientData> data;

    std::string measurementTable = "Microbe_Measurement";
    measurementTable += measureTableSuffix;

    std::string microbesTable = "Microbes";
    microbesTable += microbeTableSuffix;

    std::stringstream queryss;
    queryss << "SELECT Patient.p_condition, Patient.last_name, unix_timestamp(" << measurementTable << ".timestamp) as timestamp, " << microbesTable << ".phylum, sum(" << measurementTable << ".value) as value from " << measurementTable << " inner join " << microbesTable << " on " << measurementTable << ".taxonomy_id = " << microbesTable << ".taxonomy_id inner join Patient on Patient.patient_id = " << measurementTable << ".patient_id where Patient.region = \"US\" group by Patient.last_name, " << measurementTable << ".timestamp, " << microbesTable << ".phylum;";

    std::string querystr = queryss.str();

    //std::cerr << querystr << std::endl;

    if(ComController::instance()->isMaster())
    {
	if(_dbm)
	{
	    DBMQueryResult result;

	    _dbm->runQuery(querystr,result);

	    std::map<patType,std::map<std::string,std::map<time_t,std::map<std::string,float> > > > sortMap;
	    for(int i = 0; i < result.numRows(); ++i)
	    {
		std::string condition = result(i,"p_condition").c_str();
		patType type;
		if(condition == "CD" || condition == "crohn's disease")
		{
		    type = CROHNS;
		}
		else if(condition == "healthy")
		{
		    type = HEALTHY;
		}
		else if(condition == "Larry")
		{
		    type = SMARR;
		}
		else if(condition == "UC" || condition == "ulcerous colitis")
		{
		    type = UC;
		}
		else
		{
		    continue;
		}
		sortMap[type][result(i,"last_name").c_str()][atol(result(i,"timestamp").c_str())][result(i,"phylum").c_str()] = atof(result(i,"value").c_str());
	    }

	    for(std::map<patType,std::map<std::string,std::map<time_t,std::map<std::string,float> > > >::iterator git = sortMap.begin(); git != sortMap.end(); ++git)
	    {
		for(std::map<std::string,std::map<time_t,std::map<std::string,float> > >::iterator pit = git->second.begin(); pit != git->second.end(); ++pit)
		{
		    for(std::map<time_t,std::map<std::string,float> >::iterator tsit = pit->second.begin(); tsit != pit->second.end(); ++tsit)
		    {
			patientData pd;
			pd.type = git->first;
			strncpy(pd.name,pit->first.c_str(),511);
			pd.timestamp = tsit->first;
			for(int i = 0; i < phylumOrder.size(); ++i)
			{
			    pd.data[i] = tsit->second[phylumOrder[i]];
			}
			data.push_back(pd);
		    }
		}
	    }
	    numPatients = data.size();
	}

	ComController::instance()->sendSlaves(&numPatients,sizeof(int));
	if(numPatients)
	{
	    ComController::instance()->sendSlaves(data.data(),numPatients*sizeof(struct patientData));
	}
    }
    else
    {
	ComController::instance()->readMaster(&numPatients,sizeof(int));
	if(numPatients)
	{
	    data.resize(numPatients);
	    ComController::instance()->readMaster(data.data(),numPatients*sizeof(struct patientData));
	}
    }

    std::cerr << "Got " << numPatients << " patients." << std::endl;

    if(!numPatients)
    {
	return false;
    }

    std::vector<std::string> groupNames;
    groupNames.push_back("LS");
    groupNames.push_back("Crohns");
    groupNames.push_back("UC");
    groupNames.push_back("Healthy");

    std::vector<std::vector<std::string> > patientNames;
    patientNames.push_back(std::vector<std::string>());
    patientNames.push_back(std::vector<std::string>());
    patientNames.push_back(std::vector<std::string>());
    patientNames.push_back(std::vector<std::string>());

    std::vector<std::vector<std::vector<float> > > values;
    values.push_back(std::vector<std::vector<float> >());
    values.push_back(std::vector<std::vector<float> >());
    values.push_back(std::vector<std::vector<float> >());
    values.push_back(std::vector<std::vector<float> >());

    for(int i = 0; i < data.size(); ++i)
    {
	int index = (int)data[i].type;

	char timestamp[512];
	std::string name = data[i].name;
	if(name == "Smarr")
	{
	    name = "LS";
	}
	name += " - ";
	strftime(timestamp,511,"%F",localtime(&data[i].timestamp));
	name += timestamp;

	patientNames[index].push_back(name);
	values[index].push_back(std::vector<float>());
	for(int j = 0; j < 7; ++j)
	{
	    values[index].back().push_back(data[i].data[j]);
	}
    }

    bool tret = _graph->setGraph("Phylum Line Chart",groupNames,phylumOrder,patientNames,values,expandAxis);

    if(tret)
    {
	addChild(_graph->getRootNode());
    }

    return tret;
}

void MicrobePointLineObject::objectAdded()
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

void MicrobePointLineObject::objectRemoved()
{
    GraphLayoutObject * layout = dynamic_cast<GraphLayoutObject*>(_parent);
    if(layout && layout->getPatientKeyObject())
    {
	layout->getPatientKeyObject()->unref(this);
    }
}

void MicrobePointLineObject::setGraphSize(float width, float height)
{
    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);

    _graph->setDisplaySize(width,height);
}

void MicrobePointLineObject::selectPatients(std::map<std::string,std::vector<std::string> > & selectMap)
{
    _graph->selectItems(selectMap);
}

bool MicrobePointLineObject::processEvent(cvr::InteractionEvent * ie)
{
    if(ie->asTrackedButtonEvent() && ie->asTrackedButtonEvent()->getButton() == 0 && (ie->getInteraction() == BUTTON_DOWN || ie->getInteraction() == BUTTON_DOUBLE_CLICK))
    {
	TrackedButtonInteractionEvent * tie = (TrackedButtonInteractionEvent*)ie;

	GraphLayoutObject * layout = dynamic_cast<GraphLayoutObject*>(_parent);
	if(!layout)
	{
	    return false;
	}

	std::string selectedGroup;
	std::vector<std::string> selectedKeys;

	osg::Vec3 start, end(0,1000,0);
	start = start * tie->getTransform() * getWorldToObjectMatrix();
	end = end * tie->getTransform() * getWorldToObjectMatrix();

	osg::Vec3 planePoint;
	osg::Vec3 planeNormal(0,-1,0);
	osg::Vec3 intersect;
	float w;

	bool clickUsed = false;

	if(linePlaneIntersectionRef(start,end,planePoint,planeNormal,intersect,w))
	{
	    if(_graph->processClick(intersect,selectedGroup,selectedKeys))
	    {
		clickUsed = true;
	    }
	}

	layout->selectPatients(selectedGroup,selectedKeys);
	if(clickUsed)
	{
	    return true;
	}
    }

    return FPTiledWallSceneObject::processEvent(ie);
}

void MicrobePointLineObject::updateCallback(int handID, const osg::Matrix & mat)
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

void MicrobePointLineObject::leaveCallback(int handID)
{
    if((_desktopMode && TrackingManager::instance()->getHandTrackerType(handID) == TrackerBase::MOUSE) ||
	(!_desktopMode && TrackingManager::instance()->getHandTrackerType(handID) == TrackerBase::POINTER))
    {
	_graph->clearHoverText();
    }
}
