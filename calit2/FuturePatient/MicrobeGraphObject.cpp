#include "MicrobeGraphObject.h"
#include "GraphLayoutObject.h"

#include <cvrKernel/ComController.h>
#include <cvrInput/TrackingManager.h>
#include <cvrUtil/OsgMath.h>

#include <iostream>
#include <sstream>
#include <cstdlib>

using namespace cvr;

MicrobeGraphObject::MicrobeGraphObject(mysqlpp::Connection * conn, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : LayoutTypeObject(name,navigation,movable,clip,contextMenu,showBounds)
{
    _conn = conn;

    setBoundsCalcMode(SceneObject::MANUAL);
    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);

    _width = width;
    _height = height;

    _graph = new GroupedBarGraph(width,height);
}

MicrobeGraphObject::~MicrobeGraphObject()
{
}

void MicrobeGraphObject::setGraphSize(float width, float height)
{
    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);

    _graph->setDisplaySize(width,height);
}

void MicrobeGraphObject::setColor(osg::Vec4 color)
{
    _graph->setColor(color);
}

float MicrobeGraphObject::getGraphMaxValue()
{
    return _graph->getDataMax();
}

float MicrobeGraphObject::getGraphMinValue()
{
    return _graph->getDataMin();
}

float MicrobeGraphObject::getGraphDisplayRangeMax()
{
    return _graph->getDisplayRangeMax();
}

float MicrobeGraphObject::getGraphDisplayRangeMin()
{
    return _graph->getDisplayRangeMin();
}

void MicrobeGraphObject::setGraphDisplayRange(float min, float max)
{
    _graph->setDisplayRange(min,max);
}

void MicrobeGraphObject::resetGraphDisplayRange()
{
    _graph->setDisplayRange(_graph->getDisplayRangeMin(),_graph->getDisplayRangeMax());
}

void MicrobeGraphObject::selectMicrobes(std::string & group, std::vector<std::string> & keys)
{
    _graph->selectItems(group,keys);
}

bool MicrobeGraphObject::processEvent(InteractionEvent * ie)
{
    if(ie->asTrackedButtonEvent() && (ie->getInteraction() == BUTTON_DOWN || ie->getInteraction() == BUTTON_DOUBLE_CLICK))
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

	layout->selectMicrobes(selectedGroup,selectedKeys);
	if(clickUsed)
	{
	    return true;
	}
    }

    return TiledWallSceneObject::processEvent(ie);
}

void MicrobeGraphObject::updateCallback(int handID, const osg::Matrix & mat)
{
    if(TrackingManager::instance()->getHandTrackerType(handID) == TrackerBase::POINTER)
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

void MicrobeGraphObject::leaveCallback(int handID)
{
    if(TrackingManager::instance()->getHandTrackerType(handID) == TrackerBase::POINTER)
    {
	_graph->clearHoverText();
    }
}

bool MicrobeGraphObject::setGraph(std::string title, int patientid, std::string testLabel, int microbes)
{
    _graphTitle = title + " - " + testLabel;
    std::stringstream valuess, orderss;

    valuess << "select * from (select Microbes.description, Microbes.phylum, Microbes.species, Microbe_Measurement.value from  Microbe_Measurement inner join Microbes on Microbe_Measurement.taxonomy_id = Microbes.taxonomy_id where Microbe_Measurement.patient_id = \"" << patientid << "\" and Microbe_Measurement.timestamp = \""<< testLabel << "\" order by value desc limit " << microbes << ")t order by t.phylum, t.value desc;";

    orderss << "select t.phylum, sum(t.value) as total_value from (select Microbes.phylum, Microbe_Measurement.value from  Microbe_Measurement inner join Microbes on Microbe_Measurement.taxonomy_id = Microbes.taxonomy_id where Microbe_Measurement.patient_id = \"" << patientid << "\" and Microbe_Measurement.timestamp = \"" << testLabel << "\" order by value desc limit " << microbes << ")t group by phylum order by total_value desc;";

    return loadGraphData(valuess.str(), orderss.str());
}

bool MicrobeGraphObject::setSpecialGraph(SpecialMicrobeGraphType smgt, int microbes)
{
    std::stringstream valuess, orderss;

    switch(smgt)
    {
	case SMGT_AVERAGE:
	case SMGT_HEALTHY_AVERAGE:
	case SMGT_CROHNS_AVERAGE:
	    {

		std::string field;

		switch(smgt)
		{
		    case SMGT_AVERAGE:
			field = "average";
			_graphTitle = "UC Average";
			break;
		    case SMGT_HEALTHY_AVERAGE:
			field = "average_healthy";
			_graphTitle = "Healthy Average";
			break;
		    case SMGT_CROHNS_AVERAGE:
			field = "average_crohns";
			_graphTitle = "Crohns Average";
			break;
		    default:
			break;
		}

		valuess << "select * from (select description, phylum, species, " << field << " as value from Microbes order by value desc limit " << microbes << ")t order by t.phylum, t.value desc;";
		orderss << "select t.phylum, sum(t.value) as total_value from (select phylum, " << field << " as value from Microbes order by value desc limit " << microbes << ")t group by phylum order by total_value desc;";
		break;
	    }
	case SMGT_SRS_AVERAGE:
	case SMGT_SRX_AVERAGE:
	{

	    std::string regexp;
	    switch(smgt)
	    {
		case SMGT_SRS_AVERAGE:
		    regexp = "^SRS";
		    _graphTitle = "SRS Average";
		    break;
		case SMGT_SRX_AVERAGE:
		    regexp = "^SRX";
		    _graphTitle = "SRX Average";
		    break;
		default:
		    break;
	    }

	    valuess << "select * from (select Microbes.description, Microbes.phylum, Microbes.species, avg(Microbe_Measurement.value) as value from Microbe_Measurement inner join Microbes on Microbe_Measurement.taxonomy_id = Microbes.taxonomy_id inner join Patient on Microbe_Measurement.patient_id = Patient.patient_id where Patient.last_name regexp '" << regexp << "' group by species order by value desc limit " << microbes << ")t order by t.phylum, t.value desc;";
	    orderss << "select t.phylum, sum(t.value) as total_value from (select Microbes.description, Microbes.phylum, Microbes.species, avg(Microbe_Measurement.value) as value from Microbe_Measurement inner join Microbes on Microbe_Measurement.taxonomy_id = Microbes.taxonomy_id inner join Patient on Microbe_Measurement.patient_id = Patient.patient_id where Patient.last_name regexp '" << regexp << "' group by species order by value desc limit " << microbes << ")t group by phylum order by total_value desc;";

	    break;
	}
	default:
	    return false;
    }

    return loadGraphData(valuess.str(), orderss.str());
}

bool MicrobeGraphObject::loadGraphData(std::string valueQuery, std::string orderQuery)
{
    _graphData.clear();
    _graphOrder.clear();

    struct MicrobeDataHeader
    {
	int numDataValues;
	int numOrderValues;
	bool valid;
    };

    struct MicrobeDataValue
    {
	char phylum[1024];
	char species[1024];
	char description[1024];
	float value;
    };

    struct MicrobeOrderValue
    {
	char group[1024];
    };

    MicrobeDataHeader header;
    header.numDataValues = 0;
    header.numOrderValues = 0;
    header.valid = false;

    MicrobeDataValue * data = NULL;
    MicrobeOrderValue * order = NULL;

    if(ComController::instance()->isMaster())
    {
	if(_conn)
	{
	    mysqlpp::Query valueq = _conn->query(valueQuery.c_str());
	    mysqlpp::StoreQueryResult valuer = valueq.store();

	    header.numDataValues = valuer.num_rows();

	    if(valuer.num_rows())
	    {
		data = new MicrobeDataValue[valuer.num_rows()];

		for(int i = 0; i < valuer.num_rows(); ++i)
		{
		    strncpy(data[i].phylum,valuer[i]["phylum"].c_str(),1023);
		    strncpy(data[i].species,valuer[i]["species"].c_str(),1023);
		    strncpy(data[i].description,valuer[i]["description"].c_str(),1023);
		    data[i].value = atof(valuer[i]["value"]);
		}
	    }

	    mysqlpp::Query orderq = _conn->query(orderQuery.c_str());
	    mysqlpp::StoreQueryResult orderr = orderq.store();

	    header.numOrderValues = orderr.num_rows();

	    if(orderr.num_rows())
	    {
		order = new MicrobeOrderValue[orderr.num_rows()];

		for(int i = 0; i < orderr.num_rows(); ++i)
		{
		    strncpy(order[i].group,orderr[i]["phylum"].c_str(),1023);
		}
	    }

	    header.valid = true;
	}
	else
	{
	    std::cerr << "No Database connection." << std::endl;
	}

	ComController::instance()->sendSlaves(&header, sizeof(struct MicrobeDataHeader));
	if(header.valid)
	{
	    if(header.numDataValues)
	    {
		ComController::instance()->sendSlaves(data, header.numDataValues*sizeof(struct MicrobeDataValue));
	    }
	    if(header.numOrderValues)
	    {
		ComController::instance()->sendSlaves(order, header.numOrderValues*sizeof(struct MicrobeOrderValue));
	    }
	}
    }
    else
    {
	ComController::instance()->readMaster(&header, sizeof(struct MicrobeDataHeader));
	if(header.valid)
	{
	    if(header.numDataValues)
	    {
		data = new struct MicrobeDataValue[header.numDataValues];

		ComController::instance()->readMaster(data, header.numDataValues*sizeof(struct MicrobeDataValue));
	    }
	    if(header.numOrderValues)
	    {
		order = new struct MicrobeOrderValue[header.numOrderValues];

		ComController::instance()->readMaster(order, header.numOrderValues*sizeof(struct MicrobeOrderValue));
	    }
	}
    }

    if(!header.valid)
    {
	return false;
    }

    for(int i = 0; i < header.numDataValues; ++i)
    {
	if(_graphData.find(data[i].phylum) == _graphData.end())
	{
	    _graphData[data[i].phylum] = std::vector<std::pair<std::string,float> >();
	}

	_graphData[data[i].phylum].push_back(std::pair<std::string,float>(data[i].species,data[i].value));
    }

    for(int i = 0; i < header.numOrderValues; ++i)
    {
	_graphOrder.push_back(order[i].group);
    }
    
    bool graphValid = _graph->setGraph(_graphTitle, _graphData, _graphOrder, BGAT_LOG, "Value", "", "phylum / species",osg::Vec4(1.0,0,0,1));

    if(graphValid)
    {
	addChild(_graph->getRootNode());
    }

    if(data)
    {
	delete[] data;
    }
    if(order)
    {
	delete[] order;
    }

    return graphValid;
}
