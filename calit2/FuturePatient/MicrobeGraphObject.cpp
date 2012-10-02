#include "MicrobeGraphObject.h"

#include <cvrKernel/ComController.h>

#include <iostream>
#include <sstream>
#include <cstdlib>

using namespace cvr;

MicrobeGraphObject::MicrobeGraphObject(mysqlpp::Connection * conn, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : TiledWallSceneObject(name,navigation,movable,clip,contextMenu,showBounds)
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

bool MicrobeGraphObject::setGraph(std::string title, int patientid, std::string testLabel)
{
    _graphTitle = title;
    std::stringstream valuess, orderss;

    valuess << "select * from (select Microbes.description, Microbes.phylum, Microbes.species, Microbe_Measurement.value from  Microbe_Measurement inner join Microbes on Microbe_Measurement.taxonomy_id = Microbes.taxonomy_id where Microbe_Measurement.patient_id = \"" << patientid << "\" and Microbe_Measurement.timestamp = \""<< testLabel << "\" order by value desc limit 25)t order by t.phylum, t.value desc;";

    orderss << "select t.phylum, sum(t.value) as total_value from (select Microbes.phylum, Microbe_Measurement.value from  Microbe_Measurement inner join Microbes on Microbe_Measurement.taxonomy_id = Microbes.taxonomy_id where Microbe_Measurement.patient_id = \"" << patientid << "\" and Microbe_Measurement.timestamp = \"" << testLabel << "\" order by value desc limit 25)t group by phylum order by total_value desc;";

    return loadGraphData(valuess.str(), orderss.str());
}

bool MicrobeGraphObject::setSpecialGraph(SpecialMicrobeGraphType smgt)
{
    std::string field;

    switch(smgt)
    {
	case SMGT_AVERAGE:
	    field = "average";
	    _graphTitle = "Average";
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

    std::stringstream valuess, orderss;
    valuess << "select * from (select description, phylum, species, " << field << " as value from Microbes order by value desc limit 25)t order by t.phylum, t.value desc;";
    orderss << "select t.phylum, sum(t.value) as total_value from (select phylum, " << field << " as value from Microbes order by value desc limit 25)t group by phylum order by total_value desc;";

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
    
    bool graphValid = _graph->setGraph(_graphTitle, _graphData, _graphOrder, BGAT_LOG, _graphTitle, "%", "phylum / species",osg::Vec4(1.0,0,0,1));

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
