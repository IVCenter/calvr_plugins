#include "StrainHMObject.h"

#include <cvrKernel/ComController.h>
#include <cvrKernel/PluginHelper.h>

#include <iostream>
#include <sstream>
#include <ctime>
#include <climits>

using namespace cvr;

StrainHMObject::StrainHMObject(mysqlpp::Connection * conn, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : LayoutTypeObject(name,navigation,movable,clip,contextMenu,showBounds)
{
    _conn = conn;
    _graph = new HeatMapGraph(width,height);
    _graph->setScaleType(HMAS_LOG);
    addChild(_graph->getRootNode());
}

StrainHMObject::~StrainHMObject()
{
}

void StrainHMObject::setGraphSize(float width, float height)
{
    _graph->setDisplaySize(width,height);
}

bool StrainHMObject::setGraph(std::string title, std::string patientName, int patientid, int taxid, osg::Vec4 color)
{
    std::stringstream queryss;

    queryss << "select value, unix_timestamp(timestamp) as timestamp from EcoliShigella_Measurement where patient_id = " << patientid << " and taxonomy_id = " << taxid << " order by timestamp;";

    struct StrainData
    {
	float value;
	time_t timestamp;
    };

    StrainData * sdata = NULL;
    int dataSize = 0;

    if(ComController::instance()->isMaster())
    {
	if(_conn)
	{
	    mysqlpp::Query strainQuery = _conn->query(queryss.str().c_str());
	    mysqlpp::StoreQueryResult strainResult = strainQuery.store();

	    dataSize = strainResult.num_rows();
	    if(dataSize)
	    {
		sdata = new struct StrainData[dataSize];

		for(int i = 0; i < dataSize; ++i)
		{
		    sdata[i].value = atof(strainResult[i]["value"].c_str());
		    sdata[i].timestamp = atol(strainResult[i]["timestamp"].c_str());
		}
	    }
	}
	else
	{
	    std::cerr << "No Database connection." << std::endl;
	}

	ComController::instance()->sendSlaves(&dataSize,sizeof(int));
	if(dataSize)
	{
	    ComController::instance()->sendSlaves(sdata,dataSize*sizeof(struct StrainData));
	}
    }
    else
    {
	ComController::instance()->readMaster(&dataSize,sizeof(int));
	if(dataSize)
	{
	    sdata = new struct StrainData[dataSize];
	    ComController::instance()->readMaster(sdata,dataSize*sizeof(struct StrainData));
	}
    }

    bool graphValid = false;

    if(dataSize)
    {
	std::vector<std::string> labels;
	std::vector<float> values;
	std::vector<osg::Vec4> colors;

	float minValue = FLT_MAX;
	float maxValue = FLT_MIN;
	for(int i = 0; i < dataSize; ++i)
	{
	    char temp[1024];

	    strftime(temp,1023,"%F",localtime(&sdata[i].timestamp));

	    labels.push_back(temp);
	    values.push_back(sdata[i].value);
	    colors.push_back(color);

	    if(sdata[i].value > maxValue)
	    {
		maxValue = sdata[i].value;
	    }
	    if(sdata[i].value < minValue)
	    {
		minValue = sdata[i].value;
	    }
	}

	if(minValue < 0.00003)
	{
	    std::cerr << "Graph discarded for low range." << std::endl;
	    if(sdata)
	    {
		delete[] sdata;
	    }
	    return false;
	}

	std::stringstream titless;
	titless << patientName << " : " << title;

	graphValid = _graph->setGraph(titless.str(),labels,values,minValue,maxValue,0.1,1.0,colors);
    }

    PluginHelper::registerSceneObject(this,"FuturePatient");
    attachToScene();

    if(sdata)
    {
	delete[] sdata;
    }

    return graphValid;
}

float StrainHMObject::getGraphMaxValue()
{
    return _graph->getMaxValue();
}

float StrainHMObject::getGraphMinValue()
{
    return _graph->getMinValue();
}

float StrainHMObject::getGraphDisplayRangeMax()
{
    return _graph->getMaxDisplayValue();
}

float StrainHMObject::getGraphDisplayRangeMin()
{
    return _graph->getMinDisplayValue();
}

void StrainHMObject::setGraphDisplayRange(float min, float max)
{
    _graph->setDisplayRange(min,max);
}

void StrainHMObject::resetGraphDisplayRange()
{
    _graph->resetDisplayRange();
}
