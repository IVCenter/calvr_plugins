#include "StrainHMObject.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/ComController.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrInput/TrackingManager.h>
#include <cvrUtil/OsgMath.h>

#include <iostream>
#include <sstream>
#include <ctime>
#include <climits>

using namespace cvr;

StrainHMObject::StrainHMObject(DBManager * dbm, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : LayoutTypeObject(name,navigation,movable,clip,contextMenu,showBounds)
{
    _dbm = dbm;

    setBoundsCalcMode(SceneObject::MANUAL);
    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);

    _graph = new HeatMapGraph(width,height);
    _graph->setScaleType(HMAS_LOG);
    _desktopMode = ConfigManager::getBool("Plugin.FuturePatient.DesktopMode",false);
    addChild(_graph->getRootNode());
}

StrainHMObject::~StrainHMObject()
{
}

void StrainHMObject::setGraphSize(float width, float height)
{
    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);

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
	if(_dbm)
	{
	    DBMQueryResult result;

	    _dbm->runQuery(queryss.str(),result);

	    dataSize = result.numRows();
	    if(dataSize)
	    {
		sdata = new struct StrainData[dataSize];

		for(int i = 0; i < dataSize; ++i)
		{
		    sdata[i].value = atof(result(i,"value").c_str());
		    sdata[i].timestamp = atol(result(i,"timestamp").c_str());
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

void StrainHMObject::updateCallback(int handID, const osg::Matrix & mat)
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

void StrainHMObject::leaveCallback(int handID)
{
    if((_desktopMode && TrackingManager::instance()->getHandTrackerType(handID) == TrackerBase::MOUSE) ||
	(!_desktopMode && TrackingManager::instance()->getHandTrackerType(handID) == TrackerBase::POINTER))
    {
	_graph->clearHoverText();
    }
}
