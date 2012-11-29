#include "SymptomGraphObject.h"

#include <cvrKernel/ComController.h>

#include <iostream>
#include <sstream>

using namespace cvr;

SymptomGraphObject::SymptomGraphObject(mysqlpp::Connection * conn, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : LayoutTypeObject(name,navigation,movable,clip,contextMenu,showBounds)
{
    _conn = conn;
    _graph = new TimeRangeDataGraph();
    _graph->setDisplaySize(width,height);

    addChild(_graph->getGraphRoot());

    setBoundsCalcMode(SceneObject::MANUAL);
    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);
}

SymptomGraphObject::~SymptomGraphObject()
{
}

bool SymptomGraphObject::addGraph(std::string name)
{

    struct timeRange
    {
	time_t start;
	time_t end;
    };

    int numRanges = 0;
    struct timeRange * ranges = NULL;

    if(ComController::instance()->isMaster())
    {
	if(_conn)
	{
	    std::stringstream qss;
	    qss << "select unix_timestamp(start_timestamp) as start, unix_timestamp(end_timestamp) as end from Event where patient_id = \"1\" and name = \"" << name << "\";";

	    mysqlpp::Query query = _conn->query(qss.str().c_str());
	    mysqlpp::StoreQueryResult result = query.store();

	    numRanges = result.num_rows();
	    if(numRanges)
	    {
		ranges = new struct timeRange[numRanges];
		for(int i = 0; i < numRanges; ++i)
		{
		    ranges[i].start = atol(result[i]["start"].c_str());
		    ranges[i].end = atol(result[i]["end"].c_str());
		}
	    }
	}

	ComController::instance()->sendSlaves(&numRanges,sizeof(int));
	if(numRanges)
	{
	    ComController::instance()->sendSlaves(ranges,numRanges*sizeof(struct timeRange));
	}
    }
    else
    {
	ComController::instance()->readMaster(&numRanges,sizeof(int));
	if(numRanges)
	{
	    ranges = new struct timeRange[numRanges];
	    ComController::instance()->readMaster(ranges,numRanges*sizeof(struct timeRange));
	}
    }

    if(numRanges)
    {
	std::vector<std::pair<time_t,time_t> > rangeList;

	for(int i = 0; i < numRanges; ++i)
	{
	    rangeList.push_back(std::pair<time_t,time_t>(ranges[i].start,ranges[i].end));
	}

	_graph->addGraph(name,rangeList);

	delete[] ranges;

	return true;
    }
    else
    {
	std::cerr << "Warning: no entries for symptom: " << name << std::endl;
	return false;
    }
}

void SymptomGraphObject::setGraphSize(float width, float height)
{
    _graph->setDisplaySize(width,height);

    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);
}

void SymptomGraphObject::setGraphDisplayRange(time_t start, time_t end)
{
    _graph->setDisplayRange(start,end);
}

void SymptomGraphObject::resetGraphDisplayRange()
{
    time_t start, end;
    start = _graph->getMinTimestamp();
    end = _graph->getMaxTimestamp();
    _graph->setDisplayRange(start,end);
}

void SymptomGraphObject::getGraphDisplayRange(time_t & start, time_t & end)
{
    _graph->getDisplayRange(start,end);
}

time_t SymptomGraphObject::getMaxTimestamp()
{
    return _graph->getMaxTimestamp();
}

time_t SymptomGraphObject::getMinTimestamp()
{
    return _graph->getMinTimestamp();
}
