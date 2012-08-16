#include "GraphObject.h"

#include <cvrKernel/ComController.h>

#include <iostream>
#include <sstream>
#include <cstdlib>

using namespace cvr;

GraphObject::GraphObject(mysqlpp::Connection * conn, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : TiledWallSceneObject(name,navigation,movable,clip,contextMenu,showBounds)
{
    _conn = conn;
    _graph = new DataGraph();
    _graph->setDisplaySize(width,height);

    setBoundsCalcMode(SceneObject::MANUAL);
    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);
}

GraphObject::~GraphObject()
{

}

bool GraphObject::addGraph(std::string name)
{
    for(int i = 0; i < _nameList.size(); i++)
    {
	if(name == _nameList[i])
	{
	    return false;
	}
    }

    osg::ref_ptr<osg::Vec3Array> points;
    osg::ref_ptr<osg::Vec4Array> colors;

    struct graphData
    {
	char displayName[256];
	char units[256];
	float minValue;
	float maxValue;
	time_t minTime;
	time_t maxTime;
	int numPoints;
	bool valid;
    };

    struct graphData gd;

    if(ComController::instance()->isMaster())
    {
	if(_conn)
	{
	    std::stringstream mss;
	    mss << "select * from Measure where name = \"" << name << "\";";

	    //std::cerr << "Query: " << mss.str() << std::endl;

	    mysqlpp::Query metaQuery = _conn->query(mss.str().c_str());
	    mysqlpp::StoreQueryResult metaRes = metaQuery.store();

	    if(!metaRes.num_rows())
	    {
		std::cerr << "Meta Data query result empty for value: " << name << std::endl;
		gd.valid = false;
	    }
	    else
	    {

		int measureId = atoi(metaRes[0]["measure_id"].c_str());

		std::stringstream qss;
		qss << "select Measurement.timestamp, unix_timestamp(Measurement.timestamp) as utime, Measurement.value from Measurement inner join Measure on Measurement.measure_id = Measure.measure_id where Measure.measure_id = \"" << measureId << "\" order by utime;";

		//std::cerr << "Query: " << qss.str() << std::endl;

		mysqlpp::Query query = _conn->query(qss.str().c_str());
		mysqlpp::StoreQueryResult res;
		res = query.store();
		//std::cerr << "Num Rows: " << res.num_rows() << std::endl;
		if(!res.num_rows())
		{
		    std::cerr << "Empty query result for name: " << name << " id: " << measureId << std::endl;
		    gd.valid = false;
		}
		else
		{

		    points = new osg::Vec3Array(res.num_rows());
		    colors = new osg::Vec4Array(res.num_rows());

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

		    // remove range size of zero
		    if(minval == maxval)
		    {
			minval -= 1.0;
			maxval += 1.0;
		    }

		    if(mint == maxt)
		    {
			mint -= 86400;
			maxt += 86400;
		    }

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
		    gd.valid = true;

		    strncpy(gd.displayName, metaRes[0]["display_name"].c_str(), 255);
		    strncpy(gd.units, metaRes[0]["units"].c_str(), 255);
		    gd.minValue = minval;
		    gd.maxValue = maxval;
		    gd.minTime = mint;
		    gd.maxTime = maxt;
		    gd.numPoints = res.num_rows();
		}
	    }
	}
	else
	{
	    std::cerr << "No database connection." << std::endl;
	    gd.valid = false;
	}

	ComController::instance()->sendSlaves(&gd,sizeof(struct graphData));
	if(gd.valid)
	{
	    ComController::instance()->sendSlaves((void*)points->getDataPointer(),points->size()*sizeof(osg::Vec3));
	    ComController::instance()->sendSlaves((void*)colors->getDataPointer(),colors->size()*sizeof(osg::Vec4));
	}
    }
    else
    {
	ComController::instance()->readMaster(&gd,sizeof(struct graphData));
	if(gd.valid)
	{
	    osg::Vec3 * pointData = new osg::Vec3[gd.numPoints];
	    osg::Vec4 * colorData = new osg::Vec4[gd.numPoints];
	    ComController::instance()->readMaster(pointData,gd.numPoints*sizeof(osg::Vec3));
	    ComController::instance()->readMaster(colorData,gd.numPoints*sizeof(osg::Vec4));
	    points = new osg::Vec3Array(gd.numPoints,pointData);
	    colors = new osg::Vec4Array(gd.numPoints,colorData);
	}
    }

    if(gd.valid)
    {
	_graph->addGraph(gd.displayName, points, POINTS_WITH_LINES, "Time", gd.units, osg::Vec4(0,1.0,0,1.0),colors);
	_graph->setZDataRange(gd.displayName,gd.minValue,gd.maxValue);
	_graph->setXDataRangeTimestamp(gd.displayName,gd.minTime,gd.maxTime);
	addChild(_graph->getGraphRoot());
	_nameList.push_back(name);
    }
    //std::cerr << "Graph added" << std::endl;

    return gd.valid;
}

void GraphObject::setGraphSize(float width, float height)
{
    _graph->setDisplaySize(width,height);

    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);
}

time_t GraphObject::getMaxTimestamp()
{
    std::vector<std::string> names;
    _graph->getGraphNameList(names);

    time_t max;
    if(names.size())
    {
	max = _graph->getMaxTimestamp(names[0]);
    }
    else
    {
	max = 0;
    }

    for(int i = 1; i < names.size(); i++)
    {
	time_t temp = _graph->getMaxTimestamp(names[i]);
	if(temp > max)
	{
	    max = temp;
	}
    }

    return max;
}

time_t GraphObject::getMinTimestamp()
{
    std::vector<std::string> names;
    _graph->getGraphNameList(names);

    time_t min;
    if(names.size())
    {
	min = _graph->getMinTimestamp(names[0]);
    }
    else
    {
	min = 0;
    }

    for(int i = 1; i < names.size(); i++)
    {
	time_t temp = _graph->getMinTimestamp(names[i]);
	if(temp && temp < min)
	{
	    min = temp;
	}
    }

    return min;
}
