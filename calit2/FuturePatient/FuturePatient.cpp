#include "FuturePatient.h"
#include "DataGraph.h"

#include <cvrKernel/PluginHelper.h>

#include <cstdlib>
#include <iostream>
#include <sstream>

#include <mysql++/mysql++.h>

using namespace cvr;

CVRPLUGIN(FuturePatient)

FuturePatient::FuturePatient()
{
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

    makeGraph("SIga");

    return true;
}

void FuturePatient::makeGraph(std::string name)
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
}
