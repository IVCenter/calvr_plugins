#include "MicrobeScatterGraphObject.h"

#include <cvrKernel/ComController.h>

#include <iostream>
#include <sstream>
#include <cstring>

using namespace cvr;

MicrobeScatterGraphObject::MicrobeScatterGraphObject(mysqlpp::Connection * conn, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : LayoutTypeObject(name,navigation,movable,clip,contextMenu,showBounds)
{
    _graph = new GroupedScatterPlot(width,height);
    _conn = conn;

    setBoundsCalcMode(SceneObject::MANUAL);
    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);
}

MicrobeScatterGraphObject::~MicrobeScatterGraphObject()
{
}

bool MicrobeScatterGraphObject::setGraph(std::string title, std::string primaryPhylum, std::string secondaryPhylum)
{
    struct DataPoint
    {
	char name[1024];
	time_t timestamp;
	float firstValue;
	float secondValue;
    };

    // Data order: Smarr, CD, UC, HE

    int numGroups = 4;
    int * sizes = new int[numGroups];
    struct DataPoint ** data = new struct DataPoint*[numGroups];

    for(int i = 0; i < numGroups; ++i)
    {
	sizes[i] = 0;
	data[i] = NULL;
    }

    if(ComController::instance()->isMaster())
    {
	if(_conn)
	{
	    std::string queryPart1 = "SELECT Patient.last_name, unix_timestamp(Microbe_Measurement.timestamp) as timestamp, sum(Microbe_Measurement.value) as value FROM Microbe_Measurement INNER JOIN Patient ON Microbe_Measurement.patient_id = Patient.patient_id AND Patient.last_name regexp '";
	    std::string queryPart2 = "' INNER JOIN Microbes ON Microbe_Measurement.taxonomy_id = Microbes.taxonomy_id WHERE Microbes.phylum = '";
	    std::string queryPart3 = "' GROUP BY Patient.last_name, Microbe_Measurement.timestamp ORDER BY Patient.last_name, Microbe_Measurement.timestamp;";
	    
	    for(int i = 0; i < numGroups; ++i)
	    {
		std::string regexp = "None";

		switch(i)
		{
		    case 0:
			regexp = "^Smarr";
			break;
		    case 1:
			regexp = "^CD-";
			break;
		    case 2:
			regexp = "^UC-";
			break;
		    case 3:
			regexp = "^HE-";
			break;
		    default:
			break;
		}

		std::stringstream queryPriSS, querySecSS;
		queryPriSS << queryPart1 << regexp << queryPart2 << primaryPhylum << queryPart3;
		querySecSS << queryPart1 << regexp << queryPart2 << secondaryPhylum << queryPart3;

		mysqlpp::Query queryPri = _conn->query(queryPriSS.str().c_str());
		mysqlpp::Query querySec = _conn->query(querySecSS.str().c_str());
		mysqlpp::StoreQueryResult resPri = queryPri.store();
		mysqlpp::StoreQueryResult resSec = querySec.store();

		if(resPri.num_rows() == resSec.num_rows())
		{
		    sizes[i] = resPri.num_rows();

		    if(resPri.num_rows())
		    {
			data[i] = new struct DataPoint[resPri.num_rows()];
			for(int j = 0; j < resPri.num_rows(); ++j)
			{
			    strncpy(data[i][j].name,resPri[j]["last_name"].c_str(),1023);
			    data[i][j].timestamp = atol(resPri[j]["timestamp"].c_str());
			    data[i][j].firstValue = atof(resPri[j]["value"].c_str());
			    data[i][j].secondValue = atof(resSec[j]["value"].c_str());
			}
		    }
		}
		else
		{
		    std::cerr << "Number of rows do not match!?" << std::endl;
		}
	    }
	}

	ComController::instance()->sendSlaves(sizes,sizeof(int)*4);
	for(int i = 0; i < numGroups; ++i)
	{
	    if(sizes[i])
	    {
		ComController::instance()->sendSlaves(data[i],sizeof(struct DataPoint)*sizes[i]);
	    }
	}
    }
    else
    {
	ComController::instance()->readMaster(sizes,sizeof(int)*4);

	for(int i = 0; i < numGroups; ++i)
	{
	    if(sizes[i])
	    {
		data[i] = new struct DataPoint[sizes[i]];
		ComController::instance()->readMaster(data[i],sizeof(struct DataPoint)*sizes[i]);
	    }
	}
    }

    _graph->setLabels(title,primaryPhylum,secondaryPhylum);
    _graph->setAxisTypes(GSP_LOG,GSP_LOG);

    for(int i = 0; i < numGroups; ++i)
    {
	std::vector<std::pair<float,float> > dataList;
	for(int j = 0; j < sizes[i]; ++j)
	{
	    if(data[i][j].firstValue > 0.0 && data[i][j].secondValue > 0.0)
	    {
		dataList.push_back(std::pair<float,float>(data[i][j].firstValue,data[i][j].secondValue));
	    }
	}
	if(dataList.size())
	{
	    _graph->addGroup(i,dataList);
	}
    }

    addChild(_graph->getRootNode());

    for(int i = 0; i < numGroups; ++i)
    {
	if(data[i])
	{
	    delete[] data[i];
	}
    }
    delete[] data;

    return true;
}

void MicrobeScatterGraphObject::setGraphSize(float width, float height)
{
    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);

    _graph->setDisplaySize(width,height);
}
