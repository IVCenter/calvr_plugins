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

MicrobeScatterGraphObject::MicrobeScatterGraphObject(mysqlpp::Connection * conn, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : LayoutTypeObject(name,navigation,movable,clip,contextMenu,showBounds), SelectableObject()
{
    _graph = new GroupedScatterPlot(width,height);
    _conn = conn;

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

bool MicrobeScatterGraphObject::setGraph(std::string title, std::string primaryPhylum, std::string secondaryPhylum)
{
    if(!_dataInit)
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

    return true;
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

void MicrobeScatterGraphObject::selectPatients(std::string & group, std::vector<std::string> & patients)
{
    _graph->selectPoints(group,patients);
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
	if(_conn)
	{
	    std::string sizesQuery = "SELECT q.phylum, count(q.value) as count from (SELECT Microbes.phylum, Patient.last_name, unix_timestamp(Microbe_Measurement.timestamp) as timestamp, sum(Microbe_Measurement.value) as value FROM Microbe_Measurement INNER JOIN Patient ON Microbe_Measurement.patient_id = Patient.patient_id INNER JOIN Microbes ON Microbe_Measurement.taxonomy_id = Microbes.taxonomy_id GROUP BY Patient.last_name, Microbe_Measurement.timestamp, Microbes.phylum ORDER BY Microbes.phylum, Patient.last_name, Microbe_Measurement.timestamp)q group by q.phylum order by q.phylum;";
	    std::string orderQuery = "SELECT q.phylum, q.last_name, q.timestamp from (SELECT Microbes.phylum, Patient.last_name, unix_timestamp(Microbe_Measurement.timestamp) as timestamp, sum(Microbe_Measurement.value) as value FROM Microbe_Measurement INNER JOIN Patient ON Microbe_Measurement.patient_id = Patient.patient_id INNER JOIN Microbes ON Microbe_Measurement.taxonomy_id = Microbes.taxonomy_id GROUP BY Patient.last_name, Microbe_Measurement.timestamp, Microbes.phylum ORDER BY Microbes.phylum, Patient.last_name, Microbe_Measurement.timestamp)q order by q.phylum, q.last_name, q.timestamp;";
	    std::string query = "SELECT Microbes.phylum, Patient.last_name, unix_timestamp(Microbe_Measurement.timestamp) as timestamp, sum(Microbe_Measurement.value) as value FROM Microbe_Measurement INNER JOIN Patient ON Microbe_Measurement.patient_id = Patient.patient_id INNER JOIN Microbes ON Microbe_Measurement.taxonomy_id = Microbes.taxonomy_id GROUP BY Patient.last_name, Microbe_Measurement.timestamp, Microbes.phylum ORDER BY Microbes.phylum, Patient.last_name, Microbe_Measurement.timestamp;";

	    std::cerr << "Starting first query." << std::endl;

	    mysqlpp::Query sQuery = _conn->query(sizesQuery.c_str());
	    mysqlpp::StoreQueryResult sRes = sQuery.store();

	    numPhylum = sRes.num_rows();

	    std::cerr << "Num Phylum: " << numPhylum << std::endl;

	    if(numPhylum)
	    {
		phylums = new struct phylumEntry[numPhylum];
		orderSizes = new int[numPhylum];
		for(int i = 0; i < numPhylum; ++i)
		{
		    strncpy(phylums[i].name,sRes[i]["phylum"].c_str(),1023);
		    orderSizes[i] = atoi(sRes[i]["count"].c_str());
		    orderTotal += orderSizes[i];
		}

		std::cerr << "OrderTotal: " << orderTotal << std::endl;

		mysqlpp::Query oQuery = _conn->query(orderQuery.c_str());
		mysqlpp::StoreQueryResult oRes = oQuery.store();

		std::cerr << "Second done." << std::endl;

		if(oRes.num_rows() == orderTotal)
		{
		    order = new struct orderEntry[orderTotal];
		    for(int i = 0; i < orderTotal; ++i)
		    {
			strncpy(order[i].name,oRes[i]["last_name"].c_str(),1023);
			order[i].timestamp = atol(oRes[i]["timestamp"].c_str());
		    }
		}
		else
		{
		    std::cerr << "Number of order rows different than expected. Wanted: " << orderTotal << " Got: " << oRes.num_rows() << std::endl;
		    orderTotal = 0;
		}

		std::cerr << "Last started." << std::endl;

		mysqlpp::Query vQuery = _conn->query(query.c_str());
		mysqlpp::StoreQueryResult vRes = vQuery.store();

		std::cerr << "Last done." << std::endl;

		if(vRes.num_rows() == orderTotal)
		{
		    data = new float[orderTotal];
		    for(int i = 0; i < orderTotal; ++i)
		    {
			data[i] = atof(vRes[i]["value"].c_str());
		    }
		}
		else
		{
		    std::cerr << "Number of values different than expected. Wanted: " << orderTotal << " Got: " << vRes.num_rows() << std::endl;
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
    _selectGeom->getBound();
}
