#include "SingleMicrobeObject.h"
#include "GraphGlobals.h"
#include "GraphLayoutObject.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrInput/TrackingManager.h>
#include <cvrKernel/ComController.h>
#include <cvrUtil/OsgMath.h>

#include <osg/Material>

#include <iostream>
#include <sstream>

using namespace cvr;

SingleMicrobeObject::SingleMicrobeObject(DBManager * dbm, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : LayoutTypeObject(name,navigation,movable,clip,contextMenu,showBounds)
{
    _dbm = dbm;

    setBoundsCalcMode(SceneObject::MANUAL);
    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);

    _desktopMode = ConfigManager::getBool("Plugin.FuturePatient.DesktopMode",false);

    _graph = new GroupedBarGraph(width,height);
}

SingleMicrobeObject::~SingleMicrobeObject()
{
}

bool rankOrderSort(const std::pair<std::string, float> & first, const std::pair<std::string, float> & second)
{
    return first.second > second.second;
}

bool SingleMicrobeObject::setGraph(std::string microbe, std::string titleSuffix, int taxid, std::string microbeTableSuffix, std::string measureTableSuffix, MicrobeGraphType type, bool rankOrder, bool labels, bool firstOnly, bool groupPatients)
{
    std::string measurementTable = "Microbe_Measurement";
    measurementTable += measureTableSuffix;

    std::string microbesTable = "Microbes";
    microbesTable += microbeTableSuffix;

    struct microbeData
    {
	char name[512];
	char condition[512];
	int id;
	time_t timestamp;
	float value;
    };

    std::stringstream queryss;

    switch( type )
    {
        case MGT_SPECIES:
        {
	    queryss << "select Patient.last_name, Patient.p_condition, Patient.patient_id, unix_timestamp(" << measurementTable << ".timestamp) as timestamp, " << measurementTable << ".value from " << measurementTable << " inner join Patient on Patient.patient_id = " << measurementTable << ".patient_id where " << measurementTable << ".taxonomy_id = " << taxid << " and Patient.region = \"US\" order by p_condition, last_name, timestamp;";
            break;
        }
    
        case MGT_FAMILY:
        {
	    queryss << "select Patient.p_condition, Patient.last_name, t.patient_id, unix_timestamp(t.timestamp) as timestamp, t.value from (select " << microbesTable << ".family, " << measurementTable << ".patient_id, " << measurementTable << ".timestamp, sum(" << measurementTable << ".value) as value from "<< measurementTable << " inner join " << microbesTable << " on " << measurementTable << ".taxonomy_id = " << microbesTable << ".taxonomy_id where " << microbesTable << ".family = \"" << microbe << "\" group by " << measurementTable << ".patient_id, " << measurementTable << ".timestamp)t inner join Patient on t.patient_id = Patient.patient_id where Patient.region = \"US\" order by Patient.p_condition, Patient.last_name, t.timestamp;";
	        //queryss << "select Patient.last_name, Patient.p_condition, Patient.patient_id, unix_timestamp(" << measurementTable << ".timestamp) as timestamp, " << microbesTable << ".family, sum(" << measurementTable << ".value) as value from " << measurementTable << " inner join Patient on Patient.patient_id = " << measurementTable << ".patient_id inner join " << microbesTable << " on " << microbesTable << ".taxonomy_id = " << measurementTable << ".taxonomy_id where " << microbesTable << ".family = \"" << microbe << "\" and Patient.region = \"US\" group by patient_id, timestamp order by p_condition, last_name, timestamp;";
            break;
        }

        case MGT_GENUS:
        {
	    queryss << "select Patient.p_condition, Patient.last_name, t.patient_id, unix_timestamp(t.timestamp) as timestamp, t.value from (select " << microbesTable << ".genus, " << measurementTable << ".patient_id, " << measurementTable << ".timestamp, sum(" << measurementTable << ".value) as value from "<< measurementTable << " inner join " << microbesTable << " on " << measurementTable << ".taxonomy_id = " << microbesTable << ".taxonomy_id where " << microbesTable << ".genus = \"" << microbe << "\" group by " << measurementTable << ".patient_id, " << measurementTable << ".timestamp)t inner join Patient on t.patient_id = Patient.patient_id where Patient.region = \"US\" order by Patient.p_condition, Patient.last_name, t.timestamp;";
	        //queryss << "select Patient.last_name, Patient.p_condition, Patient.patient_id, unix_timestamp(" << measurementTable << ".timestamp) as timestamp, " << microbesTable << ".genus, sum(" << measurementTable << ".value) as value from " << measurementTable << " inner join Patient on Patient.patient_id = " << measurementTable << ".patient_id inner join " << microbesTable << " on " << microbesTable << ".taxonomy_id = " << measurementTable << ".taxonomy_id where " << microbesTable << ".genus = \"" << microbe << "\" and Patient.region = \"US\" group by patient_id, timestamp order by p_condition, last_name, timestamp;";

            break;
        }

	case MGT_PHYLUM:
	{
	    queryss << "select Patient.p_condition, Patient.last_name, t.patient_id, unix_timestamp(t.timestamp) as timestamp, t.value from (select " << microbesTable << ".phylum, " << measurementTable << ".patient_id, " << measurementTable << ".timestamp, sum(" << measurementTable << ".value) as value from "<< measurementTable << " inner join " << microbesTable << " on " << measurementTable << ".taxonomy_id = " << microbesTable << ".taxonomy_id where " << microbesTable << ".phylum = \"" << microbe << "\" group by " << measurementTable << ".patient_id, " << measurementTable << ".timestamp)t inner join Patient on t.patient_id = Patient.patient_id where Patient.region = \"US\" order by Patient.p_condition, Patient.last_name, t.timestamp;";
	    break;
	}

        default:
        {
	        queryss << "select Patient.last_name, Patient.p_condition, Patient.patient_id, unix_timestamp(" << measurementTable << ".timestamp) as timestamp, " << measurementTable << ".value from " << measurementTable << " inner join Patient on Patient.patient_id = " << measurementTable << ".patient_id where " << measurementTable << ".taxonomy_id = " << taxid << " and Patient.region = \"US\" order by p_condition, last_name, timestamp;";

            break;
        }
    }

    //std::cerr << "Query: " << queryss.str() << std::endl;

    struct microbeData * data = NULL;
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
		data = new struct microbeData[dataSize];
		for(int i = 0; i < dataSize; ++i)
		{
		    data[i].name[511] = '\0';
		    strncpy(data[i].name,result(i,"last_name").c_str(),511);
		    data[i].condition[511] = '\0';
		    strncpy(data[i].condition,result(i,"p_condition").c_str(),511);
		    data[i].id = atoi(result(i,"patient_id").c_str());
		    data[i].timestamp = atol(result(i,"timestamp").c_str());
		    data[i].value = atof(result(i,"value").c_str());
		}
	    }
	}

	/*if(_conn)
	{
	    mysqlpp::Query query = _conn->query(queryss.str().c_str());
	    mysqlpp::StoreQueryResult res = query.store();

	    dataSize = res.num_rows();

	    if(dataSize)
	    {
		data = new struct microbeData[dataSize];
		for(int i = 0; i < dataSize; ++i)
		{
		    data[i].name[511] = '\0';
		    strncpy(data[i].name,res[i]["last_name"].c_str(),511);
		    data[i].condition[511] = '\0';
		    strncpy(data[i].condition,res[i]["p_condition"].c_str(),511);
		    data[i].id = atoi(res[i]["patient_id"].c_str());
		    data[i].timestamp = atol(res[i]["timestamp"].c_str());
		    data[i].value = atof(res[i]["value"].c_str());
		}
	    }
	}*/

	ComController::instance()->sendSlaves(&dataSize,sizeof(int));
	if(dataSize)
	{
	    ComController::instance()->sendSlaves(data,dataSize*sizeof(struct microbeData));
	}
    }
    else
    {
	ComController::instance()->readMaster(&dataSize,sizeof(int));
	if(dataSize)
	{
	    data = new struct microbeData[dataSize];
	    ComController::instance()->readMaster(data,dataSize*sizeof(struct microbeData));
	}
    }

    std::map<std::string, std::vector<std::pair<std::string, float> > > dataMap;
    std::vector<std::string> orderList;

    std::map<std::string,osg::Vec4> colorMap = GraphGlobals::getPatientColorMap();

    for(int i = 0; i < dataSize; ++i)
    {
	std::string condition = data[i].condition;
	if(condition == "CD" || condition == "crohn's disease" || condition == "healthy" || condition == "Larry" || condition == "UC" || condition == "ulcerous colitis")
	{
	    //if(data[i].value > 0.0)
	    {
		char timestamp[512];
		timestamp[511] = '\0';
		strftime(timestamp,511,"%F",localtime(&data[i].timestamp));

		std::string group;

		std::string name = data[i].name;
		name = name + " - " + timestamp;

		if(groupPatients || firstOnly)
		{
		    if(condition == "CD" || condition == "crohn's disease")
		    {
			if((data[i].id >= 238 && data[i].id <= 242) || (data[i].id >= 339 && data[i].id <= 343))
			{
			    if(_cdCountMap.find(data[i].name) == _cdCountMap.end())
			    {
				_cdCountMap[data[i].name] = true;
			    }

			    std::stringstream groupss;
			    groupss << "Crohns - " << _cdCountMap.size();
			    group = groupss.str();
			}
			else
			{
			    continue;
			}
		    }
		    else if(condition == "UC" || condition == "ulcerous colitis")
		    {
			if((data[i].id >= 243 && data[i].id <= 244) || (data[i].id >= 344 && data[i].id <= 349))
			{
			    if(_ucCountMap.find(data[i].name) == _ucCountMap.end())
			    {
				_ucCountMap[data[i].name] = true;
			    }

			    std::stringstream groupss;
			    groupss << "UC - " << _ucCountMap.size();
			    group = groupss.str();
			}
			else
			{
			    continue;
			}
		    }
		}
		else
		{
		    if(condition == "CD" || condition == "crohn's disease")
		    {
			group = "Crohns";
		    }
		    else if(condition == "UC" || condition == "ulcerous colitis")
		    {
			group = "UC";
		    }
		}

		if(condition == "healthy")
		{
		    group = "Healthy";
		}
		else if(condition == "Larry")
		{
		    //group = "Smarr";
		    group = "LS";
		}
		
		if(!groupPatients || group != "Healthy")
		{
		    if(!firstOnly || group == "Healthy" || !dataMap[group].size())
		    {
			dataMap[group].push_back(std::pair<std::string,float>(name,data[i].value));
		    }
		}
	    }
	}
    }

    //orderList.push_back("Smarr");
    orderList.push_back("LS");

    if(firstOnly || groupPatients)
    {
	for(int i = 0; i < _cdCountMap.size(); ++i)
	{
	    std::stringstream cdss;
	    cdss << "Crohns - " << (i+1);
	    colorMap[cdss.str()] = colorMap["Crohns"];
	    orderList.push_back(cdss.str());
	}
	for(int i = 0; i < _ucCountMap.size(); ++i)
	{
	    std::stringstream ucss;
	    ucss << "UC - " << (i+1);
	    colorMap[ucss.str()] = colorMap["UC"];
	    orderList.push_back(ucss.str());
	}
    }
    else
    {
	orderList.push_back("Crohns");
	orderList.push_back("UC");
    }


    if(!groupPatients)
    {
	orderList.push_back("Healthy");
    }

    _graph->setColorMapping(GraphGlobals::getDefaultPhylumColor(),colorMap);
    _graph->setColorMode(BGCM_GROUP);

    if(data)
    {
	delete[] data;
    }

    if(rankOrder)
    {
	for(std::map<std::string, std::vector<std::pair<std::string, float> > >::iterator it = dataMap.begin(); it != dataMap.end(); ++it)
	{
	    std::sort(it->second.begin(),it->second.end(),rankOrderSort);
	}
    }

    bool status = _graph->setGraph(microbe + titleSuffix,dataMap,orderList,BGAT_LOG,"Relative Abundance","","condition / patient", osg::Vec4());

    if(status)
    {
	addChild(_graph->getRootNode());
	_graph->addMathFunction(new BandingFunction());
	_graph->setShowLabels(labels);
	//_graph->setDisplayRange(0.000007,1.0);
    }

    return status;
}

void SingleMicrobeObject::objectAdded()
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

void SingleMicrobeObject::objectRemoved()
{
    GraphLayoutObject * layout = dynamic_cast<GraphLayoutObject*>(_parent);
    if(layout && layout->getPatientKeyObject())
    {
	layout->getPatientKeyObject()->unref(this);
    }
}

void SingleMicrobeObject::setGraphSize(float width, float height)
{
    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);

    _graph->setDisplaySize(width,height);
}

void SingleMicrobeObject::selectPatients(std::map<std::string,std::vector<std::string> > & selectMap)
{
    _graph->selectItems(selectMap);
}

bool SingleMicrobeObject::processEvent(cvr::InteractionEvent * ie)
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

void SingleMicrobeObject::updateCallback(int handID, const osg::Matrix & mat)
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

void SingleMicrobeObject::leaveCallback(int handID)
{
    if((_desktopMode && TrackingManager::instance()->getHandTrackerType(handID) == TrackerBase::MOUSE) ||
	(!_desktopMode && TrackingManager::instance()->getHandTrackerType(handID) == TrackerBase::POINTER))
    {
	_graph->clearHoverText();
    }
}

void SingleMicrobeObject::setLogScale(bool logScale)
{
    if(logScale)
    {
	_graph->setAxisType(BGAT_LOG);
    }
    else
    {
	_graph->setAxisType(BGAT_LINEAR);
    }
}

void SingleMicrobeObject::setShowStdDev(bool show)
{
    for(int i = 0; i < _graph->getNumMathFunctions(); ++i)
    {
	BandingFunction * bf = dynamic_cast<BandingFunction*>(_graph->getMathFunction(i));
	if(bf)
	{
	    bf->setShowStdDev(show);
	}
    }
}

void BandingFunction::added(osg::Geode * geode)
{
    if(!_bandGeometry)
    {
	_bandGeometry = new osg::Geometry();
	_bandGeometry->setUseDisplayList(false);
	_bandGeometry->setUseVertexBufferObjects(true);

	osg::StateSet * stateset = _bandGeometry->getOrCreateStateSet();
	stateset->setMode(GL_BLEND,osg::StateAttribute::ON);
	stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
	stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

	_lineGeometry = new osg::Geometry();
	_lineGeometry->setUseDisplayList(false);
	_lineGeometry->setUseVertexBufferObjects(true);

    // changed from 5.0
    _lineWidth = new osg::LineWidth(3.0);

	stateset = _lineGeometry->getOrCreateStateSet();
	stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
	stateset->setAttributeAndModes(_lineWidth,osg::StateAttribute::ON);

	_boundsCallback = new SetBoundsCallback;
	_bandGeometry->setComputeBoundingBoxCallback(_boundsCallback.get());
	_lineGeometry->setComputeBoundingBoxCallback(_boundsCallback.get());
    }

    geode->addDrawable(_bandGeometry);
    geode->addDrawable(_lineGeometry);
    _myGeode = geode;
}

void BandingFunction::removed(osg::Geode * geode)
{
    geode->removeDrawable(_bandGeometry);
    geode->removeDrawable(_lineGeometry);
    _myGeode = NULL;
}

void BandingFunction::update(float left, float right, float top, float bottom, float barWidth, std::map<std::string, std::vector<std::pair<std::string, float> > > & data, BarGraphDisplayMode displayMode, const std::vector<std::string> & groupOrder, const std::vector<std::pair<std::string,int> > & customOrder, float displayMin, float displayMax, BarGraphAxisType axisType, const std::vector<std::pair<float,float> > & groupRanges)
{
    _bandGeometry->removePrimitiveSet(0,_bandGeometry->getNumPrimitiveSets());
    _lineGeometry->removePrimitiveSet(0,_lineGeometry->getNumPrimitiveSets());

    if(displayMode == BGDM_GROUPED)
    {
	osg::ref_ptr<osg::Vec3Array> verts = new osg::Vec3Array();
	osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();

	osg::ref_ptr<osg::Vec3Array> lverts = new osg::Vec3Array();
	osg::ref_ptr<osg::Vec4Array> lcolors = new osg::Vec4Array();

	for(int i = 0; i < groupOrder.size(); ++i)
	{
	    float sum = 0.0;
	    for(int j = 0; j < data[groupOrder[i]].size(); ++j)
	    {
		sum += data[groupOrder[i]][j].second;
	    }
	    sum /= ((float)data[groupOrder[i]].size());

	    float dev = 0.0;
	    for(int j = 0; j < data[groupOrder[i]].size(); ++j)
	    {
		float temp = data[groupOrder[i]][j].second - sum;
		temp *= temp;
		dev += temp;
	    }
	    dev /= ((float)data[groupOrder[i]].size());
	    dev = sqrt(dev);

	    float bTop = 0, bBottom = 0, avg = 0;

	    switch(axisType)
	    {
		case BGAT_LINEAR:
		{
		    if(sum >= displayMin && sum <= displayMax)
		    {
			avg = bottom + ((sum - displayMin) / (displayMax - displayMin) * (top-bottom));
			bTop = sum + 1.0*dev;
			bTop = std::min(bTop,displayMax);
			bBottom = sum - 1.0*dev;
			bBottom = std::max(bBottom,displayMin);
			bTop = bottom + ((bTop - displayMin) / (displayMax - displayMin) * (top-bottom));
			bBottom = bottom + ((bBottom - displayMin) / (displayMax - displayMin) * (top-bottom));
		    }
		    break;
		}
		case BGAT_LOG:
		{
		    avg = sum;
		    bTop = sum + 1.0*dev;
		    bTop = std::min(bTop,displayMax);
		    bBottom = sum - 1.0*dev;
		    bBottom = std::max(bBottom,displayMin);
		    float logMin = log10(displayMin);
		    float logMax = log10(displayMax);
		    if(avg > 0.0)
		    {
			avg = log10(avg);
			avg = bottom + ((avg-logMin)/(logMax-logMin))*(top-bottom);
		    }
		    bTop = log10(bTop);
		    bTop = std::max(bTop,logMin);
		    bTop = std::min(bTop,logMax);
		    bBottom = log10(bBottom);
		    bBottom = std::max(bBottom,logMin);
		    bBottom = std::min(bBottom,logMax);
		    bBottom = bottom + ((bBottom-logMin)/(logMax-logMin))*(top-bottom);
		    bTop = bottom + ((bTop-logMin)/(logMax-logMin))*(top-bottom);
		    break;
		}
		default:
		    break;
	    }

	    verts->push_back(osg::Vec3(groupRanges[i].first,-1.75,bBottom));
	    verts->push_back(osg::Vec3(groupRanges[i].second,-1.75,bBottom));
	    verts->push_back(osg::Vec3(groupRanges[i].second,-1.75,bTop));
	    verts->push_back(osg::Vec3(groupRanges[i].first,-1.75,bTop));

	    osg::Vec4 color(1.0,0,0,0.2);
	    colors->push_back(color);
	    colors->push_back(color);
	    colors->push_back(color);
	    colors->push_back(color);

	    if(sum > 0.0  && sum >= displayMin && sum <= displayMax)
	    {
		lverts->push_back(osg::Vec3(groupRanges[i].first,-1.95,avg));
		lverts->push_back(osg::Vec3(groupRanges[i].second,-1.95,avg));

		osg::Vec4 lcolor(0,0,0,1);
		lcolors->push_back(lcolor);
		lcolors->push_back(lcolor);
	    }
	}

	_bandGeometry->setVertexArray(verts);
	_bandGeometry->setColorArray(colors);
	_bandGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

	if(verts->size())
	{
	    osg::ref_ptr<osg::DrawArrays> drawArrays = new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,0);
	    drawArrays->setCount(verts->size());
	    _bandGeometry->addPrimitiveSet(drawArrays);
	}

	_lineGeometry->setVertexArray(lverts);
	_lineGeometry->setColorArray(lcolors);
	_lineGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

	if(lverts->size())
	{
	    osg::ref_ptr<osg::DrawArrays> drawArrays = new osg::DrawArrays(osg::PrimitiveSet::LINES,0,lverts->size());
	    _lineGeometry->addPrimitiveSet(drawArrays);
	}
    }
    _boundsCallback->bbox.set(left,-1.95,bottom,right,-1.75,top);
    _bandGeometry->dirtyBound();
    _lineGeometry->dirtyBound();
    _bandGeometry->getBound();
    _lineGeometry->getBound();
}

void BandingFunction::setShowStdDev(bool show)
{
    if(_myGeode.valid())
    {
	_myGeode->removeDrawable(_bandGeometry);
    }
    if(show && _myGeode.valid())
    {
	_myGeode->addDrawable(_bandGeometry);
    }
}
