#include "MicrobeGraphObject.h"
#include "GraphLayoutObject.h"
#include "GraphGlobals.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/ComController.h>
#include <cvrKernel/PluginManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrInput/TrackingManager.h>
#include <cvrUtil/OsgMath.h>

#include <PluginMessageType.h>

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <ctime>

// for time debug
#include <sys/time.h>

using namespace cvr;

MicrobeGraphObject::MicrobeGraphObject(mysqlpp::Connection * conn, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : LayoutTypeObject(name,navigation,movable,clip,contextMenu,showBounds)
{
    _conn = conn;

    setBoundsCalcMode(SceneObject::MANUAL);
    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);

    _width = width;
    _height = height;

    _desktopMode = ConfigManager::getBool("Plugin.FuturePatient.DesktopMode",false);

    if(_myMenu)
    {
	_colorModeML = new MenuList();
	_colorModeML->setCallback(this);
	addMenuItem(_colorModeML);

	std::vector<std::string> listItems;
	listItems.push_back("Solid Color");
	listItems.push_back("Group Colors");
	_colorModeML->setValues(listItems);
	_colorModeML->setIndex(1);

	_microbeText = new MenuText("");
	_microbeText->setCallback(this);
	_searchButton = new MenuButton("Web Search");
	_searchButton->setCallback(this);
    }
    else
    {
	_microbeText = NULL;
	_searchButton = NULL;
	_colorModeML = NULL;
    }
    

    _graph = new GroupedBarGraph(width,height);
    _graph->setColorMapping(GraphGlobals::getDefaultPhylumColor(),GraphGlobals::getPhylumColorMap());
    _graph->setColorMode(_colorModeML ? (BarGraphColorMode)_colorModeML->getIndex() : BGCM_SOLID);
}

MicrobeGraphObject::~MicrobeGraphObject()
{
    if(_microbeText)
    {
	delete _microbeText;
    }

    if(_searchButton)
    {
	delete _searchButton;
    }
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

void MicrobeGraphObject::setBGColor(osg::Vec4 color)
{
    _graph->setBGColor(color);
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

void MicrobeGraphObject::dumpState(std::ostream & out)
{
    out << "MICROBE_GRAPH" << std::endl;
    out << _specialGraph << std::endl;

    out << _microbes << std::endl;
    out << _lsOrdered << std::endl;

    if(_specialGraph)
    {
	out << _specialType << std::endl;
    }
    else
    {
	out << _graphTitle << std::endl;
	out << _testLabel << std::endl;
	out << _patientid << std::endl;
    }

    out << _graph->getDisplayRangeMin() << std::endl;
    out << _graph->getDisplayRangeMax() << std::endl;
}

bool MicrobeGraphObject::loadState(std::istream & in)
{
    bool special, lsOrder;
    int microbes;
    in >> special >> microbes >> lsOrder;

    if(special)
    {
	int stype;
	in >> stype;
	setSpecialGraph((SpecialMicrobeGraphType)stype,microbes,lsOrder);
    }
    else
    {
	char tempstr[1024];
	// consume endl
	in.getline(tempstr,1024);

	std::string title, tlabel;
	in.getline(tempstr,1024);
	title = tempstr;
	in.getline(tempstr,1024);
	tlabel = tempstr;

	int patientid;
	in >> patientid;

	setGraph(title,patientid,tlabel,microbes,lsOrder);
    }

    float drmin, drmax;
    in >> drmin >> drmax;

    _graph->setDisplayRange(drmin,drmax);

    return true;
}

bool MicrobeGraphObject::processEvent(InteractionEvent * ie)
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

	layout->selectMicrobes(selectedGroup,selectedKeys);
	if(clickUsed)
	{
	    return true;
	}
    }

    TrackedButtonInteractionEvent * tie = ie->asTrackedButtonEvent();

    if(tie && _contextMenu && tie->getButton() == _menuButton)
    {
	if(tie->getInteraction() == BUTTON_DOWN)
	{
	    if(!_myMenu->isVisible() || !_graph->getHoverItem().empty())
	    {
		_myMenu->setVisible(true);
		osg::Vec3 start(0,0,0), end(0,1000,0);
		start = start * tie->getTransform();
		end = end * tie->getTransform();

		osg::Vec3 p1, p2;
		bool n1, n2;
		float dist = 0;

		if(intersects(start,end,p1,n1,p2,n2))
		{
		    float d1 = (p1 - start).length();
		    if(n1)
		    {
			d1 = -d1;
		    }

		    float d2 = (p2 - start).length();
		    if(n2)
		    {
			d2 = -d2;
		    }

		    if(n1)
		    {
			dist = d2;
		    }
		    else if(n2)
		    {
			dist = d1;
		    }
		    else
		    {
			if(d1 < d2)
			{
			    dist = d1;
			}
			else
			{
			    dist = d2;
			}
		    }
		}

		dist = std::min(dist,
			SceneManager::instance()->getDefaultContextMenuMaxDistance());
		dist = std::max(dist,
			SceneManager::instance()->getDefaultContextMenuMinDistance());

		osg::Vec3 menuPoint(0,dist,0);
		menuPoint = menuPoint * tie->getTransform();

		osg::Vec3 viewerPoint =
		    TrackingManager::instance()->getHeadMat(0).getTrans();
		osg::Vec3 viewerDir = viewerPoint - menuPoint;
		viewerDir.z() = 0.0;

		osg::Matrix menuRot;

		// point towards viewer if not on tiled wall
		if(!ie->asPointerEvent())
		{
		    menuRot.makeRotate(osg::Vec3(0,-1,0),viewerDir);
		}

		osg::Matrix m;
		m.makeTranslate(menuPoint);
		_myMenu->setTransform(menuRot * m);

		_myMenu->setScale(SceneManager::instance()->getDefaultContextMenuScale());

		SceneManager::instance()->setMenuOpenObject(this);
	    }
            else
	    {
		SceneManager::instance()->closeOpenObjectMenu();
		return true;
	    }

	    if(!_graph->getHoverItem().empty())
	    {
		_menuMicrobe = _graph->getHoverItem();
		_microbeText->setText(std::string("Microbe: ") + _menuMicrobe);
		_myMenu->addMenuItem(_microbeText);
		_myMenu->addMenuItem(_searchButton);
	    }
	    else
	    {
		_myMenu->removeMenuItem(_microbeText);
		_myMenu->removeMenuItem(_searchButton);
		_menuMicrobe = "";
	    }

	    return true;
	}
    }

    return TiledWallSceneObject::processEvent(ie);
}

void MicrobeGraphObject::updateCallback(int handID, const osg::Matrix & mat)
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

void MicrobeGraphObject::leaveCallback(int handID)
{
    if((_desktopMode && TrackingManager::instance()->getHandTrackerType(handID) == TrackerBase::MOUSE) ||
	(!_desktopMode && TrackingManager::instance()->getHandTrackerType(handID) == TrackerBase::POINTER))
    {
	_graph->clearHoverText();
    }
}

void MicrobeGraphObject::menuCallback(MenuItem * item)
{
    if(item == _searchButton)
    {
	std::cerr << "Do search for microbe: " << _menuMicrobe << std::endl;
	
	if(PluginManager::instance()->getPluginLoaded("OsgVnc"))
	{
	    struct OsgVncRequest gqr;
	    gqr.query = _menuMicrobe;

	    PluginHelper::sendMessageByName("OsgVnc",VNC_GOOGLE_QUERY,(char*)&gqr);
	}
	else
	{
	    std::cerr << "OsgVnc plugin not loaded." << std::endl;
	}
    }

    if(item == _colorModeML)
    {
	_graph->setColorMode((BarGraphColorMode)_colorModeML->getIndex());
    }

    TiledWallSceneObject::menuCallback(item);
}

bool MicrobeGraphObject::setGraph(std::string title, int patientid, std::string testLabel, time_t testTime, int microbes, bool group, bool lsOrdering)
{
    struct tm timetm = *localtime(&testTime);
    char timestr[256];
    timestr[255] = '\0';
    strftime(timestr, 255, "%F", &timetm);

    _graphTitle = title + " - " + timestr;
    std::stringstream valuess, orderss;

    //valuess << "select * from (select Microbes.description, Microbes.phylum, Microbes.species, Microbe_Measurement.value from  Microbe_Measurement inner join Microbes on Microbe_Measurement.taxonomy_id = Microbes.taxonomy_id where Microbe_Measurement.patient_id = \"" << patientid << "\" and Microbe_Measurement.timestamp = \""<< testLabel << "\" order by value desc limit " << microbes << ")t order by t.phylum, t.value desc;";
    valuess << "select description, phylum, species, value from (select taxonomy_id, value from Microbe_Measurement where Microbe_Measurement.patient_id = " << patientid << " and Microbe_Measurement.timestamp = \"" << testLabel << "\" order by value desc limit " << microbes << ")v inner join Microbes on v.taxonomy_id = Microbes.taxonomy_id order by phylum, value desc;";

    orderss << "select t.phylum, sum(t.value) as total_value from (select Microbes.phylum, Microbe_Measurement.value from  Microbe_Measurement inner join Microbes on Microbe_Measurement.taxonomy_id = Microbes.taxonomy_id where Microbe_Measurement.patient_id = \"" << patientid << "\" and Microbe_Measurement.timestamp = \"" << testLabel << "\" order by value desc limit " << microbes << ")t group by phylum order by total_value desc;";

    _specialGraph = false;
    _patientid = patientid;
    _testLabel = testLabel;
    _microbes = microbes;
    _lsOrdered = lsOrdering;

    return loadGraphData(valuess.str(), orderss.str(), group, lsOrdering);
}

bool MicrobeGraphObject::setSpecialGraph(SpecialMicrobeGraphType smgt, int microbes, bool group, bool lsOrdering)
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

    _specialGraph = true;
    _specialType = smgt;
    _microbes = microbes;
    _lsOrdered = lsOrdering;

    return loadGraphData(valuess.str(), orderss.str(), group, lsOrdering);
}

void MicrobeGraphObject::objectAdded()
{
    GraphLayoutObject * layout = dynamic_cast<GraphLayoutObject*>(_parent);
    if(layout && layout->getPhylumKeyObject())
    {
	bool addKey = !layout->getPhylumKeyObject()->hasRef();
	layout->getPhylumKeyObject()->ref(this);

	if(addKey)
	{
	    layout->addLineObject(layout->getPhylumKeyObject());
	}
    }
}

void MicrobeGraphObject::objectRemoved()
{
    GraphLayoutObject * layout = dynamic_cast<GraphLayoutObject*>(_parent);
    if(layout && layout->getPhylumKeyObject())
    {
	layout->getPhylumKeyObject()->unref(this);
    }
}

bool MicrobeGraphObject::loadGraphData(std::string valueQuery, std::string orderQuery, bool group, bool lsOrdering)
{
    //std::cerr << "Query1: " << valueQuery << std::endl << "Query2: " << orderQuery << std::endl;
    struct timeval start,end;
    gettimeofday(&start,NULL);

    _graphData.clear();
    _graphOrder.clear();

    struct MicrobeDataHeader
    {
	int numDataValues;
	int numOrderValues;
	float totalValue;
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
    header.totalValue = 0;
    header.valid = false;

    MicrobeDataValue * data = NULL;
    MicrobeOrderValue * order = NULL;

    if(ComController::instance()->isMaster())
    {
	if(_conn)
	{
	    struct timeval start,end;
	    gettimeofday(&start,NULL);
	    mysqlpp::Query valueq = _conn->query(valueQuery.c_str());
	    mysqlpp::StoreQueryResult valuer = valueq.store();
	    gettimeofday(&end,NULL);
	    //std::cerr << "Query1: " << (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec) / 1000000.0) << std::endl;

	    header.numDataValues = valuer.num_rows();

	    std::map<std::string,float> totalMap;

	    if(valuer.num_rows())
	    {
		data = new MicrobeDataValue[valuer.num_rows()];

		for(int i = 0; i < valuer.num_rows(); ++i)
		{
		    strncpy(data[i].phylum,valuer[i]["phylum"].c_str(),1023);
		    strncpy(data[i].species,valuer[i]["species"].c_str(),1023);
		    strncpy(data[i].description,valuer[i]["description"].c_str(),1023);
		    data[i].value = atof(valuer[i]["value"]);
		    totalMap[data[i].phylum] += data[i].value;
		    header.totalValue += data[i].value;
		}
	    }

	    std::vector<std::pair<float,std::string> > orderList;
	    for(std::map<std::string,float>::iterator it = totalMap.begin(); it != totalMap.end(); ++it)
	    {
		orderList.push_back(std::pair<float,std::string>(it->second,it->first));
	    }

	    header.numOrderValues = orderList.size();

	    std::sort(orderList.begin(),orderList.end());
	    if(orderList.size())
	    {
		order = new MicrobeOrderValue[orderList.size()];

		//default sort is backwards, reverse add order
		for(int i = 0; i < orderList.size(); ++i)
		{
		    strncpy(order[i].group,orderList[orderList.size()-i-1].second.c_str(),1023);
		}
	    }

	    /*struct timeval start1, end1;
	    gettimeofday(&start1,NULL);

	    mysqlpp::Query orderq = _conn->query(orderQuery.c_str());
	    mysqlpp::StoreQueryResult orderr = orderq.store();
	    gettimeofday(&end1,NULL);
	    //std::cerr << "Query2: " << (end1.tv_sec - start1.tv_sec) + ((end1.tv_usec - start1.tv_usec) / 1000000.0) << std::endl;

	    header.numOrderValues = orderr.num_rows();

	    if(orderr.num_rows())
	    {
		order = new MicrobeOrderValue[orderr.num_rows()];

		for(int i = 0; i < orderr.num_rows(); ++i)
		{
		    strncpy(order[i].group,orderr[i]["phylum"].c_str(),1023);
		}
	    }*/

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

    if(lsOrdering)
    {
	std::vector<std::string> reorderVec;
	reorderVec.push_back("Spirochaetes");
	reorderVec.push_back("Tenericutes");
	reorderVec.push_back("Cyanobacteria");
	reorderVec.push_back("Planctomycetes");
	reorderVec.push_back("Synergistetes");
	reorderVec.push_back("Ascomycota");
	reorderVec.push_back("Euryarchaeota");
	reorderVec.push_back("Fusobacteria");
	reorderVec.push_back("Actinobacteria");
	reorderVec.push_back("Proteobacteria");
	reorderVec.push_back("Verrucomicrobia");
	reorderVec.push_back("Firmicutes");
	reorderVec.push_back("Bacteroidetes");

	for(int i = 0; i < reorderVec.size(); ++i)
	{
	    for(std::vector<std::string>::iterator it = _graphOrder.begin(); it != _graphOrder.end(); ++it)
	    {
		if(*it == reorderVec[i])
		{
		    _graphOrder.erase(it);
		    _graphOrder.insert(_graphOrder.begin(),reorderVec[i]);
		    break;
		}
	    }
	}
    }
    
    std::stringstream titless;
    titless << _graphTitle << " - Count=" << header.numDataValues << " - " << round(header.totalValue * 100.0) << "%";

    bool graphValid = _graph->setGraph(titless.str(), _graphData, _graphOrder, BGAT_LOG, "Value", "", "phylum / species",osg::Vec4(1.0,0,0,1));

    if(graphValid)
    {
	addChild(_graph->getRootNode());

	if(!group)
	{
	    std::vector<std::pair<std::string,int> > customOrder;

	    int totalEntries = 0;
	    for(std::map<std::string, std::vector<std::pair<std::string, float> > >::iterator it = _graphData.begin(); it != _graphData.end(); ++it)
	    {
		totalEntries += it->second.size();
	    }

	    std::map<std::string,int> groupIndexMap;

	    while(customOrder.size() < totalEntries)
	    {
		float maxVal = FLT_MIN;
		std::string group;
		for(std::map<std::string, std::vector<std::pair<std::string, float> > >::iterator it = _graphData.begin(); it != _graphData.end(); ++it)
		{
		    if(groupIndexMap[it->first] >= it->second.size())
		    {
			continue;
		    }

		    if(it->second[groupIndexMap[it->first]].second > maxVal)
		    {
			group = it->first;
			maxVal = it->second[groupIndexMap[it->first]].second;
		    }
		}

		customOrder.push_back(std::pair<std::string,int>(group,groupIndexMap[group]));
		groupIndexMap[group]++;
	    }

	    _graph->setCustomOrder(customOrder);
	    _graph->setDisplayMode(BGDM_CUSTOM);
	}
    }

    if(data)
    {
	delete[] data;
    }
    if(order)
    {
	delete[] order;
    }

    gettimeofday(&end,NULL);

    std::cerr << "GraphLoadTime: " << (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec) / 1000000.0) << std::endl;

    return graphValid;
}
