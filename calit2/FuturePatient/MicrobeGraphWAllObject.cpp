#include "MicrobeGraphWAllObject.h"
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
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>

// for time debug
#include <sys/time.h>

bool MicrobeGraphWAllObject::_mapInit = false;
std::map<int,std::pair<std::string,std::string> > MicrobeGraphWAllObject::_patientMap;
DBManager * MicrobeGraphWAllObject::_dbm;

using namespace cvr;

MicrobeGraphWAllObject::MicrobeGraphWAllObject(DBManager * dbm, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : LayoutTypeObject(name,navigation,movable,clip,contextMenu,showBounds), SelectableObject()
{
    _dbm = dbm;

    setBoundsCalcMode(SceneObject::MANUAL);
    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);

    _width = width;
    _height = height;

    _desktopMode = ConfigManager::getBool("Plugin.FuturePatient.DesktopMode",false);

    makeSelect();
    updateSelect();

    if(_myMenu)
    {
	_selectCB = new MenuCheckbox("Selected",false);
	_selectCB->setCallback(this);
	addMenuItem(_selectCB);

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
    

    _graph = new GroupedBarGraphWPoints(width,height);
    _graph->setColorMapping(GraphGlobals::getDefaultPhylumColor(),GraphGlobals::getPhylumColorMap());
    _graph->setColorMode(_colorModeML ? (BarGraphColorMode)_colorModeML->getIndex() : BGCM_SOLID);
    _graph->setPointColorMapping(GraphGlobals::getPatientColorMap());
}

MicrobeGraphWAllObject::~MicrobeGraphWAllObject()
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

void MicrobeGraphWAllObject::setGraphSize(float width, float height)
{
    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);

    updateSelect();

    _graph->setDisplaySize(width,height);
}

void MicrobeGraphWAllObject::setColor(osg::Vec4 color)
{
    _graph->setColor(color);
}

void MicrobeGraphWAllObject::setBGColor(osg::Vec4 color)
{
    _graph->setBGColor(color);
}

float MicrobeGraphWAllObject::getGraphMaxValue()
{
    return _graph->getDataMax();
}

float MicrobeGraphWAllObject::getGraphMinValue()
{
    return _graph->getDataMin();
}

float MicrobeGraphWAllObject::getGraphDisplayRangeMax()
{
    return _graph->getDisplayRangeMax();
}

float MicrobeGraphWAllObject::getGraphDisplayRangeMin()
{
    return _graph->getDisplayRangeMin();
}

void MicrobeGraphWAllObject::setGraphDisplayRange(float min, float max)
{
    _graph->setDisplayRange(min,max);
}

void MicrobeGraphWAllObject::resetGraphDisplayRange()
{
    _graph->setDisplayRange(_graph->getDisplayRangeMin(),_graph->getDisplayRangeMax());
}

void MicrobeGraphWAllObject::selectPatients(std::map<std::string,std::vector<std::string> > & selectMap)
{
    _graph->selectOther(selectMap);
}

void MicrobeGraphWAllObject::selectMicrobes(std::string & group, std::vector<std::string> & keys)
{
    _graph->selectItems(group,keys);
}

float MicrobeGraphWAllObject::getGroupValue(std::string group, int i)
{
    return _graph->getGroupValue(group);
}

float MicrobeGraphWAllObject::getMicrobeValue(std::string group, std::string key, int i)
{
    return _graph->getKeyValue(group,key);
}

int MicrobeGraphWAllObject::getNumDisplayValues()
{
    return 1;
}

std::string MicrobeGraphWAllObject::getDisplayLabel(int i)
{
    return getTitle();
}

void MicrobeGraphWAllObject::dumpState(std::ostream & out)
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

bool MicrobeGraphWAllObject::loadState(std::istream & in)
{
    bool special, lsOrder;
    int microbes;
    in >> special >> microbes >> lsOrder;

    if(special)
    {
	int stype;
	in >> stype;
	//setSpecialGraph((SpecialMicrobeGraphType)stype,microbes,"",lsOrder);
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

	setGraph(title,patientid,tlabel,0,microbes,"","",lsOrder);
    }

    float drmin, drmax;
    in >> drmin >> drmax;

    _graph->setDisplayRange(drmin,drmax);

    return true;
}

std::string MicrobeGraphWAllObject::getTitle()
{
    return _graphTitle;
}

bool MicrobeGraphWAllObject::processEvent(InteractionEvent * ie)
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

	if(selectedGroup.empty() && !selectedKeys.size())
	{
	    std::string pg;
	    std::vector<std::string> pkeys;
	    layout->selectPatients(pg,pkeys);
	}

	layout->selectPatients(selectedGroup,selectedKeys);
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

    return FPTiledWallSceneObject::processEvent(ie);
}

void MicrobeGraphWAllObject::updateCallback(int handID, const osg::Matrix & mat)
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

void MicrobeGraphWAllObject::leaveCallback(int handID)
{
    if((_desktopMode && TrackingManager::instance()->getHandTrackerType(handID) == TrackerBase::MOUSE) ||
	(!_desktopMode && TrackingManager::instance()->getHandTrackerType(handID) == TrackerBase::POINTER))
    {
	_graph->clearHoverText();
    }
}

void MicrobeGraphWAllObject::menuCallback(MenuItem * item)
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

bool MicrobeGraphWAllObject::setGraph(std::string title, int patientid, std::string testLabel, time_t testTime, int microbes, std::string microbeTableSuffix, std::string measureTableSuffix, bool group, bool lsOrdering, MicrobeGraphType type)
{
    struct tm timetm = *localtime(&testTime);
    char timestr[256];
    timestr[255] = '\0';
    strftime(timestr, 255, "%F", &timetm);

    _graphTitle = title + " - " + timestr;
    std::stringstream valuess, orderss, allss;

    std::string measurementTable = "Microbe_Measurement";
    measurementTable += measureTableSuffix;

    std::string microbesTable = "Microbes";
    microbesTable += microbeTableSuffix;

    switch( type )
    {
	default:
        case MGT_SPECIES:
        {
	        valuess << "select description, phylum, species as name, v.taxonomy_id, value from (select taxonomy_id, value from " << measurementTable << " where " << measurementTable << ".patient_id = " << patientid << " and " << measurementTable << ".timestamp = \"" << testLabel << "\" order by value desc limit " << microbes << ")v inner join " << microbesTable << " on v.taxonomy_id = " << microbesTable << ".taxonomy_id order by phylum, value desc;";

		allss << "select patient_id, phylum, species as name, unix_timestamp(timestamp) as timestamp, " << measurementTable << ".value from (select taxonomy_id, value from " << measurementTable << " where " << measurementTable << ".patient_id = " << patientid << " and " << measurementTable << ".timestamp = \"" << testLabel << "\" order by value desc limit " << microbes << ")v inner join " << measurementTable << " on v.taxonomy_id = " << measurementTable << ".taxonomy_id inner join " << microbesTable << " on v.taxonomy_id = " << microbesTable << ".taxonomy_id where " << measurementTable << ".seq_type = 'fast' order by patient_id, value desc;";
            break;
        }

        case MGT_FAMILY:
        {
	        valuess << "select description, phylum, family as name, sum(value) as value from (select taxonomy_id, value from " << measurementTable << " where " << measurementTable << ".patient_id = " << patientid << " and " << measurementTable << ".timestamp = \"" << testLabel << "\" order by value desc limit " << microbes << ")v inner join " << microbesTable << " on v.taxonomy_id = " << microbesTable << ".taxonomy_id group by family order by phylum, value desc;";
		allss << "select patient_id, phylum, family as name, unix_timestamp(timestamp) as timestamp, sum(" << measurementTable << ".value) as value from (select taxonomy_id, value from " << measurementTable << " where " << measurementTable << ".patient_id = " << patientid << " and " << measurementTable << ".timestamp = \"" << testLabel << "\" order by value desc limit " << microbes << ")v inner join " << measurementTable << " on v.taxonomy_id = " << measurementTable << ".taxonomy_id inner join " << microbesTable << " on v.taxonomy_id = " << microbesTable << ".taxonomy_id where " << measurementTable << ".seq_type = 'fast' group by patient_id, timestamp, family order by patient_id, value desc;";
            break;
        }

        case MGT_GENUS:
        {
	    valuess << "select description, phylum, genus as name, sum(value) as value from (select taxonomy_id, value from " << measurementTable << " where " << measurementTable << ".patient_id = " << patientid << " and " << measurementTable << ".timestamp = \"" << testLabel << "\" order by value desc limit " << microbes << ")v inner join " << microbesTable << " on v.taxonomy_id = " << microbesTable << ".taxonomy_id group by genus order by phylum, value desc;";
	    allss << "select patient_id, phylum, genus as name, unix_timestamp(timestamp) as timestamp, sum(" << measurementTable << ".value) as value from (select taxonomy_id, value from " << measurementTable << " where " << measurementTable << ".patient_id = " << patientid << " and " << measurementTable << ".timestamp = \"" << testLabel << "\" order by value desc limit " << microbes << ")v inner join " << measurementTable << " on v.taxonomy_id = " << measurementTable << ".taxonomy_id inner join " << microbesTable << " on v.taxonomy_id = " << microbesTable << ".taxonomy_id where " << measurementTable << ".seq_type = 'fast' group by patient_id, timestamp, genus order by patient_id, value desc;";

            //valuess << "select description, phylum, genus, sum(value) as value from (select taxonomy_id, value from " << measurementTable << " where " << measurementTable << ".patient_id = " << patientid << " and " << measurementTable << ".timestamp = \"" << testLabel << "\" order by value desc limit " << microbes << ")v inner join " << microbesTable << " on v.taxonomy_id = " << microbesTable << ".taxonomy_id group by genus order by phylum, value desc;";
            break;
        }
    }

    _specialGraph = false;
    _patientid = patientid;
    _testLabel = testLabel;
    _microbes = microbes;
    _lsOrdered = lsOrdering;

    return loadGraphData(valuess.str(), allss.str(), group, lsOrdering, type);
}

bool MicrobeGraphWAllObject::setSpecialGraph(SpecialMicrobeGraphType smgt, int microbes, std::string region, std::string microbeSuffix, std::string measureSuffix, bool group, bool lsOrdering, MicrobeGraphType type)
{
    std::stringstream valuess, orderss, allss;

    std::string measurementTable = "Microbe_Measurement";
    measurementTable += measureSuffix;

    std::string microbesTable = "Microbes";
    microbesTable += microbeSuffix;

    switch(smgt)
    {
	case SMGT_AVERAGE:
	case SMGT_HEALTHY_AVERAGE:
	case SMGT_CROHNS_AVERAGE:
	case SMGT_SMARR_AVERAGE:
	    {

		std::string field;
		std::string condition;

		switch(smgt)
		{
		    case SMGT_AVERAGE:
			field = "average";
			condition = "ulcerous colitis";
			_graphTitle = "UC Average";
			break;
		    case SMGT_HEALTHY_AVERAGE:
			field = "average_healthy";
			condition = "healthy";
			_graphTitle = "Healthy Average";
			break;
		    case SMGT_CROHNS_AVERAGE:
			field = "average_crohns";
			condition = "crohn's disease";
			_graphTitle = "Crohns Average";
			break;
		    case SMGT_SMARR_AVERAGE:
			field = "";
			condition = "Larry";
			_graphTitle = "Smarr Average";
			break;
		    default:
			break;
		}

		_graphTitle = _graphTitle + " (" + region + ")";

		std::string regionField;

		if(region == "US")
		{
		    regionField = " and Microbe_Stats.patient_region = \"US\"";
		}
		else if(region == "EU")
		{
		    regionField = " and Microbe_Stats.patient_region = \"EU\"";
		}
		else
		{
		    regionField = "";
		}

        switch ( type )
		{
	    default:
            case MGT_SPECIES:
            {
		        valuess << "select Microbes.description, Microbes.phylum, Microbes.species as name, Microbes.taxonomy_id, avg(Microbe_Stats.value) as value from Microbes inner join Microbe_Stats on Microbes.taxonomy_id = Microbe_Stats.taxonomy_id and Microbe_Stats.stat_type = \"average\" and Microbe_Stats.patient_condition = \"" << condition << "\"" << regionField << " group by Microbes.taxonomy_id order by value desc limit " << microbes << ";";

			allss << "select patient_id, phylum, species as name, unix_timestamp(timestamp) as timestamp, " << measurementTable << ".value from (select Microbes.taxonomy_id, avg(Microbe_Stats.value) as value from Microbes inner join Microbe_Stats on Microbes.taxonomy_id = Microbe_Stats.taxonomy_id and Microbe_Stats.stat_type = \"average\" and Microbe_Stats.patient_condition = \"" << condition << "\"" << regionField << " group by Microbes.taxonomy_id order by value desc limit " << microbes << ")v inner join " << measurementTable << " on v.taxonomy_id = " << measurementTable << ".taxonomy_id inner join " << microbesTable << " on v.taxonomy_id = " << microbesTable << ".taxonomy_id where " << measurementTable << ".seq_type = 'fast' order by patient_id, value desc;";
		        break;
            }
            
            case MGT_FAMILY:
		    {
		        valuess << "select Microbes.description, Microbes.phylum, Microbes.family as name, sum(t.value) as value from (select taxonomy_id, avg(value) as value from Microbe_Stats where patient_condition = \"" << condition << "\"" << regionField << " group by taxonomy_id order by value desc limit " << microbes << ")t inner join Microbes on t.taxonomy_id = Microbes.taxonomy_id group by Microbes.family order by value desc;";

			allss << "select patient_id, phylum, family as name, unix_timestamp(timestamp) as timestamp, sum(" << measurementTable << ".value) as value from (select Microbes.taxonomy_id, avg(Microbe_Stats.value) as value from Microbes inner join Microbe_Stats on Microbes.taxonomy_id = Microbe_Stats.taxonomy_id and Microbe_Stats.stat_type = \"average\" and Microbe_Stats.patient_condition = \"" << condition << "\"" << regionField << " group by Microbes.taxonomy_id order by value desc limit " << microbes << ")v inner join " << measurementTable << " on v.taxonomy_id = " << measurementTable << ".taxonomy_id inner join " << microbesTable << " on v.taxonomy_id = " << microbesTable << ".taxonomy_id where " << measurementTable << ".seq_type = 'fast' group by patient_id, timestamp, family order by patient_id, value desc;";


		        break;
            }

            case MGT_GENUS:
            {
		        valuess << "select Microbes.description, Microbes.phylum, Microbes.genus as name, sum(t.value) as value from (select taxonomy_id, avg(value) as value from Microbe_Stats where patient_condition = \"" << condition << "\"" << regionField << " group by taxonomy_id order by value desc limit " << microbes << ")t inner join Microbes on t.taxonomy_id = Microbes.taxonomy_id group by Microbes.genus order by value desc;";

			allss << "select patient_id, phylum, genus as name, unix_timestamp(timestamp) as timestamp, sum(" << measurementTable << ".value) as value from (select Microbes.taxonomy_id, avg(Microbe_Stats.value) as value from Microbes inner join Microbe_Stats on Microbes.taxonomy_id = Microbe_Stats.taxonomy_id and Microbe_Stats.stat_type = \"average\" and Microbe_Stats.patient_condition = \"" << condition << "\"" << regionField << " group by Microbes.taxonomy_id order by value desc limit " << microbes << ")v inner join " << measurementTable << " on v.taxonomy_id = " << measurementTable << ".taxonomy_id inner join " << microbesTable << " on v.taxonomy_id = " << microbesTable << ".taxonomy_id where " << measurementTable << ".seq_type = 'fast' group by patient_id, timestamp, genus order by patient_id, value desc;";

                break;
            }
        }

		break;
	    }
	default:
	    return false;
    }

    _specialGraph = true;
    _specialType = smgt;
    _microbes = microbes;
    _lsOrdered = lsOrdering;

    return loadGraphData(valuess.str(), allss.str(), group, lsOrdering, type);
}

void MicrobeGraphWAllObject::objectAdded()
{
    GraphLayoutObject * layout = dynamic_cast<GraphLayoutObject*>(_parent);
    if(layout && layout->getPhylumKeyObject())
    {
	bool addKey = !layout->getPatientKeyObject()->hasRef();
	layout->getPatientKeyObject()->ref(this);

	if(addKey)
	{
	    layout->addLineObject(layout->getPatientKeyObject());
	}
    }
}

void MicrobeGraphWAllObject::objectRemoved()
{
    GraphLayoutObject * layout = dynamic_cast<GraphLayoutObject*>(_parent);
    if(layout && layout->getPatientKeyObject())
    {
	layout->getPatientKeyObject()->unref(this);
    }
}

bool MicrobeGraphWAllObject::loadGraphData(std::string valueQuery, std::string allQuery, bool group, bool lsOrdering, MicrobeGraphType type)
{
    //std::cerr << "Query1: " << valueQuery << std::endl << "Query2: " << orderQuery << std::endl;
    std::cerr << "Value Query: " << valueQuery << std::endl;
    std::cerr << "All Query: " << allQuery << std::endl;

    if(!_mapInit)
    {
	makePatientMap();
    }

    struct timeval start,end;
    gettimeofday(&start,NULL);

    _graphData.clear();
    _graphOrder.clear();

    struct MicrobeDataHeader
    {
	int numDataValues;
	int numOrderValues;
	int numOtherValues;
	float totalValue;
	bool valid;
    };

    struct MicrobeDataValue
    {
	char phylum[1024];
	char species[1024];
	char description[1024];
	int tax_id;
	float value;
    };

    struct MicrobeOrderValue
    {
	char group[1024];
    };

    struct MicrobeOtherValue
    {
	char phylum[1024];
	char name[1024];
	int patient_id;
	time_t timestamp;
	float value;
    };

    MicrobeDataHeader header;
    header.numDataValues = 0;
    header.numOrderValues = 0;
    header.numOtherValues = 0;
    header.totalValue = 0;
    header.valid = false;

    MicrobeDataValue * data = NULL;
    MicrobeOrderValue * order = NULL;
    MicrobeOtherValue * other = NULL;

    if(ComController::instance()->isMaster())
    {
	if(_dbm)
	{
	    struct timeval start,end;
	    gettimeofday(&start,NULL);

	    DBMQueryResult result;

	    _dbm->runQuery(valueQuery,result);

	    gettimeofday(&end,NULL);
	    //std::cerr << "Query1: " << (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec) / 1000000.0) << std::endl;

	    header.numDataValues = result.numRows();

	    std::map<std::string,float> totalMap;

	    if(result.numRows())
	    {
		data = new MicrobeDataValue[result.numRows()];

		for(int i = 0; i < result.numRows(); ++i)
		{
		    strncpy(data[i].phylum,result(i,"phylum").c_str(),1023);
		    strncpy(data[i].species,result(i,"name").c_str(),1023);
		    strncpy(data[i].description,result(i,"description").c_str(),1023);
		    data[i].value = atof(result(i,"value").c_str());
		    data[i].tax_id = atoi(result(i,"taxonomy_id").c_str());
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

	    DBMQueryResult result2;

	    _dbm->runQuery(allQuery,result2);

	    header.numOtherValues = result2.numRows();

	    if(result2.numRows())
	    {
		other = new struct MicrobeOtherValue[result2.numRows()];
		for(int i = 0; i < result2.numRows(); ++i)
		{
		    strncpy(other[i].phylum,result2(i,"phylum").c_str(),1023);
		    strncpy(other[i].name,result2(i,"name").c_str(),1023);
		    other[i].patient_id = atoi(result2(i,"patient_id").c_str());
		    other[i].timestamp = atol(result2(i,"timestamp").c_str());
		    other[i].value = atof(result2(i,"value").c_str());
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
	    if(header.numOtherValues)
	    {
		ComController::instance()->sendSlaves(other, header.numOtherValues*sizeof(struct MicrobeOtherValue));
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
	    if(header.numOtherValues)
	    {
		other = new struct MicrobeOtherValue[header.numOtherValues];

		ComController::instance()->readMaster(other, header.numOtherValues*sizeof(struct MicrobeOtherValue));
	    }
	}
    }

    if(!header.valid)
    {
	return false;
    }

    for(int i = 0; i < header.numDataValues; ++i)
    {
	//taxidMap[data[i].tax_id] = std::pair<std::string,std::string>(data[i].phylum,data[i].species);

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

    std::string sublabel;
    
    switch( type )
    {
        case MGT_SPECIES:
        {
	        sublabel = "phylum / microbe";
            break;
        }

        case MGT_FAMILY:
        {
	        sublabel = "phylum / family";
            break;
        }

        case MGT_GENUS:
        {
            sublabel = "phylum / genus";
            break;
        }

        default:
        {
            sublabel = "phylum / species";
            break;
        }
    }

    std::map<std::string,std::map<std::string,std::map<std::string,std::vector<std::pair<std::string,float> > > > > otherPointData;

    // PHILIP
    //std::ofstream testTab("test1000.sim");

    for(int i = 0; i < header.numOtherValues; ++i)
    {
	struct tm timetm = *localtime(&other[i].timestamp);
	char timestr[256];
	timestr[255] = '\0';
	strftime(timestr, 255, "%F", &timetm);

	std::string label = _patientMap[other[i].patient_id].first + " - " + timestr;
	//std::cerr << "Other: " << label << " phylum: " << other[i].phylum << " name: " << other[i].name << " value: " << other[i].value << std::endl;
		    
/*	
	// PHILIP tempory just to build a tab delimited file to produce graphs
	if( other[i].value > 0.0 )
	{
	    testTab << label; 
	    testTab << '\t';
	    testTab << other[i].name;
	    testTab << '\t';
	    testTab << other[i].value;
	    testTab <<'\n';
	}
*/	

	otherPointData[other[i].phylum][other[i].name][_patientMap[other[i].patient_id].second].push_back(std::pair<std::string,float>(label,other[i].value));
    }

    // PHILIP
    //testTab.close();

    bool graphValid = _graph->setGraph(titless.str(), _graphData, _graphOrder, otherPointData, BGAT_LOG, "Relative Abundance", "", sublabel,osg::Vec4(1.0,0,0,1));

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
		float maxVal = -FLT_MAX;
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
    if(other)
    {
	delete[] other;
    }

    gettimeofday(&end,NULL);

    std::cerr << "GraphLoadTime: " << (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec) / 1000000.0) << std::endl;

    return graphValid;
}

void MicrobeGraphWAllObject::makeSelect()
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

void MicrobeGraphWAllObject::updateSelect()
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
    _selectGeom->getBoundingBox();
}

void MicrobeGraphWAllObject::makePatientMap()
{
    std::cerr << "Creating patient map" << std::endl;
    std::string q = "select patient_id, last_name, p_condition from Patient;";

    struct patientEntry
    {
	int patient_id;
	char name[1024];
	char condition[1024];
    };

    int numEntries = 0;
    struct patientEntry * entries = NULL;

    if(ComController::instance()->isMaster())
    {
	if(_dbm)
	{
	    DBMQueryResult result;

	    _dbm->runQuery(q,result);

	    if(result.numRows())
	    {
		numEntries = result.numRows();
		entries = new struct patientEntry[numEntries];

		for(int i = 0; i < result.numRows(); ++i)
		{
		    //strncpy(data[i].phylum,result(i,"phylum").c_str(),1023);
		    entries[i].name[1023] = '\0';
		    entries[i].condition[1023] = '\0';
		    strncpy(entries[i].name,result(i,"last_name").c_str(),1023);
		    strncpy(entries[i].condition,result(i,"p_condition").c_str(),1023);
		    entries[i].patient_id = atoi(result(i,"patient_id").c_str());
		}
	    }

	    ComController::instance()->sendSlaves(&numEntries,sizeof(int));
	    if(numEntries)
	    {
		ComController::instance()->sendSlaves(entries,numEntries*sizeof(struct patientEntry));
	    }
	}
    }
    else
    {
	ComController::instance()->readMaster(&numEntries,sizeof(int));
	if(numEntries)
	{
	    entries = new struct patientEntry[numEntries];
	    ComController::instance()->readMaster(entries,numEntries*sizeof(struct patientEntry));
	}
    }

    for(int i = 0; i < numEntries; ++i)
    {
	std::string condition = entries[i].condition;
	std::string group;
	if(condition == "Larry")
	{
	    group = "LS";
	}
	else if(condition == "healthy")
	{
	    group = "Healthy";
	}
	else if(condition == "ulcerous colitis" || condition == "UC")
	{
	    group = "UC";
	}
	else if(condition == "crohn's disease" || condition == "CD")
	{
	    group = "Crohns";
	}

	if(group.empty())
	{
	    continue;
	}

	_patientMap[entries[i].patient_id] = std::pair<std::string,std::string>(entries[i].name,group);
    }

    std::cerr << "Map size: " << _patientMap.size() << std::endl;

    if(entries)
    {
	delete[] entries;
    }

    _mapInit = true;
}
