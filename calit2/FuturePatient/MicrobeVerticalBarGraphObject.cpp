#include "MicrobeVerticalBarGraphObject.h"
#include "GraphLayoutObject.h"
#include "GraphGlobals.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/ComController.h>
#include <cvrInput/TrackingManager.h>
#include <cvrUtil/OsgMath.h>

#include <iostream>
#include <sstream>
#include <string>

using namespace cvr;

MicrobeVerticalBarGraphObject::Microbe * MicrobeVerticalBarGraphObject::_microbeList = NULL;
int MicrobeVerticalBarGraphObject::_microbeCount = 0;

MicrobeVerticalBarGraphObject::MicrobeVerticalBarGraphObject(DBManager * dbm, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : LayoutTypeObject(name,navigation,movable,clip,contextMenu,showBounds), SelectableObject()
{
    _dbm = dbm;
    _graph = new VerticalStackedBarGraph("Microbe Graph");
    _graph->setColorMapping(GraphGlobals::getPhylumColorMap(),GraphGlobals::getDefaultPhylumColor());

    _graph->setDisplaySize(width,height);
    _width = width;
    _height = height;

    addChild(_graph->getRoot());

    if(contextMenu)
    {
	_selectCB = new MenuCheckbox("Selected",false);
	_selectCB->setCallback(this);
	addMenuItem(_selectCB);
    }

    _desktopMode = ConfigManager::getBool("Plugin.FuturePatient.DesktopMode",false);

    setBoundsCalcMode(SceneObject::MANUAL);
    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);
}

MicrobeVerticalBarGraphObject::~MicrobeVerticalBarGraphObject()
{
}

bool MicrobeVerticalBarGraphObject::addGraph(std::string label, int patientid, std::string testLabel, time_t testTime, std::string seqType,std::string microbeTableSuffix, std::string measureTableSuffix, MicrobeGraphType type)
{
    struct tm timetm = *localtime(&testTime);
    char timestr[256];
    timestr[255] = '\0';
    strftime(timestr, 255, "%F", &timetm);

    std::string title = label + " - " + timestr;
    std::stringstream valuess;

    std::string measurementTable = "Microbe_Measurement";
    measurementTable += measureTableSuffix;

    std::string microbesTable = "Microbes";
    microbesTable += microbeTableSuffix;

    int microbes = 1;

    switch( type )
    {
	default:
	case MGT_SPECIES:
	    {
		valuess << "select species as label, phylum, value from (select taxonomy_id, value from " << measurementTable << " where " << measurementTable << ".patient_id = " << patientid << " and " << measurementTable << ".seq_type = \"" << seqType << "\" and " << measurementTable << ".timestamp = \"" << testLabel << "\")v inner join " << microbesTable << " on v.taxonomy_id = " << microbesTable << ".taxonomy_id order by species;";

		break;
	    }

	case MGT_FAMILY:
	    {
		//valuess << "select description, phylum, family, sum(value) as value from (select taxonomy_id, value from " << measurementTable << " where " << measurementTable << ".patient_id = " << patientid << " and " << measurementTable << ".timestamp = \"" << testLabel << "\" order by value desc limit " << microbes << ")v inner join " << microbesTable << " on v.taxonomy_id = " << microbesTable << ".taxonomy_id group by family order by phylum, value desc;";
		valuess << "select family as label, phylum, sum(value) as value from (select taxonomy_id, value from " << measurementTable << " where " << measurementTable << ".patient_id = " << patientid << " and " << measurementTable << ".seq_type = \"" << seqType << "\" and " << measurementTable << ".timestamp = \"" << testLabel << "\")v inner join " << microbesTable << " on v.taxonomy_id = " << microbesTable << ".taxonomy_id group by family order by family;";
		break;
	    }

	case MGT_GENUS:
	    {
		//valuess << "select description, phylum, genus, sum(value) as value from (select taxonomy_id, value from " << measurementTable << " where " << measurementTable << ".patient_id = " << patientid << " and " << measurementTable << ".timestamp = \"" << testLabel << "\" order by value desc limit " << microbes << ")v inner join " << microbesTable << " on v.taxonomy_id = " << microbesTable << ".taxonomy_id group by genus order by phylum, value desc;";
		valuess << "select genus as label, phylum, sum(value) as value from (select taxonomy_id, value from " << measurementTable << " where " << measurementTable << ".patient_id = " << patientid << " and " << measurementTable << ".seq_type = \"" << seqType << "\" and " << measurementTable << ".timestamp = \"" << testLabel << "\")v inner join " << microbesTable << " on v.taxonomy_id = " << microbesTable << ".taxonomy_id group by genus order by genus;";
		break;
	    }
    }


    /*std::stringstream qss;
    qss << "select Microbes.species, Microbe_Measurement.value from Microbe_Measurement inner join Microbes on Microbe_Measurement.taxonomy_id = Microbes.taxonomy_id where Microbe_Measurement.patient_id = \"" << patientid << "\" and Microbe_Measurement.timestamp = \""<< testLabel << "\";";

    std::string title = label + "\n" + testLabel;

    struct LoadData ld;
    ld.special = false;
    ld.label = label;
    ld.patientid = patientid;
    ld.testLabel = testLabel;

    _loadedGraphs.push_back(ld);*/

    return addGraph(title,valuess.str());
}

bool MicrobeVerticalBarGraphObject::addSpecialGraph(SpecialMicrobeGraphType smgt, std::string seqType, std::string microbeTableSuffix, std::string measureTableSuffix, std::string region, MicrobeGraphType type)
{
    std::stringstream valuess;

    std::string title;

    switch(smgt)
    {
	case SMGT_AVERAGE:
	case SMGT_HEALTHY_AVERAGE:
	case SMGT_CROHNS_AVERAGE:
	    {

		std::string field;
		std::string condition;

		switch(smgt)
		{
		    case SMGT_AVERAGE:
			field = "average";
			condition = "ulcerous colitis";
			title = "UC Average";
			break;
		    case SMGT_HEALTHY_AVERAGE:
			field = "average_healthy";
			condition = "healthy";
			title = "Healthy Average";
			break;
		    case SMGT_CROHNS_AVERAGE:
			field = "average_crohns";
			condition = "crohn's disease";
			title = "Crohns Average";
			break;
		    default:
			break;
		}

		title = title + " (" + region + ")";

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
			    valuess << "select Microbes.phylum as phylum, Microbes.species as label, avg(Microbe_Stats.value) as value from Microbes inner join Microbe_Stats on Microbes.taxonomy_id = Microbe_Stats.taxonomy_id and Microbe_Stats.stat_type = \"average\"  and Microbes.seq_type = \"" << seqType << "\" and  Microbe_Stats.patient_condition = \"" << condition << "\"" << regionField << " group by Microbes.species order by species;";
			    break;
			}

		    case MGT_FAMILY:
			{
			    valuess << "select Microbes.phylum as phylum, Microbes.family as label, sum(t.value) as value from (select taxonomy_id, avg(value) as value from Microbe_Stats where Microbes.seq_type = \"" << seqType << "\" and patient_condition = \"" << condition << "\"" << regionField << " group by taxonomy_id)t inner join Microbes on t.taxonomy_id = Microbes.taxonomy_id group by Microbes.family order by family;";
			    break;
			}

		    case MGT_GENUS:
			{
			    valuess << "select Microbes.phylum as phylum, Microbes.genus as label, sum(t.value) as value from (select taxonomy_id, avg(value) as value from Microbe_Stats where Microbes.seq_type = \"" << seqType << "\" and patient_condition = \"" << condition << "\"" << regionField << " group by taxonomy_id)t inner join Microbes on t.taxonomy_id = Microbes.taxonomy_id group by Microbes.genus order by genus;";

			    break;
			}
		}

		break;
	    }
	default:
	    return false;
    }

    return addGraph(title,valuess.str());
}

bool MicrobeVerticalBarGraphObject::addGraph(std::string & label, std::string query)
{

    //std::cerr << "label: " << label << " query:" << std::endl << query << std::endl;

    struct tempData
    {
	char label[256];
	char phylum[256];
	float value;
    };

    struct tempData * data = NULL;
    int dataSize = 0;

    if(ComController::instance()->isMaster())
    {
	if(_dbm)
	{
	    DBMQueryResult result;

	    _dbm->runQuery(query,result);

	    dataSize = result.numRows();

	    if(dataSize)
	    {
		data = new struct tempData[dataSize];
	    }

	    for(int i = 0; i < dataSize; ++i)
	    {
		strncpy(data[i].label,result(i,"label").c_str(),254);
		data[i].label[255] = '\0';
		strncpy(data[i].phylum,result(i,"phylum").c_str(),254);
		data[i].phylum[255] = '\0';
		data[i].value = atof(result(i,"value").c_str());
	    }
	}

	ComController::instance()->sendSlaves(&dataSize,sizeof(int));
	if(dataSize)
	{
	    ComController::instance()->sendSlaves(data,dataSize*sizeof(struct tempData));
	}
    }
    else
    {
	ComController::instance()->readMaster(&dataSize,sizeof(int));
	if(dataSize)
	{
	    data = new struct tempData[dataSize];
	    ComController::instance()->readMaster(data,dataSize*sizeof(struct tempData));
	}
    }

    if(!dataSize)
    {
	std::cerr << "Warning MicrobeVerticalBarGraphObject: add Graph, query: " << query << ": zero data size" << std::endl;
	return false;
    }

    std::map<std::string,std::pair<float, int> > dataMap;

    for(int i = 0; i < dataSize; ++i)
    {
	dataMap[data[i].label] = std::pair<float,int>(data[i].value,_groupIndexMap[data[i].phylum]);
    }

    if(data)
    {
	delete[] data;
    }

    std::vector<float> dataValues;
    std::vector<int> groupIndexList;

    for(int i = 0; i < _nameList.size(); ++i)
    {
	dataValues.push_back(dataMap[_nameList[i]].first);
	groupIndexList.push_back(dataMap[_nameList[i]].second);
    }

    // sanity check
    float total = 0.0;
    for(int i = 0; i < dataValues.size(); ++i)
    {
	total += dataValues[i];
    }
    std::cerr << "Total: " << total << std::endl;

    return _graph->addBar(label,dataValues,groupIndexList);
}

void MicrobeVerticalBarGraphObject::objectAdded()
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

void MicrobeVerticalBarGraphObject::objectRemoved()
{
    GraphLayoutObject * layout = dynamic_cast<GraphLayoutObject*>(_parent);
    if(layout && layout->getPhylumKeyObject())
    {
	layout->getPhylumKeyObject()->unref(this);
    }
}

void MicrobeVerticalBarGraphObject::setGraphSize(float width, float height)
{
    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);

    updateSelect();

    _graph->setDisplaySize(width,height);
}

void MicrobeVerticalBarGraphObject::setNameList(std::vector<std::string> & nameList)
{
    _nameList = nameList;
    _graph->setDataLabels(nameList);
}

void MicrobeVerticalBarGraphObject::setGroupList(std::vector<std::string> & groupList)
{
    _groupList = groupList;
    _graph->setDataGroups(groupList);
    _groupIndexMap.clear();
    for(int i = 0; i < groupList.size(); ++i)
    {
	_groupIndexMap[groupList[i]] = i;
    }
}

void MicrobeVerticalBarGraphObject::selectMicrobes(std::string & group, std::vector<std::string> & keys)
{
    //std::cerr << "Select microbes called group: " << group << " m: " << (keys.size() ? keys[0] : "") << std::endl;
    _graph->selectItems(group,keys);
}

float MicrobeVerticalBarGraphObject::getGroupValue(std::string group, int i)
{
    return _graph->getGroupValue(group,i);
}

float MicrobeVerticalBarGraphObject::getMicrobeValue(std::string group, std::string key, int i)
{
    return _graph->getValue(group,key,i);
}

int MicrobeVerticalBarGraphObject::getNumDisplayValues()
{
    return _graph->getNumBars();
}

std::string MicrobeVerticalBarGraphObject::getDisplayLabel(int i)
{
    return _graph->getBarLabel(i);
}

void MicrobeVerticalBarGraphObject::dumpState(std::ostream & out)
{
    out << "MICROBE_BAR_GRAPH" << std::endl;

    out << _loadedGraphs.size() << std::endl;

    for(int i = 0; i < _loadedGraphs.size(); ++i)
    {
	out << _loadedGraphs[i].special << std::endl;
	if(_loadedGraphs[i].special)
	{
	    out << _loadedGraphs[i].type << std::endl;
	}
	else
	{
	    out << _loadedGraphs[i].patientid << std::endl;
	    out << _loadedGraphs[i].label << std::endl;
	    out << _loadedGraphs[i].testLabel << std::endl;
	}
    }
}

bool MicrobeVerticalBarGraphObject::loadState(std::istream & in)
{
    int graphs;
    std::string seqType;
    in >> graphs >> seqType;

    for(int i = 0; i < graphs; ++i)
    {
	bool special;
	in >> special;

	if(special)
	{
	    int type;
	    in >> type;
	    addSpecialGraph((SpecialMicrobeGraphType)type,seqType,"","","");
	}
	else
	{
	    int id;
	    in >> id;

	    char tempstr[1024];
	    // consume endl
	    in.getline(tempstr,1024);

	    std::string label, testLabel;
	    in.getline(tempstr,1024);
	    label = tempstr;
	    in.getline(tempstr,1024);
	    testLabel = tempstr;

	    addGraph(label,id,testLabel,0,seqType,"","");
	}
    }

    return true;
}

bool MicrobeVerticalBarGraphObject::processEvent(InteractionEvent * ie)
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
	bool selectValid = false;

	if(linePlaneIntersectionRef(start,end,planePoint,planeNormal,intersect,w))
	{
	    if(_graph->processClick(intersect,selectedGroup,selectedKeys,selectValid))
	    {
		clickUsed = true;
	    }
	    if(selectValid)
	    {
		layout->selectMicrobes(selectedGroup,selectedKeys);
	    }
	}

	if(clickUsed)
	{
	    return true;
	}
    }

    return FPTiledWallSceneObject::processEvent(ie);
}

void MicrobeVerticalBarGraphObject::updateCallback(int handID, const osg::Matrix & mat)
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

void MicrobeVerticalBarGraphObject::leaveCallback(int handID)
{
    if((_desktopMode && TrackingManager::instance()->getHandTrackerType(handID) == TrackerBase::MOUSE) ||
	(!_desktopMode && TrackingManager::instance()->getHandTrackerType(handID) == TrackerBase::POINTER))
    {
	_graph->clearHoverText();
    }
}

void MicrobeVerticalBarGraphObject::menuCallback(MenuItem * item)
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

void MicrobeVerticalBarGraphObject::makeSelect()
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

void MicrobeVerticalBarGraphObject::updateSelect()
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
