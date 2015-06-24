#include "OtuGraphObject.h"
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
#include <fstream>
#include <cstdlib>
#include <ctime>

// for time debug
#include <sys/time.h>

bool OtuGraphObject::_otuTableLoaded = false;
std::map<int,OtuGraphObject::OTUClassification*> OtuGraphObject::_otuTable;

using namespace cvr;

OtuGraphObject::OtuGraphObject(DBManager * dbm, float width, float height, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : LayoutTypeObject(name,navigation,movable,clip,contextMenu,showBounds), SelectableObject()
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
    }
   
    if(!_otuTableLoaded)
    {
	initOTUTable();
    }

    _graph = new GroupedBarGraph(width,height);
    _graph->setColorMapping(GraphGlobals::getDefaultPhylumColor(),GraphGlobals::getPhylumColorMap());
    _graph->setColorMode(BGCM_GROUP);
}

OtuGraphObject::~OtuGraphObject()
{
}

void OtuGraphObject::objectAdded()
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

void OtuGraphObject::objectRemoved()
{
    GraphLayoutObject * layout = dynamic_cast<GraphLayoutObject*>(_parent);
    if(layout && layout->getPhylumKeyObject())
    {
	layout->getPhylumKeyObject()->unref(this);
    }
}

void OtuGraphObject::setGraphSize(float width, float height)
{
    osg::BoundingBox bb(-(width*0.5),-2,-(height*0.5),width*0.5,0,height*0.5);
    setBoundingBox(bb);

    updateSelect();

    _graph->setDisplaySize(width,height);
}

void OtuGraphObject::setColor(osg::Vec4 color)
{
    _graph->setColor(color);
}

void OtuGraphObject::setBGColor(osg::Vec4 color)
{
    _graph->setBGColor(color);
}

float OtuGraphObject::getGraphMaxValue()
{
    return _graph->getDataMax();
}

float OtuGraphObject::getGraphMinValue()
{
    return _graph->getDataMin();
}

float OtuGraphObject::getGraphDisplayRangeMax()
{
    return _graph->getDisplayRangeMax();
}

float OtuGraphObject::getGraphDisplayRangeMin()
{
    return _graph->getDisplayRangeMin();
}

void OtuGraphObject::setGraphDisplayRange(float min, float max)
{
    _graph->setDisplayRange(min,max);
}

void OtuGraphObject::resetGraphDisplayRange()
{
    _graph->setDisplayRange(_graph->getDisplayRangeMin(),_graph->getDisplayRangeMax());
}

void OtuGraphObject::selectMicrobes(std::string & group, std::vector<std::string> & keys)
{
    _graph->selectItems(group,keys);
}

float OtuGraphObject::getGroupValue(std::string group, int i)
{
    return _graph->getGroupValue(group);
}

float OtuGraphObject::getMicrobeValue(std::string group, std::string key, int i)
{
    return _graph->getKeyValue(group,key);
}

int OtuGraphObject::getNumDisplayValues()
{
    return 1;
}

std::string OtuGraphObject::getDisplayLabel(int i)
{
    return getTitle();
}

void OtuGraphObject::dumpState(std::ostream & out)
{
    out << "OTU_GRAPH" << std::endl;
}

bool OtuGraphObject::loadState(std::istream & in)
{
    return true;
}

std::string OtuGraphObject::getTitle()
{
    return _graphTitle;
}

bool OtuGraphObject::processEvent(InteractionEvent * ie)
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

	    return true;
	}
    }

    return FPTiledWallSceneObject::processEvent(ie);
}

void OtuGraphObject::updateCallback(int handID, const osg::Matrix & mat)
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

void OtuGraphObject::leaveCallback(int handID)
{
    if((_desktopMode && TrackingManager::instance()->getHandTrackerType(handID) == TrackerBase::MOUSE) ||
	(!_desktopMode && TrackingManager::instance()->getHandTrackerType(handID) == TrackerBase::POINTER))
    {
	_graph->clearHoverText();
    }
}

void OtuGraphObject::menuCallback(MenuItem * item)
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

bool otuSort(const std::pair<std::string,float> & first, const std::pair<std::string,float> & second)
{
    return first.second > second.second;
}

bool OtuGraphObject::setGraph(std::string sample, int displayCount, MicrobeGraphType type)
{
    std::string otuDir = ConfigManager::getEntry("value","Plugin.FuturePatient.OtuDir","");
    std::string path = otuDir + "/" + sample;

    std::ifstream infile(path.c_str());
    float relab;
    std::string otus;

    std::string sublabel;
    OTUHierarchy level;
    switch(type)
    {
	default:
	case MGT_SPECIES:
	    level = OTU_S;
	    sublabel = "phylum / microbe";
	    break;
	case MGT_FAMILY:
	    level = OTU_F;
	    sublabel = "phylum / family";
	    break;
	case MGT_GENUS:
	    level = OTU_G;
	    sublabel = "phylum / genus";
	    break;
    }

    float total = 0.0;
    std::map<std::string,float> phylumTotals;
    std::map<std::string,std::map<std::string,float> > phylumData;
    //std::vector<std::pair<std::string,float> > otuList;
    while(infile >> otus >> relab)
    {
	int otu = atoi(otus.c_str());
	if(_otuTable.find(otu) == _otuTable.end())
	{
	    continue;
	}
	if(_otuTable[otu]->lastH < OTU_P)
	{
	    continue;
	}

	total += relab;

	phylumTotals[_otuTable[otu]->c[OTU_P]] += relab;
	if(_otuTable[otu]->lastH < level)
	{
	    phylumData[_otuTable[otu]->c[OTU_P]]["OTU_OTHER"] += relab;
	}
	else
	{
	    phylumData[_otuTable[otu]->c[OTU_P]][_otuTable[otu]->c[level]] += relab;
	}
	//otuList.push_back(std::pair<std::string,float>(otus,relab));
    }

    std::vector<std::pair<std::string,float> > phylumList;
    for(std::map<std::string,float>::iterator it = phylumTotals.begin(); it != phylumTotals.end(); ++it)
    {
	phylumList.push_back(std::pair<std::string,float>(it->first,it->second));
    }
    std::sort(phylumList.begin(),phylumList.end(),otuSort);

    std::vector<std::string> groupOrder;
    for(int i = 0; i < phylumList.size(); ++i)
    {
	groupOrder.push_back(phylumList[i].first);
    }

    if(true)
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
	    for(std::vector<std::string>::iterator it = groupOrder.begin(); it != groupOrder.end(); ++it)
	    {
		if(*it == reorderVec[i])
		{
		    groupOrder.erase(it);
		    groupOrder.insert(groupOrder.begin(),reorderVec[i]);
		    break;
		}
	    }
	}
    }

    std::map<std::string,std::vector<std::pair<std::string,float> > > dataMap;
    for(std::map<std::string,std::map<std::string,float> >::iterator it = phylumData.begin(); it != phylumData.end(); ++it)
    {
	for(std::map<std::string,float>::iterator itt = it->second.begin(); itt != it->second.end(); ++itt)
	{
	    dataMap[it->first].push_back(std::pair<std::string,float>(itt->first,itt->second));
	}
	std::sort(dataMap[it->first].begin(),dataMap[it->first].end(),otuSort);
    }

    size_t dotPos = sample.find_last_of(".");
    std::string title = sample.substr(0,dotPos);
    std::stringstream titless;
    titless << title << " - " << round(total * 100.0) << "%";

    bool graphValid = _graph->setGraph(titless.str(), dataMap, groupOrder, BGAT_LOG, "Relative Abundance", "", sublabel,osg::Vec4(1.0,0,0,1));

    if(graphValid)
    {
	addChild(_graph->getRootNode());
    }

    return graphValid;

    /*std::sort(otuList.begin(),otuList.end(),otuSort);

    std::map<std::string,std::vector<std::pair<std::string,float> > > dataMap;
    dataMap["OTU"] = std::vector<std::pair<std::string,float> >();

    for(int i = 0; i < otuList.size(); ++i)
    {
	dataMap["OTU"].push_back(otuList[i]);
	if(dataMap["OTU"].size() >= displayCount)
	{
	    break;
	}
    }

    std::vector<std::string> groupOrder;
    groupOrder.push_back("OTU");

    size_t dotPos = sample.find_last_of(".");
    std::string title = sample.substr(0,dotPos);

    bool graphValid = _graph->setGraph(title, dataMap, groupOrder, BGAT_LOG, "Relative Abundance", "", "",osg::Vec4(1.0,0,0,1));

    if(graphValid)
    {
	addChild(_graph->getRootNode());
    }

    return true;*/
}

void OtuGraphObject::makeSelect()
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

void OtuGraphObject::updateSelect()
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

void OtuGraphObject::initOTUTable()
{
    std::string otuFile = ConfigManager::getEntry("value","Plugin.FuturePatient.OtuFile","");

    std::ifstream infile(otuFile.c_str());
    if(!infile.fail())
    {
	std::string line;
	while(std::getline(infile,line))
	{
	    if(line.size() == 0)
	    {
		break;
	    }
	    if(line[0] == '#')
	    {
		continue;
	    }

	    OTUClassification * otuc = new OTUClassification;
	    size_t pos = line.find_first_of("k");
	    int id = atoi(line.substr(0,pos).c_str());

	    std::stringstream ss(line.substr(pos-1));

	    std::string token;
	    std::getline(ss,token,';');
	    otuc->c[0] = token.substr(4);
	    if(otuc->c[0].size() > 0)
	    {
		otuc->lastH = OTU_K;
	    }

	    std::getline(ss,token,';');
	    otuc->c[1] = token.substr(4);
	    if(otuc->c[1].size() > 0)
	    {
		otuc->lastH = OTU_P;
	    }

	    std::getline(ss,token,';');
	    otuc->c[2] = token.substr(4);
	    if(otuc->c[2].size() > 0)
	    {
		otuc->lastH = OTU_C;
	    }

	    std::getline(ss,token,';');
	    otuc->c[3] = token.substr(4);
	    if(otuc->c[3].size() > 0)
	    {
		otuc->lastH = OTU_O;
	    }

	    std::getline(ss,token,';');
	    otuc->c[4] = token.substr(4);
	    if(otuc->c[4].size() > 0)
	    {
		otuc->lastH = OTU_F;
	    }

	    std::getline(ss,token,';');
	    otuc->c[5] = token.substr(4);
	    if(otuc->c[5].size() > 0)
	    {
		otuc->lastH = OTU_G;
	    }

	    std::getline(ss,token,'\n');
	    otuc->c[6] = token.substr(4);
	    if(otuc->c[6].size() > 0)
	    {
		otuc->lastH = OTU_S;
	    }

	    _otuTable[id] = otuc;

	    //std::cerr << "id: " << id << " k: " << otuc->c[0] << " p: " << otuc->c[1] << " c: " << otuc->c[2] << " o: " << otuc->c[3] << " f: " << otuc->c[4] << " g: " << otuc->c[5] << " s: " << otuc->c[6] << " last: " << ((int)(otuc->lastH)) << std::endl;
	}
    }
    else
    {
	std::cerr << "Unable to open OTU file: " << otuFile << std::endl;
	return;
    }
    infile.close();

    _otuTableLoaded = true;
}
