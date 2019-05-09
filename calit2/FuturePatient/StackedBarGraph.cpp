#include "StackedBarGraph.h"
#include "ColorGenerator.h"
#include "GraphGlobals.h"

#include <cvrKernel/CalVR.h>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/ComController.h>

#include <osg/Geometry>

#include <iostream>
#include <sstream>
#include <cfloat>

using namespace cvr;

StackedBarGraph::StackedBarGraph(std::string title, float width, float height)
{
    _title = title;
    _width = width;
    _height = height;

    _root = new osg::Group();
    _bgScaleMT = new osg::MatrixTransform();
    _bgGeode = new osg::Geode();
    _axisGeode = new osg::Geode();
    _graphGeode = new osg::Geode();

    _graphGeode->setCullingActive(false);
    _axisGeode->setCullingActive(false);

    _root->addChild(_bgScaleMT);
    _bgScaleMT->addChild(_bgGeode);
    _root->addChild(_axisGeode);
    _root->addChild(_graphGeode);

    _lineWidth = new osg::LineWidth();
    _lineWidth->setWidth(ConfigManager::getFloat("value","Plugin.FuturePatient.StackedBarLineWidth",1.0));

    if(ComController::instance()->isMaster())
    {
	_lineWidth->setWidth(_lineWidth->getWidth()*ConfigManager::getFloat("value","Plugin.FuturePatient.MasterLineScale",1.0));
    }

    osg::StateSet * stateset = _root->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    stateset->setAttributeAndModes(_lineWidth,osg::StateAttribute::ON);

    stateset = _graphGeode->getOrCreateStateSet();
    stateset->setMode(GL_BLEND,osg::StateAttribute::ON);
    stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    _leftPaddingMult = 0.08;
    _rightPaddingMult = 0.04;
    _topPaddingMult = 0.15;
    _bottomPaddingMult = 0.05;
    _barToConnectorRatio = 1.1;

    _topTitleMult = 0.4;
    _topLevelMult = 0.35;
    _topCatHeaderMult = 0.25;

    _lineGeometry = new osg::Geometry();
    _lineGeometry->setUseDisplayList(false);
    osg::Vec3Array * verts = new osg::Vec3Array();
    osg::Vec4Array * colors = new osg::Vec4Array(1);
    colors->at(0) = osg::Vec4(0.0,0.0,0.0,1.0);
    _lineGeometry->setVertexArray(verts);
    _lineGeometry->setColorArray(colors);
    _lineGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    _linePrimitive = new osg::DrawArrays(osg::PrimitiveSet::LINES,0,0);
    _lineGeometry->addPrimitiveSet(_linePrimitive);
    _graphGeode->addDrawable(_lineGeometry);
    
    makeBG();
    makeHover();

    update();
}

StackedBarGraph::~StackedBarGraph()
{
}

bool StackedBarGraph::addBar(SBGData * dataRoot, std::vector<std::string> & dataLabels, std::string dataUnits)
{
    _dataList.push_back(dataRoot);
    _dataLabelList.push_back(dataLabels);
    _dataUnitsList.push_back(dataUnits);

    if(_geometryList.size())
    {
	_connectionGeometryList.push_back(makeGeometry(dataRoot->flat.size()));
	_graphGeode->addDrawable(_connectionGeometryList.back());
    }
    _geometryList.push_back(makeGeometry(dataRoot->flat.size()));
    _graphGeode->addDrawable(_geometryList.back());

    update();
}

void StackedBarGraph::setDisplaySize(float width, float height)
{
    _width = width;
    _height = height;

    update();
}

void StackedBarGraph::setHover(osg::Vec3 intersect)
{
    //std::cerr << "Set Hover" << std::endl;
    if(!_hoverGeode)
    {
	return;
    }

    float graphLeft = _width * (_leftPaddingMult - 0.5);
    float graphRight = _width * (0.5 - _rightPaddingMult);
    float graphBottom = _height * (_bottomPaddingMult - 0.5);
    float graphTop = _height * (0.5 - _topPaddingMult);

    if(intersect.x() < graphLeft || intersect.x() > graphRight || intersect.z() < graphBottom || intersect.z() > graphTop)
    {
	clearHoverText();
	return;
    }

    //std::cerr << "Intersect x: " << intersect.x() << " y: " << intersect.y() << " z: " << intersect.z() << std::endl;

    float total = ((float)_dataList.size())*_barToConnectorRatio + ((float)_dataList.size()) - 1.0f;

    float barHeight = (_barToConnectorRatio / total) * (graphTop - graphBottom);
    float connectorHeight = (1.0 / total) * (graphTop - graphBottom);

    std::vector<SBGData*> currentNodes;

    // find root nodes for current level in all datasets
    int index = 0;
    for(int i = 0; i < _dataList.size(); ++i)
    {
	SBGData * myNode = _dataList[i];

	for(int j = 0; j < _currentPath.size(); ++j)
	{
	    bool found = false;
	    for(int k = 0; k < myNode->groups.size(); ++k)
	    {
		if(_currentPath[j] == myNode->groups[k]->name)
		{
		    found = true;
		    myNode = myNode->groups[k];
		    break;
		}
		else
		{
		    index += myNode->groups[k]->flat.size();
		}
	    }
	    if(!found)
	    {
		std::cerr << "Unable to find current node" << std::endl;
		clearHoverText();
		return;
	    }
	}
	currentNodes.push_back(myNode);
    }

    float maxSize = FLT_MIN;
    for(int i = 0; i < currentNodes.size(); ++i)
    {
	maxSize = std::max(maxSize,currentNodes[i]->value);
    }

    bool graphHit = false;
    bool connectorHit = false;

    int graphIndex = 0;
    int connectorIndex = 0;

    float graphHeight = graphTop;

    while(graphHeight > graphBottom)
    {
	graphHeight -= barHeight;
	if(intersect.z() > graphHeight)
	{
	    graphHit = true;
	    break;
	}
	graphIndex++;

	graphHeight -= connectorHeight;
	if(intersect.z() > graphHeight)
	{
	    connectorHit = true;
	    break;
	}
	connectorIndex++;
    }

    if(graphHit)
    {
	if(graphIndex >= currentNodes.size())
	{
	    clearHoverText();
	    return;
	}

	float myLeft = graphLeft + ((1.0 - (currentNodes[graphIndex]->value / maxSize)) * (graphRight-graphLeft)) / 2.0;
	bool found = false;
	int foundIndex;

	for(int i = 0; i < currentNodes[graphIndex]->flat.size(); ++i)
	{
	    if(intersect.x() < myLeft)
	    {
		break;
	    }

	    myLeft += (currentNodes[graphIndex]->flat[i]->value / maxSize) * (graphRight - graphLeft);

	    if(intersect.x() < myLeft)
	    {
		found = true;
		foundIndex = i;
		break;
	    }
	}

	if(!found)
	{
	    clearHoverText();
	    return;
	}
	else
	{
	    int tempIndex = foundIndex;
	    std::string hoverGroup;
	    for(int i = 0; i < currentNodes[graphIndex]->groups.size(); ++i)
	    {
		if(tempIndex < currentNodes[graphIndex]->groups[i]->flat.size())
		{
		    hoverGroup = currentNodes[graphIndex]->groups[i]->name;
		    break;
		}
		tempIndex -= currentNodes[graphIndex]->groups[i]->flat.size();
	    }

	    std::stringstream hoverss;
	    hoverss << hoverGroup << std::endl;
	    hoverss << currentNodes[graphIndex]->flat[foundIndex]->name << std::endl;
	    hoverss << "Value: " << currentNodes[graphIndex]->flat[foundIndex]->value;
	    if(!_dataUnitsList[graphIndex].empty())
	    {
		hoverss << " " << _dataUnitsList[graphIndex];
	    }

	    _hoverText->setCharacterSize(1.0);
	    _hoverText->setText(hoverss.str());
	    _hoverText->setAlignment(osgText::Text::LEFT_TOP);
	    osg::BoundingBox bb = _hoverText->getBoundingBox();
	    float csize = GraphGlobals::getHoverHeight() / (bb.zMax() - bb.zMin());
	    _hoverText->setCharacterSize(csize);
	    _hoverText->setPosition(osg::Vec3(intersect.x(),-2.5,intersect.z()));

	    float bgheight = (bb.zMax() - bb.zMin()) * csize;
	    float bgwidth = (bb.xMax() - bb.xMin()) * csize;
	    osg::Vec3Array * verts = dynamic_cast<osg::Vec3Array*>(_hoverBGGeom->getVertexArray());
	    if(verts)
	    {
		verts->at(0) = osg::Vec3(intersect.x()+bgwidth,-2,intersect.z()-bgheight);
		verts->at(1) = osg::Vec3(intersect.x()+bgwidth,-2,intersect.z());
		verts->at(2) = osg::Vec3(intersect.x(),-2,intersect.z());
		verts->at(3) = osg::Vec3(intersect.x(),-2,intersect.z()-bgheight);
		verts->dirty();
	    }

	    _currentHoverValue = currentNodes[graphIndex]->flat[foundIndex]->name;
	}
    }
    else if(connectorHit)
    {
	if(connectorIndex >= _connectionGeometryList.size())
	{
	    clearHoverText();
	    return;
	}

	float ratio = (intersect.z() - graphHeight) / connectorHeight;

	float graphUpLeft = graphLeft + ((1.0 - (currentNodes[connectorIndex]->value / maxSize)) * (graphRight-graphLeft)) / 2.0;
	float graphDownLeft = graphLeft + ((1.0 - (currentNodes[connectorIndex+1]->value / maxSize)) * (graphRight-graphLeft)) / 2.0;

	float myLeft = graphUpLeft * ratio + graphDownLeft * (1.0 - ratio);
	bool found = false;
	int foundIndex = -1;

	for(int i = 0; i < currentNodes[connectorIndex]->flat.size(); ++i)
	{
	    if(intersect.x() < myLeft)
	    {
		break;
	    }

	    graphUpLeft += (currentNodes[connectorIndex]->flat[i]->value / maxSize) * (graphRight - graphLeft);
	    graphDownLeft += (currentNodes[connectorIndex+1]->flat[i]->value / maxSize) * (graphRight - graphLeft);

	    myLeft = graphUpLeft * ratio + graphDownLeft * (1.0 - ratio);

	    if(intersect.x() < myLeft)
	    {
		found = true;
		foundIndex = i;
		break;
	    }
	}

	if(!found)
	{
	    clearHoverText();
	    return;
	}
	else
	{
	    int tempIndex = foundIndex;
	    std::string hoverGroup;
	    for(int i = 0; i < currentNodes[connectorIndex]->groups.size(); ++i)
	    {
		if(tempIndex < currentNodes[connectorIndex]->groups[i]->flat.size())
		{
		    hoverGroup = currentNodes[connectorIndex]->groups[i]->name;
		    break;
		}
		tempIndex -= currentNodes[connectorIndex]->groups[i]->flat.size();
	    }

	    std::stringstream hoverss;
	    //hoverss << currentNodes[connectorIndex]->name << " - " << currentNodes[connectorIndex+1]->name << std::endl;
	    hoverss << hoverGroup << std::endl;
	    hoverss << currentNodes[connectorIndex]->flat[foundIndex]->name << std::endl;
	    hoverss << "Value:";

	    _hoverText->setCharacterSize(1.0);
	    _hoverText->setText(hoverss.str());
	    _hoverText->setAlignment(osgText::Text::LEFT_TOP);
	    osg::BoundingBox bb = _hoverText->getBoundingBox();
	    float csize = GraphGlobals::getHoverHeight() / (bb.zMax() - bb.zMin());
	    _hoverText->setCharacterSize(csize);
	    _hoverText->setPosition(osg::Vec3(intersect.x(),-2.5,intersect.z()));

	    float bgheight = (bb.zMax() - bb.zMin()) * csize;
	    float bgwidth = (bb.xMax() - bb.xMin()) * csize;
	    osg::Vec3Array * verts = dynamic_cast<osg::Vec3Array*>(_hoverBGGeom->getVertexArray());
	    if(verts)
	    {
		verts->at(0) = osg::Vec3(intersect.x()+bgwidth,-2,intersect.z()-bgheight);
		verts->at(1) = osg::Vec3(intersect.x()+bgwidth,-2,intersect.z());
		verts->at(2) = osg::Vec3(intersect.x(),-2,intersect.z());
		verts->at(3) = osg::Vec3(intersect.x(),-2,intersect.z()-bgheight);
		verts->dirty();
	    }

	    _currentHoverValue = currentNodes[connectorIndex]->flat[foundIndex]->name;
	}
    }
    else
    {
	clearHoverText();
	return;
    }

    if(!_hoverGeode->getNumParents())
    {
	_root->addChild(_hoverGeode);
    }
}

void StackedBarGraph::clearHoverText()
{
    if(!_hoverGeode)
    {
	return;
    }

    _currentHoverValue = "";

    if(_hoverGeode->getNumParents())
    {
	_root->removeChild(_hoverGeode);
    }
}

void StackedBarGraph::selectItems(std::string & group, std::vector<std::string> & keys)
{
    _lastSelectGroup = group;
    _lastSelectKeys = keys;

    if(!_dataList.size())
    {
	return;
    }

    SBGData * myNode = _dataList[0];

    for(int j = 0; j < _currentPath.size(); ++j)
    {
	bool found = false;
	for(int k = 0; k < myNode->groups.size(); ++k)
	{
	    if(_currentPath[j] == myNode->groups[k]->name)
	    {
		found = true;
		myNode = myNode->groups[k];
		break;
	    }
	}
	if(!found)
	{
	    std::cerr << "Unable to find current node" << std::endl;
	    return;
	}
    }

    std::vector<int> indexList;
    for(int i = 0; i < keys.size(); ++i)
    {
	//std::cerr << "Input key: " << keys[i] << std::endl;
	for(int j = 0; j < myNode->flat.size(); ++j)
	{
	    if(keys[i] == myNode->flat[j]->name)
	    {
		//std::cerr << "Found index: " << j << std::endl;
		indexList.push_back(j);
		break;
	    }
	}
    }

    float selectedAlpha = 1.0;
    float notSelectedAlpha;

    if(keys.size())
    { 
	notSelectedAlpha = 0.3;
    }
    else
    {
	notSelectedAlpha = 1.0;
    }

    for(int i = 0; i < _geometryList.size(); ++i)
    {
	osg::Vec4Array * colors = dynamic_cast<osg::Vec4Array*>(_geometryList[i]->getColorArray());
	if(!colors)
	{
	    continue;
	}

	for(int j = 0; j < colors->size(); ++j)
	{
	    colors->at(j).w() = notSelectedAlpha;
	}

	for(int j = 0; j < indexList.size(); ++j)
	{
	    colors->at(indexList[j]*4).w() = selectedAlpha;
	    colors->at(indexList[j]*4+1).w() = selectedAlpha;
	    colors->at(indexList[j]*4+2).w() = selectedAlpha;
	    colors->at(indexList[j]*4+3).w() = selectedAlpha;
	}
	colors->dirty();
	_geometryList[i]->dirtyDisplayList();
    }

    for(int i = 0; i < _connectionGeometryList.size(); ++i)
    {
	osg::Vec4Array * colors = dynamic_cast<osg::Vec4Array*>(_connectionGeometryList[i]->getColorArray());
	if(!colors)
	{
	    continue;
	}

	for(int j = 0; j < colors->size(); ++j)
	{
	    colors->at(j).w() = notSelectedAlpha;
	}

	for(int j = 0; j < indexList.size(); ++j)
	{
	    colors->at(indexList[j]*4).w() = selectedAlpha;
	    colors->at(indexList[j]*4+1).w() = selectedAlpha;
	    colors->at(indexList[j]*4+2).w() = selectedAlpha;
	    colors->at(indexList[j]*4+3).w() = selectedAlpha;
	}
	colors->dirty();
	_connectionGeometryList[i]->dirtyDisplayList();
    }
}

bool StackedBarGraph::processClick(osg::Vec3 & intersect, std::vector<std::string> & selectedKeys, bool & selectValid)
{
    if(!_dataList.size())
    {
	return false;
    }

    float graphLeft = _width * (_leftPaddingMult - 0.5);
    float graphRight = _width * (0.5 - _rightPaddingMult);
    float graphBottom = _height * (_bottomPaddingMult - 0.5);
    float graphTop = _height * (0.5 - _topPaddingMult);

    // see if intersect is in graph
    if(intersect.x() >= graphLeft && intersect.x() <= graphRight && intersect.z() >= graphBottom && intersect.z() <= graphTop)
    {
	if(!_currentHoverValue.empty())
	{
	    selectedKeys.push_back(_currentHoverValue);
	}
	selectValid = true;
	return true;
    }

    // find if intersect is in group label
    float groupLabelHeight = _height * _topPaddingMult * _topCatHeaderMult;
    if(intersect.z() > graphTop && intersect.z() - graphTop < groupLabelHeight)
    {
	std::vector<SBGData*> currentNodes;

	// find root nodes for current level in all datasets
	int index = 0;
	for(int i = 0; i < _dataList.size(); ++i)
	{
	    SBGData * myNode = _dataList[i];

	    for(int j = 0; j < _currentPath.size(); ++j)
	    {
		bool found = false;
		for(int k = 0; k < myNode->groups.size(); ++k)
		{
		    if(_currentPath[j] == myNode->groups[k]->name)
		    {
			found = true;
			myNode = myNode->groups[k];
			break;
		    }
		    else
		    {
			index += myNode->groups[k]->flat.size();
		    }
		}
		if(!found)
		{
		    std::cerr << "Unable to find current node" << std::endl;
		    return false;
		}
	    }
	    currentNodes.push_back(myNode);
	}

	float maxSize = FLT_MIN;
	for(int i = 0; i < currentNodes.size(); ++i)
	{
	    maxSize = std::max(maxSize,currentNodes[i]->value);
	}

	if(!currentNodes[0]->groups.size() || !currentNodes[0]->groups[0]->groups.size())
	{
	    return false;
	}

	// find intersected group

	float myLeft = graphLeft + ((1.0 - (currentNodes[0]->value / maxSize)) * (graphRight-graphLeft)) / 2.0;
	float myRight = myLeft + (currentNodes[0]->value / maxSize) * (graphRight-graphLeft);
	float mySize = myRight - myLeft;
	
	for(int i = 0; i < currentNodes[0]->groups.size(); ++i)
	{
	    if(intersect.x() < myLeft)
	    {
		return false;
	    }

	    myLeft += (currentNodes[0]->groups[i]->value / currentNodes[0]->value) * mySize;

	    if(intersect.x() < myLeft)
	    {
		_currentPath.push_back(currentNodes[0]->groups[i]->name);
		update();
		break;
	    }
	}

	return true;
    }

    // find if intersect should pop path
    // TODO: do this better, maybe add a back button
    if(intersect.z() > graphTop)
    {
	if(_currentPath.size())
	{
	    _currentPath.pop_back();
	    update();
	}
	return true;
    }

    selectValid = true;
    return false;
}

void StackedBarGraph::makeBG()
{
    osg::Geometry * geom = new osg::Geometry();

    osg::Vec3Array * verts = new osg::Vec3Array(4);
    verts->at(0) = osg::Vec3(0.5,1,0.5);
    verts->at(1) = osg::Vec3(0.5,1,-0.5);
    verts->at(2) = osg::Vec3(-0.5,1,0.5);
    verts->at(3) = osg::Vec3(-0.5,1,-0.5);

    osg::Vec4Array * colors = new osg::Vec4Array(1);
    colors->at(0) = GraphGlobals::getBackgroundColor();

    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);
    geom->setVertexArray(verts);
    geom->setUseDisplayList(false);

    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP,0,4));

    _bgGeode->addDrawable(geom);
}

void StackedBarGraph::makeHover()
{
    _hoverGeode = new osg::Geode();
    _hoverBGGeom = new osg::Geometry();
    _hoverBGGeom->setUseDisplayList(false);
    _hoverText = GraphGlobals::makeText("",osg::Vec4(1,1,1,1));
    _hoverGeode->addDrawable(_hoverBGGeom);
    _hoverGeode->addDrawable(_hoverText);
    _hoverGeode->setCullingActive(false);

    osg::Vec3Array * verts = new osg::Vec3Array(4);
    osg::Vec4Array * colors = new osg::Vec4Array(1);
    colors->at(0) = osg::Vec4(0,0,0,1);

    _hoverBGGeom->setVertexArray(verts);
    _hoverBGGeom->setColorArray(colors);
    _hoverBGGeom->setColorBinding(osg::Geometry::BIND_OVERALL);
    _hoverBGGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,4));
}

osg::Geometry * StackedBarGraph::makeGeometry(int elements)
{
    osg::Geometry * geometry = new osg::Geometry();

    osg::Vec3Array * verts = new osg::Vec3Array(elements*4);
    osg::Vec4Array * colors = new osg::Vec4Array(elements*4);

    geometry->setVertexArray(verts);
    geometry->setColorArray(colors);
    geometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    geometry->setUseDisplayList(false);
    geometry->setUseVertexBufferObjects(true);

    geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,0));

    return geometry;
}

void StackedBarGraph::update()
{
    osg::Vec3 scale(_width,1.0,_height);
    osg::Matrix scaleMat;
    scaleMat.makeScale(scale);
    _bgScaleMT->setMatrix(scaleMat);

    updateAxis();
    updateGraph();
}

void StackedBarGraph::updateAxis()
{
    _axisGeode->removeDrawables(0,_axisGeode->getNumDrawables());

    if(!_dataList.size())
    {
	return;
    }

    // TODO: move this into update function and pass down to both levels
    std::vector<SBGData*> currentNodes;

    // find root nodes for current level in all datasets
    int index = 0;
    for(int i = 0; i < _dataList.size(); ++i)
    {
	SBGData * myNode = _dataList[i];

	for(int j = 0; j < _currentPath.size(); ++j)
	{
	    bool found = false;
	    for(int k = 0; k < myNode->groups.size(); ++k)
	    {
		if(_currentPath[j] == myNode->groups[k]->name)
		{
		    found = true;
		    myNode = myNode->groups[k];
		    break;
		}
		else
		{
		    index += myNode->groups[k]->flat.size();
		}
	    }
	    if(!found)
	    {
		std::cerr << "Unable to find current node" << std::endl;
		return;
	    }
	}
	currentNodes.push_back(myNode);
    }

    float maxSize = FLT_MIN;
    for(int i = 0; i < currentNodes.size(); ++i)
    {
	maxSize = std::max(maxSize,currentNodes[i]->value);
    }

    float numElements = currentNodes[0]->flat.size();

    //std::cerr << "Current Nodes size: " << currentNodes.size() << " elements: " << numElements << std::endl;

    float graphLeft = (_width * _leftPaddingMult) - (_width / 2.0);
    float graphRight = _width * (1.0 - _rightPaddingMult) - (_width / 2.0);
    float graphTop = _height * (1.0 - _topPaddingMult) - (_height / 2.0);
    float graphBottom = (_height * _bottomPaddingMult) - (_height / 2.0);

    float total = ((float)currentNodes.size())*_barToConnectorRatio + ((float)currentNodes.size()) - 1.0f;

    float barHeight = (_barToConnectorRatio / total) * (graphTop - graphBottom);
    float connectorHeight = (1.0 / total) * (graphTop - graphBottom);

    osg::Geometry * lineGeom = new osg::Geometry();
    osg::Vec3Array * lineVertArray = new osg::Vec3Array();
    osg::Vec4Array * lineColorArray = new osg::Vec4Array(1);
    lineColorArray->at(0) = osg::Vec4(0,0,0,1);
    lineGeom->setVertexArray(lineVertArray);
    lineGeom->setColorArray(lineColorArray);
    lineGeom->setColorBinding(osg::Geometry::BIND_OVERALL);
    _axisGeode->addDrawable(lineGeom);

    // draw top and bottom lines
    float myTop = graphTop;
    for(int i = 0; i < currentNodes.size(); ++i)
    {
	float myLeft = graphLeft + ((1.0 - (currentNodes[i]->value / maxSize)) * (graphRight-graphLeft)) / 2.0;
	float myRight = myLeft + (currentNodes[i]->value / maxSize) * (graphRight-graphLeft);
	lineVertArray->push_back(osg::Vec3(myLeft,-1,myTop));
	lineVertArray->push_back(osg::Vec3(myRight,-1,myTop));
	lineVertArray->push_back(osg::Vec3(myLeft,-1,myTop-barHeight));
	lineVertArray->push_back(osg::Vec3(myRight,-1,myTop-barHeight));
	myTop -= barHeight + connectorHeight;
    }

    float groupLabelHeight = _height * _topPaddingMult * _topCatHeaderMult;

    osg::ref_ptr<osgText::Text> tempText = GraphGlobals::makeText("Ay",osg::Vec4(0,0,0,1));
    osg::BoundingBox bb = tempText->getBoundingBox();
    float groupLabelCharSize = (_topCatHeaderMult * 0.7 * _topPaddingMult * _height) / (bb.zMax() - bb.zMin());

    // draw current group outlines
    for(int i = 0; i < currentNodes[0]->groups.size(); ++i)
    {
	myTop = graphTop;
	for(int j = 0; j < currentNodes.size(); ++j)
	{
	    // find start/end offsets
	    float offsetStart = 0;
	    float offsetEnd;
	    for(int k = 0; k < i; ++k)
	    {
		offsetStart += currentNodes[j]->groups[k]->value;
	    }
	    offsetEnd = offsetStart + currentNodes[j]->groups[i]->value;
	    //std::cerr << "offsetStart: " << offsetStart << " end: " << offsetEnd << std::endl;

	    // turn into absolute pos
	    float mySize = (currentNodes[j]->value / maxSize) * (graphRight-graphLeft);
	    float myLeft = graphLeft + ((1.0 - (currentNodes[j]->value / maxSize)) * (graphRight-graphLeft)) / 2.0;

	    offsetStart = myLeft + (offsetStart / currentNodes[j]->value) * mySize;
	    offsetEnd = myLeft + (offsetEnd / currentNodes[j]->value) * mySize;

	    // draw top labels if first graph
	    if(j == 0)
	    {
		lineVertArray->push_back(osg::Vec3(offsetStart,-1,graphTop+groupLabelHeight));
		lineVertArray->push_back(osg::Vec3(offsetStart,-1,graphTop));
		//if(i+1 == currentNodes[0]->groups.size())
		//{
		    //std::cerr << "offsetStart: " << offsetStart << " end: " << offsetEnd << std::endl;
		    //std::cerr << "Drawing end line" << std::endl;
		    //lineVertArray->push_back(osg::Vec3(offsetEnd,-1,graphTop+groupLabelHeight));
		    //lineVertArray->push_back(osg::Vec3(offsetEnd,-1,graphTop));
		//}

		// add group text
		osgText::Text * text = GraphGlobals::makeText(currentNodes[0]->groups[i]->name,osg::Vec4(0,0,0,1));
		text->setAlignment(osgText::Text::CENTER_CENTER);
		text->setCharacterSize(groupLabelCharSize);
		text->setPosition(osg::Vec3(offsetStart+((offsetEnd-offsetStart)/2.0),-1,graphTop+(groupLabelHeight/2.0)));
		GraphGlobals::makeTextFit(text,offsetEnd-offsetStart);

		_axisGeode->addDrawable(text);
	    }

	    if(j)
	    {
		lineVertArray->push_back(osg::Vec3(offsetStart,-1,myTop));
	    }

	    lineVertArray->push_back(osg::Vec3(offsetStart,-1,myTop));
	    lineVertArray->push_back(osg::Vec3(offsetStart,-1,myTop-barHeight));

	    if(j+1 != currentNodes.size())
	    {
		lineVertArray->push_back(osg::Vec3(offsetStart,-1,myTop-barHeight));
	    }

	    myTop -= barHeight + connectorHeight;
	}
    }

    // draw right lines
    myTop = graphTop;
    for(int i = 0; i < currentNodes.size(); ++i)
    {
	float myLeft = graphLeft + ((1.0 - (currentNodes[i]->value / maxSize)) * (graphRight-graphLeft)) / 2.0;
	float myRight = myLeft + (currentNodes[i]->value / maxSize) * (graphRight-graphLeft);
    
	if(i == 0)
	{
	    lineVertArray->push_back(osg::Vec3(myRight,-1,myTop+groupLabelHeight));
	    lineVertArray->push_back(osg::Vec3(myRight,-1,myTop));
	}

	if(i != 0)
	{
	    lineVertArray->push_back(osg::Vec3(myRight,-1,myTop));
	}

	lineVertArray->push_back(osg::Vec3(myRight,-1,myTop));
	lineVertArray->push_back(osg::Vec3(myRight,-1,myTop-barHeight));

	if(i+1 != currentNodes.size())
	{
	    lineVertArray->push_back(osg::Vec3(myRight,-1,myTop-barHeight));
	}
	myTop -= barHeight + connectorHeight;
    }

    // make title
    osgText::Text * titleText = GraphGlobals::makeText(_title,osg::Vec4(0,0,0,1));
    bb = titleText->getBoundingBox();
    float csize1 = (_topTitleMult * 0.8 * _topPaddingMult * _height) / (bb.zMax() - bb.zMin());
    float csize2 = (_width * 0.9) / (bb.xMax() - bb.xMin());
    titleText->setCharacterSize(std::min(csize1,csize2));
    titleText->setAlignment(osgText::Text::CENTER_CENTER);
    titleText->setPosition(osg::Vec3(0,-1,(_height/2.0)-(_topTitleMult*_topPaddingMult*_height)/2.0));
    _axisGeode->addDrawable(titleText);

    // make path text
    std::stringstream pathss;
    for(int i = 0; i < _currentPath.size(); ++i)
    {
	pathss << _currentPath[i];
	if(i+1 != _currentPath.size())
	{
	    pathss << " - ";
	}
    }
    if(!pathss.str().empty())
    {
	osgText::Text * pathText = GraphGlobals::makeText(pathss.str(),osg::Vec4(0,0,0,1));
	bb = pathText->getBoundingBox();

	csize1 = (_topLevelMult * 0.8 * _topPaddingMult * _height) / (bb.zMax() - bb.zMin());
	csize2 = (_width * 0.8) / (bb.xMax() - bb.xMin());
	pathText->setCharacterSize(std::min(csize1,csize2));
	pathText->setAlignment(osgText::Text::CENTER_CENTER);
	pathText->setPosition(osg::Vec3(0,-1,(_height/2.0)-((_topTitleMult+(_topLevelMult/2.0))*_topPaddingMult*_height)));
	_axisGeode->addDrawable(pathText);
    }

    osg::Quat q;
    q.makeRotate(M_PI/2.0,osg::Vec3(1.0,0,0));
    q = q * osg::Quat(-M_PI/2.0,osg::Vec3(0,1.0,0));

    // make data labels
    float dlHeight = graphTop - (barHeight / 2.0);
    for(int i = 0; i < _dataList.size(); ++i)
    {
	osgText::Text * dlText = GraphGlobals::makeText(_dataList[i]->name,osg::Vec4(0,0,0,1));
	if((barHeight * 0.9) >= (_leftPaddingMult * _width * 0.8))
	{
	    dlText->setRotation(q);
	}
	dlText->setAlignment(osgText::Text::CENTER_CENTER);
	bb = dlText->getBoundingBox();
	csize1 = (_leftPaddingMult * _width * 0.8) / (bb.xMax() - bb.xMin());
	csize2 = (barHeight * 0.9) / (bb.zMax() - bb.zMin());
	dlText->setCharacterSize(std::min(csize1,csize2));
	dlText->setPosition(osg::Vec3((-_width/2.0)+(_width*_leftPaddingMult/2.0),-1,dlHeight));
	_axisGeode->addDrawable(dlText);

	dlHeight -= barHeight + connectorHeight;
    }

    lineGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES,0,lineVertArray->size()));
}

void StackedBarGraph::updateGraph()
{
    if(!_dataList.size())
    {
	return;
    }

    std::vector<SBGData*> currentNodes;

    // find root nodes for current level in all datasets
    int index = 0;
    for(int i = 0; i < _dataList.size(); ++i)
    {
	index = 0;
	SBGData * myNode = _dataList[i];

	for(int j = 0; j < _currentPath.size(); ++j)
	{
	    bool found = false;
	    for(int k = 0; k < myNode->groups.size(); ++k)
	    {
		if(_currentPath[j] == myNode->groups[k]->name)
		{
		    found = true;
		    myNode = myNode->groups[k];
		    break;
		}
		else
		{
		    index += myNode->groups[k]->flat.size();
		}
	    }
	    if(!found)
	    {
		std::cerr << "Unable to find current node" << std::endl;
		return;
	    }
	}
	currentNodes.push_back(myNode);
    }

    float maxSize = FLT_MIN;
    for(int i = 0; i < currentNodes.size(); ++i)
    {
	maxSize = std::max(maxSize,currentNodes[i]->value);
    }


    float totalElements = _dataList[0]->flat.size();
    float numElements = currentNodes[0]->flat.size();

    //std::cerr << "Current Nodes size: " << currentNodes.size() << " elements: " << numElements << std::endl;

    float graphLeft = (_width * _leftPaddingMult) - (_width / 2.0);
    float graphRight = _width * (1.0 - _rightPaddingMult) - (_width / 2.0);
    float graphTop = _height * (1.0 - _topPaddingMult) - (_height / 2.0);
    float graphBottom = (_height * _bottomPaddingMult) - (_height / 2.0);

    float total = ((float)currentNodes.size())*_barToConnectorRatio + ((float)currentNodes.size()) - 1.0f;

    float barHeight = (_barToConnectorRatio / total) * (graphTop - graphBottom);
    float connectorHeight = (1.0 / total) * (graphTop - graphBottom);

    for(int i = 0; i < currentNodes.size(); ++i)
    {
	osg::Vec3Array * myVerts;
	osg::Vec4Array * myColors;
	osg::Vec3Array * upVerts = NULL;
	osg::Vec4Array * upColors = NULL;
	osg::Vec3Array * downVerts = NULL;
	osg::Vec4Array * downColors = NULL;

	myVerts = dynamic_cast<osg::Vec3Array*>(_geometryList[i]->getVertexArray());
	myColors = dynamic_cast<osg::Vec4Array*>(_geometryList[i]->getColorArray());
	if(!myVerts || !myColors)
	{
	    std::cerr << "Error getting my arrays." << std::endl;
	    continue;
	}

	if(i > 0)
	{
	    upVerts = dynamic_cast<osg::Vec3Array*>(_connectionGeometryList[i-1]->getVertexArray());
	    upColors = dynamic_cast<osg::Vec4Array*>(_connectionGeometryList[i-1]->getColorArray());
	    if(!upVerts || !upColors)
	    {
		std::cerr << "Error getting up connection arrays." << std::endl;
		upVerts = NULL;
		upColors = NULL;
	    }
	}

	if(i+1 < currentNodes.size())
	{
	    downVerts = dynamic_cast<osg::Vec3Array*>(_connectionGeometryList[i]->getVertexArray());
	    downColors = dynamic_cast<osg::Vec4Array*>(_connectionGeometryList[i]->getColorArray());
	    if(!downVerts || !downColors)
	    {
		std::cerr << "Error getting down connection arrays." << std::endl;
		downVerts = NULL;
		downColors = NULL;
	    }
	}

	float myLeft = graphLeft + ((1.0 - (currentNodes[i]->value / maxSize)) * (graphRight-graphLeft)) / 2.0;
	float myTop = graphTop - (((float)i) * (barHeight + connectorHeight));
	float myBottom = myTop - barHeight;

	for(int j = 0; j < numElements; ++j)
	{
	    float offset = (currentNodes[i]->flat[j]->value / maxSize) * (graphRight - graphLeft);
	    //std::cerr << "My left: " << myLeft << " offset: " << offset << std::endl;
	    // corners
	    osg::Vec3 ul(myLeft,0,myTop);
	    osg::Vec3 ll(myLeft,0,myBottom);
	    osg::Vec3 ur(myLeft+offset,0,myTop);
	    osg::Vec3 lr(myLeft+offset,0,myBottom);

	    myVerts->at(4*j) = ur;
	    myVerts->at(4*j+1) = ul;
	    myVerts->at(4*j+2) = ll;
	    myVerts->at(4*j+3) = lr;

	    int colorNum = j + index;
	    if(colorNum % 2)
	    {
		colorNum = colorNum / 2 + totalElements / 2;
	    }
	    else
	    {
		colorNum = colorNum / 2;
	    }

	    osg::Vec4 color = ColorGenerator::makeColor(colorNum,totalElements);
	    myColors->at(4*j) = color;
	    myColors->at(4*j+1) = color;
	    myColors->at(4*j+2) = color;
	    myColors->at(4*j+3) = color;

	    if(upVerts)
	    {
		upVerts->at(4*j+2) = ul;
		upVerts->at(4*j+3) = ur;

		upColors->at(4*j+2) = color;
		upColors->at(4*j+3) = color;
	    }

	    if(downVerts)
	    {
		downVerts->at(4*j) = lr;
		downVerts->at(4*j+1) = ll;

		downColors->at(4*j) = color;
		downColors->at(4*j+1) = color;
	    }
	    myLeft += offset;
	}

	osg::DrawArrays * da = dynamic_cast<osg::DrawArrays*>(_geometryList[i]->getPrimitiveSet(0));
	if(da)
	{
	    da->setFirst(0);
	    da->setCount(numElements*4);
	}
	else
	{
	    std::cerr << "Unable to get geometry DrawArrays." << std::endl;
	}

        myVerts->dirty();
	myColors->dirty();
	_geometryList[i]->dirtyDisplayList();

	if(upVerts)
	{
	    da = dynamic_cast<osg::DrawArrays*>(_connectionGeometryList[i-1]->getPrimitiveSet(0));
	    if(da)
	    {
		da->setFirst(0);
		da->setCount(numElements*4);
	    }
	    else
	    {
		std::cerr << "Unable to get connection geometry DrawArrays" << std::endl;
	    }
	    upVerts->dirty();
	    upColors->dirty();
            _connectionGeometryList[i-1]->dirtyDisplayList();
	}
    }

    if(_lastSelectKeys.size())
    {
	selectItems(_lastSelectGroup,_lastSelectKeys);
    }
}
