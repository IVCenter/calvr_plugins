#include "VerticalStackedBarGraph.h"
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

VerticalStackedBarGraph::VerticalStackedBarGraph(std::string title)
{
    _title = title;
    _width = 1000.0;
    _height = 1000.0;

    _root = new osg::Group();
    _bgScaleMT = new osg::MatrixTransform();
    _bgGeode = new osg::Geode();
    _axisGeode = new osg::Geode();
    _graphGeode = new osg::Geode();
    _groupLineGeode = new osg::Geode();

    _graphGeode->setCullingActive(false);
    _groupLineGeode->setCullingActive(false);
    _axisGeode->setCullingActive(false);

    _root->addChild(_bgScaleMT);
    _bgScaleMT->addChild(_bgGeode);
    _root->addChild(_axisGeode);
    _root->addChild(_graphGeode);
    _root->addChild(_groupLineGeode);

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

    _leftPaddingMult = 0.04;
    _rightPaddingMult = 0.04;
    _topPaddingMult = 0.12;
    _bottomPaddingMult = 0.1;
    _barToConnectorRatio = 1.5;

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

VerticalStackedBarGraph::~VerticalStackedBarGraph()
{
}

void VerticalStackedBarGraph::setDataLabels(std::vector<std::string> & dataLabels)
{
    _dataLabels = dataLabels;
}

void VerticalStackedBarGraph::setDataGroups(std::vector<std::string> & groupList)
{
    _dataGroups = groupList;
}

bool VerticalStackedBarGraph::addBar(std::string label, std::vector<float> & values, std::vector<int> & groupIndexList)
{
    if(!values.size() || values.size() != _dataLabels.size())
    {
	std::cerr << "Unexpected number of values: " << values.size() << std::endl;
	return false;
    }

    _dataList.push_back(values);
    _dataGroupIndexLists.push_back(groupIndexList);
    _barLabels.push_back(label);

    if(_geometryList.size())
    {
	_connectionGeometryList.push_back(makeGeometry(_dataLabels.size()));
	_graphGeode->addDrawable(_connectionGeometryList.back());
    }
    _geometryList.push_back(makeGeometry(_dataLabels.size()));
    _graphGeode->addDrawable(_geometryList.back());

    update();

    return true;
}

int VerticalStackedBarGraph::getNumBars()
{
    return _dataList.size();
}

std::string VerticalStackedBarGraph::getBarLabel(int bar)
{
    if(bar >= 0 && bar < _barLabels.size())
    {
	return _barLabels[bar];
    }

    return "";
}

float VerticalStackedBarGraph::getValue(std::string group, std::string key, int bar)
{
    if(bar >= 0 && bar < _dataList.size())
    {
	for(int i = 0; i < _dataLabels.size(); ++i)
	{
	    if(_dataLabels[i] == key && _dataGroups[_dataGroupIndexLists[bar][i]] == group)
	    {
		return _dataList[bar][i];
	    }
	}
    }
    return -1;
}

float VerticalStackedBarGraph::getGroupValue(std::string group, int bar)
{
    if(bar >= 0 && bar < _dataList.size())
    {
	int groupIndex = -1;
	for(int i = 0; i < _dataGroups.size(); ++i)
	{
	    if(group == _dataGroups[i])
	    {
		groupIndex = i;
		break;
	    }
	}
	if(groupIndex >= 0)
	{
	    float total = 0.0;
	    for(int i = 0; i < _dataGroupIndexLists[bar].size(); ++i)
	    {
		if(_dataGroupIndexLists[bar][i] == groupIndex)
		{
		    total += _dataList[bar][i];
		}
	    }
	    return total;
	}
    }
    return -1;
}

void VerticalStackedBarGraph::setDisplaySize(float width, float height)
{
    _width = width;
    _height = height;

    update();
}

void VerticalStackedBarGraph::setHover(osg::Vec3 intersect)
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

    bool graphHit = false;
    bool connectorHit = false;

    int graphIndex = 0;
    int connectorIndex = 0;

    float myLeft = _graphLeft;

    while(myLeft < _graphRight)
    {
	myLeft += _barWidth;
	if(intersect.x() < myLeft)
	{
	    graphHit = true;
	    break;
	}
	graphIndex++;

	myLeft += _connectorWidth;
	if(intersect.x() < myLeft)
	{
	    connectorHit = true;
	    break;
	}
	connectorIndex++;
    }

    if(graphHit)
    {
	if(graphIndex >= _dataList.size())
	{
	    clearHoverText();
	    return;
	}

	float myTop = _graphTop;
	bool found = false;
	int foundIndex;

	for(int i = 0; i < _dataList[graphIndex].size(); ++i)
	{
	    if(intersect.z() > myTop)
	    {
		break;
	    }

	    myTop -= _dataList[graphIndex][i] * (_graphTop - _graphBottom);

	    if(intersect.z() > myTop)
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
	    std::stringstream hoverss;
	    //hoverss << hoverGroup << std::endl;
	    hoverss << _dataLabels[foundIndex] << std::endl;
	    hoverss << "Value: " << _dataList[graphIndex][foundIndex];
	    /*if(!_dataUnitsList[graphIndex].empty())
	    {
		hoverss << " " << _dataUnitsList[graphIndex];
	    }*/

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

	    _currentHoverValue = _dataLabels[foundIndex];
	}
    }
    else if(connectorHit)
    {
	if(connectorIndex >= _connectionGeometryList.size())
	{
	    clearHoverText();
	    return;
	}

	float ratio = (myLeft - intersect.x()) / _connectorWidth;
	float graphTopLeft = _graphTop;
	float graphTopRight = _graphTop;
	float myTop = _graphTop;

	//float graphUpLeft = graphLeft + ((1.0 - (currentNodes[connectorIndex]->value / maxSize)) * (graphRight-graphLeft)) / 2.0;
	//float graphDownLeft = graphLeft + ((1.0 - (currentNodes[connectorIndex+1]->value / maxSize)) * (graphRight-graphLeft)) / 2.0;

	//float myLeft = graphUpLeft * ratio + graphDownLeft * (1.0 - ratio);
	bool found = false;
	int foundIndex = -1;

	for(int i = 0; i < _dataList[connectorIndex].size(); ++i)
	{
	    if(intersect.z() > myTop)
	    {
		break;
	    }

	    graphTopLeft -= _dataList[connectorIndex][i] * (_graphTop - _graphBottom);
	    graphTopRight -= _dataList[connectorIndex+1][i] * (_graphTop - _graphBottom);

	    myTop = graphTopLeft * ratio + graphTopRight * (1.0 - ratio);

	    if(intersect.z() > myTop)
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
	    std::stringstream hoverss;
	    //hoverss << currentNodes[connectorIndex]->name << " - " << currentNodes[connectorIndex+1]->name << std::endl;
	    //hoverss << hoverGroup << std::endl;
	    hoverss << _dataLabels[foundIndex] << std::endl;
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

	    _currentHoverValue = _dataLabels[foundIndex];
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

void VerticalStackedBarGraph::clearHoverText()
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

void VerticalStackedBarGraph::selectItems(std::string & group, std::vector<std::string> & keys)
{
    _lastSelectGroup = group;
    _lastSelectKeys = keys;

    if(!_dataList.size())
    {
	return;
    }

    //std::cerr << "select group: " << group << " keysize: " << keys.size() << std::endl;

    std::vector<int> indexList;

    // select phylum
    if(!group.empty() && !keys.size())
    {
	int groupIndex = -1;
	for(int i = 0; i < _dataGroups.size(); ++i)
	{
	    if(group == _dataGroups[i])
	    {
		groupIndex = i;
		break;
	    }
	}
	if(groupIndex != -1)
	{
	    for(int i = 0; i < _dataGroupIndexLists[0].size(); ++i)
	    {
		if(_dataGroupIndexLists[0][i] == groupIndex)
		{
		    indexList.push_back(i);
		}
	    }
	}
    }

    for(int i = 0; i < keys.size(); ++i)
    {
	//std::cerr << "Input key: " << keys[i] << std::endl;
	for(int j = 0; j < _dataLabels.size(); ++j)
	{
	    if(keys[i] == _dataLabels[j])
	    {
		//std::cerr << "Found index: " << j << std::endl;
		indexList.push_back(j);
		break;
	    }
	}
    }

    float selectedAlpha = 1.0;
    float connSelectedAlpha = 0.45;
    float notSelectedAlpha;

    if(keys.size() || !group.empty())
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
	    colors->at(j).w() = 0.3;
	}

	for(int j = 0; j < indexList.size(); ++j)
	{
	    colors->at(indexList[j]*4).w() = connSelectedAlpha;
	    colors->at(indexList[j]*4+1).w() = connSelectedAlpha;
	    colors->at(indexList[j]*4+2).w() = connSelectedAlpha;
	    colors->at(indexList[j]*4+3).w() = connSelectedAlpha;
	}
	colors->dirty();
	_connectionGeometryList[i]->dirtyDisplayList();
    }
}

bool VerticalStackedBarGraph::processClick(osg::Vec3 & intersect, std::string & selectedGroup, std::vector<std::string> & selectedKeys, bool & selectValid)
{
    if(!_dataList.size())
    {
	return false;
    }

    // see if intersect is in graph
    if(intersect.x() >= _graphLeft && intersect.x() <= _graphRight && intersect.z() >= _graphBottom && intersect.z() <= _graphTop)
    {
	if(!_currentHoverValue.empty())
	{
	    selectedKeys.push_back(_currentHoverValue);
	    //TODO store index
	    for(int i = 0; i < _dataLabels.size(); ++i)
	    {
		if(_dataLabels[i] == _currentHoverValue)
		{
		    selectedGroup = _dataGroups[_dataGroupIndexLists[0][i]];
		}
	    }
	}
	selectValid = true;
	return true;
    }

    selectValid = true;
    return false;
}

void VerticalStackedBarGraph::setColorMapping(const std::map<std::string,osg::Vec4> & colorMap, osg::Vec4 defaultColor)
{
    _colorMap = colorMap;
    _defaultColor = defaultColor;
    update();
}

void VerticalStackedBarGraph::makeBG()
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

void VerticalStackedBarGraph::makeHover()
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

osg::Geometry * VerticalStackedBarGraph::makeGeometry(int elements)
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

void VerticalStackedBarGraph::update()
{
    osg::Vec3 scale(_width,1.0,_height);
    osg::Matrix scaleMat;
    scaleMat.makeScale(scale);
    _bgScaleMT->setMatrix(scaleMat);

    updateValues();
    updateAxis();
    updateGraph();
}

void VerticalStackedBarGraph::updateAxis()
{
    _axisGeode->removeDrawables(0,_axisGeode->getNumDrawables());

    if(!_dataList.size())
    {
	return;
    }

    osg::Geometry * lineGeom = new osg::Geometry();
    osg::Vec3Array * lineVertArray = new osg::Vec3Array();
    osg::Vec4Array * lineColorArray = new osg::Vec4Array(1);
    lineColorArray->at(0) = osg::Vec4(0,0,0,1);
    lineGeom->setVertexArray(lineVertArray);
    lineGeom->setColorArray(lineColorArray);
    lineGeom->setColorBinding(osg::Geometry::BIND_OVERALL);
    _axisGeode->addDrawable(lineGeom);

    // draw graph outline
    lineVertArray->push_back(osg::Vec3(_graphLeft,-1,_graphTop));
    lineVertArray->push_back(osg::Vec3(_graphRight,-1,_graphTop));
    lineVertArray->push_back(osg::Vec3(_graphLeft,-1,_graphTop));
    lineVertArray->push_back(osg::Vec3(_graphLeft,-1,_graphBottom));
    lineVertArray->push_back(osg::Vec3(_graphLeft,-1,_graphBottom));
    lineVertArray->push_back(osg::Vec3(_graphRight,-1,_graphBottom));
    lineVertArray->push_back(osg::Vec3(_graphRight,-1,_graphTop));
    lineVertArray->push_back(osg::Vec3(_graphRight,-1,_graphBottom));

    // draw top and bottom lines
    float myLeft = _graphLeft;
    for(int i = 0; i < _dataList.size(); ++i)
    {
	lineVertArray->push_back(osg::Vec3(myLeft,-1,_graphTop));
	lineVertArray->push_back(osg::Vec3(myLeft,-1,_graphBottom));
	lineVertArray->push_back(osg::Vec3(myLeft+_barWidth,-1,_graphTop));
	lineVertArray->push_back(osg::Vec3(myLeft+_barWidth,-1,_graphBottom));
	myLeft += _barWidth + _connectorWidth;
    }

    osg::BoundingBox bb;

    // make title
    osgText::Text * titleText = GraphGlobals::makeText(_title,osg::Vec4(0,0,0,1));
    bb = titleText->getBoundingBox();
    float csize1 = (0.8 * _topPaddingMult * _height) / (bb.zMax() - bb.zMin());
    float csize2 = (_width * 0.9) / (bb.xMax() - bb.xMin());
    titleText->setCharacterSize(std::min(csize1,csize2));
    titleText->setAlignment(osgText::Text::CENTER_CENTER);
    titleText->setPosition(osg::Vec3(0,-1,(_height/2.0)-(_topPaddingMult*_height)/2.0));
    _axisGeode->addDrawable(titleText);

    osg::Quat q;
    q.makeRotate(M_PI/2.0,osg::Vec3(1.0,0,0));

    if((0.8 * _bottomPaddingMult * _height) > (_barWidth * 0.9))
    {
	q = q * osg::Quat(-M_PI/2.0,osg::Vec3(0,1.0,0));
    }

    //make bar labels
    myLeft = _graphLeft + (_barWidth / 2.0);
    for(int i = 0; i < _dataList.size(); ++i)
    {
	osgText::Text * text = GraphGlobals::makeText(_barLabels[i],osg::Vec4(0,0,0,1));
	text->setRotation(q);
	bb = text->getBoundingBox();
	float csize1 = (0.8 * _bottomPaddingMult * _height) / (bb.zMax() - bb.zMin());
	float csize2 = (_barWidth * 0.9) / (bb.xMax() - bb.xMin());
	text->setCharacterSize(std::min(csize1,csize2));
	text->setAlignment(osgText::Text::CENTER_CENTER);
	text->setPosition(osg::Vec3(myLeft,-1,-(_height/2.0)+(_bottomPaddingMult*_height)/2.0));
	_axisGeode->addDrawable(text);

	myLeft += _barWidth + _connectorWidth;
    }


    lineGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES,0,lineVertArray->size()));
}

void VerticalStackedBarGraph::updateGraph()
{
    _groupLineGeode->removeDrawables(0,_groupLineGeode->getNumDrawables());

    if(!_dataList.size())
    {
	return;
    }

    osg::Geometry * lineGeom = new osg::Geometry();
    osg::Vec3Array * lineVertArray = new osg::Vec3Array();
    osg::Vec4Array * lineColorArray = new osg::Vec4Array(1);
    lineColorArray->at(0) = osg::Vec4(0,0,0,1);
    lineGeom->setVertexArray(lineVertArray);
    lineGeom->setColorArray(lineColorArray);
    lineGeom->setColorBinding(osg::Geometry::BIND_OVERALL);
    _groupLineGeode->addDrawable(lineGeom);

    for(int i = 0; i < _dataList.size(); ++i)
    {
	int lastGroup = -1;
	osg::Vec3Array * myVerts;
	osg::Vec4Array * myColors;
	osg::Vec3Array * leftVerts = NULL;
	osg::Vec4Array * leftColors = NULL;
	osg::Vec3Array * rightVerts = NULL;
	osg::Vec4Array * rightColors = NULL;

	myVerts = dynamic_cast<osg::Vec3Array*>(_geometryList[i]->getVertexArray());
	myColors = dynamic_cast<osg::Vec4Array*>(_geometryList[i]->getColorArray());
	if(!myVerts || !myColors)
	{
	    std::cerr << "Error getting my arrays." << std::endl;
	    continue;
	}

	if(i > 0)
	{
	    leftVerts = dynamic_cast<osg::Vec3Array*>(_connectionGeometryList[i-1]->getVertexArray());
	    leftColors = dynamic_cast<osg::Vec4Array*>(_connectionGeometryList[i-1]->getColorArray());
	    if(!leftVerts || !leftColors)
	    {
		std::cerr << "Error getting left connection arrays." << std::endl;
		leftVerts = NULL;
		leftColors = NULL;
	    }
	}

	if(i+1 < _dataList.size())
	{
	    rightVerts = dynamic_cast<osg::Vec3Array*>(_connectionGeometryList[i]->getVertexArray());
	    rightColors = dynamic_cast<osg::Vec4Array*>(_connectionGeometryList[i]->getColorArray());
	    if(!rightVerts || !rightColors)
	    {
		std::cerr << "Error getting right connection arrays." << std::endl;
		rightVerts = NULL;
		rightColors = NULL;
	    }
	}

	float myLeft = _graphLeft + ((float)i) * (_barWidth + _connectorWidth);
	float myRight = myLeft + _barWidth;
	float myTop = _graphTop;

	float connAlpha = 0.3;

	bool lastColorSolid = true;

	for(int j = 0; j < _dataLabels.size(); ++j)
	{
	    float offset = _dataList[i][j] * (_graphTop - _graphBottom);
	    //std::cerr << "My left: " << myLeft << " offset: " << offset << std::endl;
	    // corners
	    osg::Vec3 ul(myLeft,0,myTop);
	    osg::Vec3 ll(myLeft,0,myTop-offset);
	    osg::Vec3 ur(myRight,0,myTop);
	    osg::Vec3 lr(myRight,0,myTop-offset);

	    myVerts->at(4*j) = ur;
	    myVerts->at(4*j+1) = ul;
	    myVerts->at(4*j+2) = ll;
	    myVerts->at(4*j+3) = lr;

	    osg::Vec4 color;

	    if(!_colorMap.size())
	    {
		int colorNum = j;
		if(colorNum % 2)
		{
		    colorNum = colorNum / 2 + _dataLabels.size() / 2;
		}
		else
		{
		    colorNum = colorNum / 2;
		}

		color = ColorGenerator::makeColor(colorNum,_dataLabels.size());
	    }
	    else
	    {
		std::map<std::string,osg::Vec4>::iterator it;
		it = _colorMap.find(_dataGroups[_dataGroupIndexLists[i][j]]);
		if(it != _colorMap.end())
		{
		    color = it->second;
		}
		else
		{
		    color = _defaultColor;
		}

		if(!lastColorSolid)
		{
		    color = color * 0.6;
		}

		lastColorSolid = !lastColorSolid;
	    }
	    myColors->at(4*j) = color;
	    myColors->at(4*j+1) = color;
	    myColors->at(4*j+2) = color;
	    myColors->at(4*j+3) = color;

	    if(leftVerts)
	    {
		leftVerts->at(4*j+0) = ul;
		leftVerts->at(4*j+3) = ll;

		leftColors->at(4*j+0) = color;
		leftColors->at(4*j+0).w() = connAlpha;
		leftColors->at(4*j+3) = color;
		leftColors->at(4*j+3).w() = connAlpha;
	    }

	    if(rightVerts)
	    {
		rightVerts->at(4*j+1) = ur;
		rightVerts->at(4*j+2) = lr;

		rightColors->at(4*j+1) = color;
		rightColors->at(4*j+1).w() = connAlpha;
		rightColors->at(4*j+2) = color;
		rightColors->at(4*j+2).w() = connAlpha;
	    }
	    myTop -= offset;
	}

	osg::DrawArrays * da = dynamic_cast<osg::DrawArrays*>(_geometryList[i]->getPrimitiveSet(0));
	if(da)
	{
	    da->setFirst(0);
	    da->setCount(_dataLabels.size()*4);
	}
	else
	{
	    std::cerr << "Unable to get geometry DrawArrays." << std::endl;
	}

        myVerts->dirty();
	myColors->dirty();
	_geometryList[i]->dirtyDisplayList();

	if(leftVerts)
	{
	    da = dynamic_cast<osg::DrawArrays*>(_connectionGeometryList[i-1]->getPrimitiveSet(0));
	    if(da)
	    {
		da->setFirst(0);
		da->setCount(_dataLabels.size()*4);
	    }
	    else
	    {
		std::cerr << "Unable to get connection geometry DrawArrays" << std::endl;
	    }
	    leftVerts->dirty();
	    leftColors->dirty();
            _connectionGeometryList[i-1]->dirtyDisplayList();
	}
    }

    if(_lastSelectKeys.size())
    {
	selectItems(_lastSelectGroup,_lastSelectKeys);
    }

    lineGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES,0,lineVertArray->size()));
}

void VerticalStackedBarGraph::updateValues()
{
    _graphLeft = (_width * _leftPaddingMult) - (_width / 2.0);
    _graphRight = _width * (1.0 - _rightPaddingMult) - (_width / 2.0);
    _graphTop = _height * (1.0 - _topPaddingMult) - (_height / 2.0);
    _graphBottom = (_height * _bottomPaddingMult) - (_height / 2.0);

    float total = ((float)_dataList.size())*_barToConnectorRatio + ((float)_dataList.size()) - 1.0f;

    _barWidth = (_barToConnectorRatio / total) * (_graphRight - _graphLeft);
    _connectorWidth = (1.0 / total) * (_graphRight - _graphLeft);
}
