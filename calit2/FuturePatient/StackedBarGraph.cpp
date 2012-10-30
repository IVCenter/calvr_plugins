#include "StackedBarGraph.h"
#include "ColorGenerator.h"

#include <cvrKernel/CalVR.h>

#include <osg/Geometry>

#include <iostream>
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

    osg::StateSet * stateset = _root->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    _leftPaddingMult = 0.1;
    _rightPaddingMult = 0.05;
    _topPaddingMult = 0.15;
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
    
    _font = osgText::readFontFile(CalVR::instance()->getHomeDir() + "/resources/arial.ttf");

    makeBG();

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

void StackedBarGraph::makeBG()
{
    osg::Geometry * geom = new osg::Geometry();

    osg::Vec3Array * verts = new osg::Vec3Array(4);
    verts->at(0) = osg::Vec3(0.5,1,0.5);
    verts->at(1) = osg::Vec3(0.5,1,-0.5);
    verts->at(2) = osg::Vec3(-0.5,1,0.5);
    verts->at(3) = osg::Vec3(-0.5,1,-0.5);

    osg::Vec4Array * colors = new osg::Vec4Array(1);
    colors->at(0) = osg::Vec4(1.0,1.0,1.0,1.0);

    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);
    geom->setVertexArray(verts);
    geom->setUseDisplayList(false);

    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP,0,4));

    _bgGeode->addDrawable(geom);
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

	    int colorNum;
	    if(j % 2)
	    {
		colorNum = j / 2 + numElements / 2;
	    }
	    else
	    {
		colorNum = j / 2;
	    }

	    osg::Vec4 color = ColorGenerator::makeColor(colorNum,numElements);
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
	}
    }
}

osgText::Text * StackedBarGraph::makeText(std::string text, osg::Vec4 color)
{
    osgText::Text * textNode = new osgText::Text();
    textNode->setCharacterSize(1.0);
    textNode->setAlignment(osgText::Text::CENTER_CENTER);
    textNode->setColor(color);
    textNode->setBackdropColor(osg::Vec4(0,0,0,0));
    textNode->setAxisAlignment(osgText::Text::XZ_PLANE);
    textNode->setText(text);
    if(_font)
    {
	textNode->setFont(_font);
    }
    return textNode;
}
