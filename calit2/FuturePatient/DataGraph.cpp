#include "DataGraph.h"
#include "ShapeTextureGenerator.h"
#include "ColorGenerator.h"

#include <cvrKernel/CalVR.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrUtil/OsgMath.h>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/ComController.h>

#include <osgText/Text>
#include <osg/Geode>

#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>

using namespace cvr;

std::string shapeVertSrc =
"#version 150 compatibility                                  \n"
"#extension GL_ARB_gpu_shader5 : enable                      \n"
"                                                            \n"
"void main(void)                                             \n"
"{                                                           \n"
"    gl_FrontColor = gl_Color;                               \n"
"    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex; \n"
"}                                                           \n";

std::string shapeFragSrc =
"#version 150 compatibility                                  \n"
"#extension GL_ARB_gpu_shader5 : enable                      \n"
"                                                            \n"
"uniform sampler2D tex;                                      \n"
"                                                            \n"
"void main(void)                                             \n"
"{                                                           \n"
"    vec4 value = texture2D(tex,gl_TexCoord[0].st);          \n"
"    if(value.r < 0.25)                                      \n"
"    {                                                       \n"
"        discard;                                            \n"
"    }                                                       \n"
"    else if(value.r < 0.75)                                 \n"
"    {                                                       \n"
"        gl_FragColor = gl_Color;                            \n"
"    }                                                       \n"
"    else                                                    \n"
"    {                                                       \n"
"        gl_FragColor = vec4(0.0,0.0,0.0,1.0);               \n"
"    }                                                       \n"
"}                                                           \n";

std::string pointSizeVertSrc =
"#version 150 compatibility                                  \n"
"#extension GL_ARB_gpu_shader5 : enable                      \n"
"#extension GL_ARB_explicit_attrib_location : enable         \n"
"                                                            \n"
"layout(location = 4) in vec4 size;                          \n"
"uniform float pointSize;                                    \n"
"                                                            \n"
"void main(void)                                             \n"
"{                                                           \n"
"                                                            \n"
"    gl_FrontColor = gl_Color;                               \n"
"    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex; \n"
"    gl_PointSize = pointSize * size.r;                      \n"
"}                                                           \n";

std::string pointSizeFragSrc =
"#version 150 compatibility                                  \n"
"#extension GL_ARB_gpu_shader5 : enable                      \n"
"                                                            \n"
"void main(void)                                             \n"
"{                                                           \n"
"    gl_FragColor = gl_Color;                                \n"
"}                                                           \n";

DataGraph::DataGraph()
{
    _root = new osg::MatrixTransform();
    _clipNode = new osg::ClipNode();
    _graphTransform = new osg::MatrixTransform();
    _graphGeode = new osg::Geode();
    _axisGeode = new osg::Geode();
    _axisGeometry = new osg::Geometry();
    _bgGeometry = new osg::Geometry();
    _bgRangesGeode = new osg::Geode();
    _labelGroup = new osg::Group();

    _root->addChild(_axisGeode);
    _root->addChild(_graphTransform);
    _root->addChild(_clipNode);
    _root->addChild(_labelGroup);
    _graphTransform->addChild(_graphGeode);
    _graphTransform->addChild(_bgRangesGeode);
    _graphGeode->addDrawable(_bgGeometry);
    _axisGeode->addDrawable(_axisGeometry);

    _bgRangesGeode->setCullingActive(false);
    _clipNode->setCullingActive(false);

    _point = new osg::Point();
    _lineWidth = new osg::LineWidth();

    _glScale = 1.0;

    _pointActionPoint = new osg::Point();
    _pointActionAlpha = 0.9;
    _pointActionAlphaDir = false;

    _pointLineScale = ConfigManager::getFloat("value","Plugin.FuturePatient.PointLineScale",1.0);

    osg::StateSet * stateset = getGraphRoot()->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    stateset->setAttributeAndModes(_point,osg::StateAttribute::ON);
    _clipNode->getOrCreateStateSet()->setAttributeAndModes(_lineWidth,osg::StateAttribute::ON);

    _width = _height = 1000.0;

    _multiGraphDisplayMode = MGDM_COLOR_PT_SIZE;
    _currentMultiGraphDisplayMode = MGDM_NORMAL;
    _labelDisplayMode = LDM_MIN_MAX;

    osg::Vec4 color(1.0,1.0,1.0,1.0);

    osg::Geometry * geo = _bgGeometry.get();
    osg::Vec3Array* verts = new osg::Vec3Array();
    verts->push_back(osg::Vec3(1.0,1,1.0));
    verts->push_back(osg::Vec3(1.0,1,0));
    verts->push_back(osg::Vec3(0,1,0));
    verts->push_back(osg::Vec3(0,1,1.0));

    geo->setVertexArray(verts);

    osg::DrawElementsUInt * ele = new osg::DrawElementsUInt(
	    osg::PrimitiveSet::QUADS,0);

    ele->push_back(0);
    ele->push_back(1);
    ele->push_back(2);
    ele->push_back(3);
    geo->addPrimitiveSet(ele);

    osg::Vec4Array* colors = new osg::Vec4Array;
    colors->push_back(color);

    osg::TemplateIndexArray<unsigned int,osg::Array::UIntArrayType,4,4> *colorIndexArray;
    colorIndexArray = new osg::TemplateIndexArray<unsigned int,
		    osg::Array::UIntArrayType,4,4>;
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);

    geo->setColorArray(colors);
    geo->setColorIndices(colorIndexArray);
    geo->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    _xAxisTimestamp = false;
    _minDisplayX = 0;
    _maxDisplayX = 1.0;
    _minDisplayZ = 0;
    _maxDisplayZ = 1.0;

    _masterPointScale = ConfigManager::getFloat("value","Plugin.FuturePatient.MasterPointScale",1.0);
    _masterLineScale = ConfigManager::getFloat("value","Plugin.FuturePatient.MasterLineScale",1.0);

    //_clipNode->addClipPlane(new osg::ClipPlane(0));
    //_clipNode->addClipPlane(new osg::ClipPlane(1));
    //_clipNode->addClipPlane(new osg::ClipPlane(2));
    //_clipNode->addClipPlane(new osg::ClipPlane(3));

    _font = osgText::readFontFile(CalVR::instance()->getHomeDir() + "/resources/arial.ttf");

    setupMultiGraphDisplayModes();
    makeHover();
    makeBar();
    updateBGRanges();
}

DataGraph::~DataGraph()
{
}

void DataGraph::setDisplaySize(float width, float height)
{
    _width = width;
    _height = height;
    update();
}

void DataGraph::addGraph(std::string name, osg::Vec3Array * points, GraphDisplayType displayType, std::string xLabel, std::string zLabel, osg::Vec4 color, osg::Vec4Array * perPointColor, osg::Vec4Array * secondaryPerPointColor)
{
    if(_dataInfoMap.find(name) != _dataInfoMap.end())
    {
	std::cerr << "Error: Graph " << name << " has already been added to DataGraph." << std::endl;
	return;
    }

    //std::cerr << "Points: " << points->size() << std::endl;

    GraphDataInfo gdi;
    gdi.name = name;
    gdi.data = points;
    gdi.colorArray = perPointColor;
    gdi.secondaryColorArray = secondaryPerPointColor;
    gdi.color = color;
    gdi.displayType = GDT_NONE;
    gdi.xLabel = xLabel;
    gdi.zLabel = zLabel;
    gdi.xAxisType = LINEAR;
    gdi.zAxisType = LINEAR;
    gdi.xMin = 0.0;
    gdi.xMax = 1.0;
    gdi.zMin = 0.0;
    gdi.zMax = 1.0;

    gdi.pointGeometry = new osg::Geometry();
    gdi.pointGeometry->setUseDisplayList(false);
    gdi.pointGeometry->setUseVertexBufferObjects(true);
    
    gdi.connectorGeometry = new osg::Geometry();
    gdi.connectorGeometry->setUseDisplayList(false);
    gdi.connectorGeometry->setUseVertexBufferObjects(true);

    gdi.singleColorArray = new osg::Vec4Array(1);
    gdi.singleColorArray->at(0) = osg::Vec4(0.0,0.0,0.0,1.0);

    gdi.pointGeode = new osg::Geode();
    gdi.connectorGeode = new osg::Geode();
    gdi.labelGeode = new osg::Geode();

    _dataInfoMap[name] = gdi;

    _pointActionMap[name] = std::map<int,PointAction*>();

    gdi.pointGeode->addDrawable(gdi.pointGeometry);
    gdi.connectorGeode->addDrawable(gdi.connectorGeometry);
    gdi.pointGeode->setCullingActive(false);
    gdi.connectorGeode->setCullingActive(false);

    _graphTransformMap[name] = new osg::MatrixTransform();
    _graphTransformMap[name]->addChild(gdi.pointGeode);
    _graphTransformMap[name]->addChild(gdi.connectorGeode);
    _graphTransformMap[name]->setCullingActive(false);
    _clipNode->addChild(_graphTransformMap[name]);

    _labelGroup->addChild(gdi.labelGeode);

    setDisplayType(name, displayType);

    update();
}

void DataGraph::setXDataRangeTimestamp(std::string graphName, time_t & start, time_t & end)
{
    if(_dataInfoMap.find(graphName) != _dataInfoMap.end())
    {
	if(!_xAxisTimestamp)
	{
	    _xAxisTimestamp = true;
	    _minDisplayXT = start;
	    _maxDisplayXT = end;
	}
	_dataInfoMap[graphName].xMinT = start;
	_dataInfoMap[graphName].xMaxT = end;
	_dataInfoMap[graphName].xAxisType = TIMESTAMP;
	update();
    }
}

void DataGraph::setZDataRange(std::string graphName, float min, float max)
{
    if(_dataInfoMap.find(graphName) != _dataInfoMap.end())
    {
	_dataInfoMap[graphName].zMin = min;
	_dataInfoMap[graphName].zMax = max;
	updateAxis();
    }
}

void DataGraph::setXDisplayRange(float min, float max)
{
    _minDisplayX = min;
    _maxDisplayX = max;
    update();
}

void DataGraph::setXDisplayRangeTimestamp(time_t & start, time_t & end)
{
    _minDisplayXT = start;
    _maxDisplayXT = end;
    update();
}

void DataGraph::setZDisplayRange(float min, float max)
{
    _minDisplayZ = min;
    _maxDisplayZ = max;
    update();
}

osg::MatrixTransform * DataGraph::getGraphRoot()
{
    return _root.get();
}

void DataGraph::getGraphNameList(std::vector<std::string> & nameList)
{
    for(std::map<std::string, GraphDataInfo>::iterator it = _dataInfoMap.begin(); it != _dataInfoMap.end(); it++)
    {
	nameList.push_back(it->first);
    }
}

time_t DataGraph::getMaxTimestamp(std::string graphName)
{
    if(_dataInfoMap.find(graphName) != _dataInfoMap.end())
    {
	return _dataInfoMap[graphName].xMaxT;
    }
    return 0;
}

time_t DataGraph::getMinTimestamp(std::string graphName)
{
    if(_dataInfoMap.find(graphName) != _dataInfoMap.end())
    {
	return _dataInfoMap[graphName].xMinT;
    }
    return 0;
}

bool DataGraph::displayHoverText(osg::Matrix & mat)
{
    std::string currentHoverGraph = _hoverGraph;

    std::string selectedGraph;
    int selectedPoint = -1;
    osg::Vec3 point;
    float dist;

    for(std::map<std::string, GraphDataInfo>::iterator it = _dataInfoMap.begin(); it != _dataInfoMap.end(); it++)
    {
	if(!it->second.data->size())
	{
	    continue;
	}

	osg::Matrix invT = osg::Matrix::inverse(_graphTransformMap[it->first]->getMatrix());
	osg::Vec3 point1, point2(0,1000,0);
	
	point1 = point1 * mat * invT;
	point2 = point2 * mat * invT;
	
	osg::Vec3 planePoint, planeNormal(0,-1,0);
	float w;

	osg::Vec3 intersect;
	if(linePlaneIntersectionRef(point1,point2,planePoint,planeNormal,intersect,w))
	{
	    //std::cerr << "Intersect Point x: " << intersect.x() << " y: " << intersect.y() << " z: " << intersect.z() << std::endl;

	    if(intersect.x() < -0.1 || intersect.x() > 1.1 || intersect.z() < -0.1 || intersect.z() > 1.1)
	    {
		continue;
	    }

	    // find nearest point on x axis
	    int start, end, current;
	    start = 0;
	    end = it->second.data->size() - 1;
	    if(intersect.x() <= 0.0)
	    {
		current = start;
	    }
	    else if(intersect.x() >= 1.0)
	    {
		current = end;
	    }
	    else
	    {
		while(end - start > 1)
		{
		    current = start + ((end-start) / 2);
		    if(intersect.x() < it->second.data->at(current).x())
		    {
			end = current;
		    }
		    else
		    {
			start = current;
		    }
		}

		if(end == start)
		{
		    current = start;
		}
		else
		{
		    float startx, endx;
		    startx = it->second.data->at(start).x();
		    endx = it->second.data->at(end).x();

		    if(fabs(intersect.x() - startx) > fabs(endx - intersect.x()))
		    {
			current = end;
		    }
		    else
		    {
			current = start;
		    }
		}
	    }
	    osg::Vec3 currentPoint = it->second.data->at(current);
	    currentPoint = currentPoint * _graphTransformMap[it->first]->getMatrix();

	    intersect = intersect * _graphTransformMap[it->first]->getMatrix();
	    if(selectedPoint < 0)
	    {
		selectedPoint = current;
		selectedGraph = it->first;
		point = currentPoint;
		dist = (intersect - currentPoint).length();
	    }
	    else if((intersect - currentPoint).length() < dist)
	    {
		selectedPoint = current;
		selectedGraph = it->first;
		point = currentPoint;
		dist = (intersect - currentPoint).length();
	    }
	}
	else
	{
	    break;
	}
    }

    bool retVal = false;

    if(selectedPoint >= 0 && dist < (_width + _height) * 0.02 / 2.0)
    {
	//std::cerr << "selecting point" << std::endl;
	if(selectedGraph != _hoverGraph || selectedPoint != _hoverPoint)
	{
	    std::stringstream textss;
	    time_t time;
	    float value;
	    value = _dataInfoMap[selectedGraph].zMin + ((_dataInfoMap[selectedGraph].zMax - _dataInfoMap[selectedGraph].zMin) * _dataInfoMap[selectedGraph].data->at(selectedPoint).z());
	    time = _dataInfoMap[selectedGraph].xMinT + (time_t)((_dataInfoMap[selectedGraph].xMaxT - _dataInfoMap[selectedGraph].xMinT) * _dataInfoMap[selectedGraph].data->at(selectedPoint).x());

	    if(getNumGraphs() > 1)
	    {
		textss << _dataInfoMap[selectedGraph].name << std::endl;
	    }

	    textss << "x: " << ctime(&time) << "y: " << value << " " << _dataInfoMap[selectedGraph].zLabel;
	    _hoverText->setText(textss.str());
	    _hoverText->setCharacterSize(1.0);

	    float targetHeight = SceneManager::instance()->getTiledWallHeight() * 0.05;
	    osg::BoundingBox bb = _hoverText->getBound();
	    _hoverText->setCharacterSize(targetHeight / (bb.zMax() - bb.zMin()));

	    std::map<std::string,std::map<int,PointAction*> >::iterator it = _pointActionMap.find(selectedGraph);
	    if(it != _pointActionMap.end())
	    {
		std::map<int,PointAction*>::iterator itt;
		if((itt = it->second.find(selectedPoint)) != it->second.end())
		{
		    if(!itt->second->getActionText().empty())
		    {
			textss << std::endl << itt->second->getActionText();
			_hoverText->setText(textss.str());
		    }
		}
	    }

	    bb = _hoverText->getBound();
	    osg::Matrix bgScale;
	    bgScale.makeScale(osg::Vec3((bb.xMax() - bb.xMin()),1.0,(bb.zMax() - bb.zMin())));
	    _hoverBGScale->setMatrix(bgScale);
	}
	if(_hoverPoint < 0)
	{
	    _root->addChild(_hoverTransform);
	}

	_hoverGraph = selectedGraph;
	_hoverPoint = selectedPoint;
	osg::Matrix m;
	m.makeTranslate(point);
	_hoverTransform->setMatrix(m);

	retVal = true;
    }
    else if(_hoverPoint != -1)
    {
	_root->removeChild(_hoverTransform);
	_hoverGraph = "";
	_hoverPoint = -1;
    }

    if(_dataInfoMap.size() > 1 && currentHoverGraph != _hoverGraph)
    {
	updateAxis();
    }

    return retVal;
}

void DataGraph::clearHoverText()
{
    if(_hoverTransform->getNumParents())
    {
	_hoverTransform->getParent(0)->removeChild(_hoverTransform);
    }

    _hoverGraph = "";
    _hoverPoint = -1;
}

void DataGraph::setBarPosition(float pos)
{
    osg::Matrix m;
    m.makeTranslate(osg::Vec3(pos,0,0));
    _barPosTransform->setMatrix(m);
}

float DataGraph::getBarPosition()
{
    return _barPosTransform->getMatrix().getTrans().x();
}

void DataGraph::setBarVisible(bool b)
{
    if(b == getBarVisible())
    {
	return;
    }

    if(b)
    {
	_clipNode->addChild(_barTransform);
    }
    else
    {
	_clipNode->removeChild(_barTransform);
    }

    updateBar();
}

bool DataGraph::getBarVisible()
{
    return _barTransform->getNumParents();
}

bool DataGraph::getGraphSpacePoint(const osg::Matrix & mat, osg::Vec3 & point)
{
    float padding = calcPadding();
    float dataWidth = _width - (2.0 * padding);
    float dataHeight = _height - (2.0 * padding);

    osg::Vec3 point1,point2(0,1000.0,0),planePoint,planeNormal(0,-1,0),intersect;
    float w;
    point1 = point1 * mat;
    point2 = point2 * mat;

    if(linePlaneIntersectionRef(point1,point2,planePoint,planeNormal,intersect,w))
    {
	if(fabs(intersect.x()) > (_width/2.0) || fabs(intersect.z()) > (_height / 2.0))
	{
	    return false;
	}
	intersect.x() /= dataWidth;
	intersect.z() /= dataHeight;
	intersect.x() += 0.5;
	intersect.z() += 0.5;
	point = intersect;
    }
    else
    {
	return false;
    }

    return true;
}

void DataGraph::setBGRanges(std::vector<std::pair<float,float> > & ranges, std::vector<osg::Vec4> & colors)
{
    if(ranges.size() != colors.size())
    {
	std::cerr << "Range list and color list sizes do no match." << std::endl;
	return;
    }
    _bgRanges = ranges;
    _bgRangesColors = colors;
    updateBGRanges();
}

void DataGraph::setDisplayType(std::string graphName, GraphDisplayType displayType)
{
    if(_dataInfoMap.find(graphName) == _dataInfoMap.end())
    {
	return;
    }

    std::map<std::string, GraphDataInfo>::iterator it = _dataInfoMap.find(graphName);

    // cleanup old mode
    switch(it->second.displayType)
    {
	case GDT_NONE:
	    break;
	case GDT_POINTS:
	{
	    it->second.pointGeometry->setColorArray(NULL);
	    it->second.pointGeometry->setVertexArray(NULL);
	    it->second.pointGeometry->removePrimitiveSet(0,it->second.pointGeometry->getNumPrimitiveSets());
	    break;
	}
	case GDT_POINTS_WITH_LINES:
	{
	    it->second.pointGeometry->setColorArray(NULL);
	    it->second.pointGeometry->setVertexArray(NULL);
	    it->second.pointGeometry->removePrimitiveSet(0,it->second.pointGeometry->getNumPrimitiveSets());
	    it->second.connectorGeometry->setColorArray(NULL);
	    it->second.connectorGeometry->setVertexArray(NULL);
	    it->second.connectorGeometry->removePrimitiveSet(0,it->second.connectorGeometry->getNumPrimitiveSets());
	    break;
	}
	default:
	    break;
    }

    it->second.displayType = displayType;

     switch(displayType)
     {
	 case GDT_NONE:
	    break;
	case GDT_POINTS:
	{
	    it->second.pointGeometry->setVertexArray(it->second.data);
	    if(!it->second.colorArray || it->second.colorArray->size() != it->second.data->size())
	    {
		it->second.pointGeometry->setColorArray(it->second.singleColorArray);
		it->second.pointGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
	    }
	    else
	    {
		it->second.pointGeometry->setColorArray(it->second.colorArray);
		it->second.pointGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
	    }

	    it->second.pointGeometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS,0,it->second.data->size()));
	    break;
	}
	case GDT_POINTS_WITH_LINES:
	{
	    it->second.pointGeometry->setVertexArray(it->second.data);
	    it->second.connectorGeometry->setVertexArray(it->second.data);
	    if(!it->second.colorArray || it->second.colorArray->size() != it->second.data->size())
	    {
		it->second.pointGeometry->setColorArray(it->second.singleColorArray);
		it->second.pointGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
		it->second.connectorGeometry->setColorArray(it->second.singleColorArray);
		it->second.connectorGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
	    }
	    else
	    {
		it->second.pointGeometry->setColorArray(it->second.colorArray);
		it->second.pointGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
		it->second.connectorGeometry->setColorArray(it->second.colorArray);
		it->second.connectorGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
	    }

	    it->second.pointGeometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS,0,it->second.data->size()));
	    it->second.connectorGeometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_STRIP,0,it->second.data->size()));
	    break;
	}
	default:
	    break;
     }

     update();
}

GraphDisplayType DataGraph::getDisplayType(std::string graphName)
{
    if(_dataInfoMap.find(graphName) == _dataInfoMap.end())
    {
	return GDT_NONE;
    }

    return _dataInfoMap[graphName].displayType;
}

void DataGraph::setLabelDisplayMode(LabelDisplayMode ldm)
{
    for(std::map<std::string,GraphDataInfo>::iterator it = _dataInfoMap.begin(); it != _dataInfoMap.end(); ++it)
    {
	it->second.labelGeode->removeDrawables(0,it->second.labelGeode->getNumDrawables());

	osg::Vec4 textColor(0.1,0.1,0.1,1.0);

	switch(ldm)
	{
	    case LDM_NONE:
		break;
	    case LDM_MIN_MAX:
	    {
		float min = FLT_MAX;
		float max = FLT_MIN;
		int minIndex = -1, maxIndex = -1;
		//find min/max value/index
		for(int i = 0; i < it->second.data->size(); ++i)
		{
		    if(it->second.data->at(i).z() > max)
		    {
			max = it->second.data->at(i).z();
			maxIndex = i;
		    }
		    if(it->second.data->at(i).z() < min)
		    {
			min = it->second.data->at(i).z();
			minIndex = i;
		    }
		}

		if(minIndex < 0 || maxIndex < 0)
		{
		    break;
		}

		osg::Vec3 minPoint = it->second.data->at(minIndex) * _graphTransformMap[it->first]->getMatrix();
		osg::Vec3 maxPoint = it->second.data->at(maxIndex) * _graphTransformMap[it->first]->getMatrix();

		float textHeight = ((_width + _height) / 2.0) * 0.02;
		maxPoint = maxPoint - osg::Vec3(0,0,textHeight) + osg::Vec3(0,-1,0);
		minPoint = minPoint + osg::Vec3(0,0,textHeight) + osg::Vec3(0,-1,0);

		std::stringstream minss;
		minss << (it->second.zMin + (it->second.data->at(minIndex).z() * (it->second.zMax-it->second.zMin)));
		osgText::Text * text = makeText(minss.str(),textColor);
		text->setAlignment(osgText::Text::CENTER_CENTER);
		osg::BoundingBox bb = text->getBound();
		float csize = textHeight / (bb.zMax() - bb.zMin());
		text->setCharacterSize(csize);
		text->setPosition(minPoint);
		it->second.labelGeode->addDrawable(text);

		std::stringstream maxss;
		maxss << (it->second.zMin + (it->second.data->at(maxIndex).z() * (it->second.zMax-it->second.zMin)));
		text = makeText(maxss.str(),textColor);
		text->setAlignment(osgText::Text::CENTER_CENTER);
		bb = text->getBound();
		csize = textHeight / (bb.zMax() - bb.zMin());
		text->setCharacterSize(csize);
		text->setPosition(maxPoint);
		it->second.labelGeode->addDrawable(text);

		break;
	    }
	    case LDM_ALL:
	    {
		for(int i = 0; i < it->second.data->size(); ++i)
		{
		    osg::Vec3 point = it->second.data->at(i) * _graphTransformMap[it->first]->getMatrix();
		    float textHeight = ((_width + _height) / 2.0) * 0.01;

		    if(fabs(point.z() - 1.5 * textHeight) > (_height/2.0) - calcPadding())
		    {
			point = point + osg::Vec3(0,0,textHeight) + osg::Vec3(0,-1,0);
		    }
		    else
		    {
			point = point - osg::Vec3(0,0,textHeight) + osg::Vec3(0,-1,0);
		    }

		    std::stringstream ss;
		    ss << (it->second.zMin + (it->second.data->at(i).z() * (it->second.zMax-it->second.zMin)));
		    osgText::Text * text = makeText(ss.str(),textColor);
		    text->setAlignment(osgText::Text::CENTER_CENTER);
		    osg::BoundingBox bb = text->getBound();
		    float csize = textHeight / (bb.zMax() - bb.zMin());
		    text->setCharacterSize(csize);
		    text->setPosition(point);
		    it->second.labelGeode->addDrawable(text);
		}

		break;
	    }
	    default:
		break;
	}
    }

    _labelDisplayMode = ldm;
}

void DataGraph::setGLScale(float scale)
{
    _glScale = scale;
    update();
}

void DataGraph::setPointActions(std::string graphname, std::map<int,PointAction*> & actionMap)
{
    if(_pointActionMap.find(graphname) == _pointActionMap.end())
    {
	return;
    }

    _pointActionMap[graphname] = actionMap;

    std::map<std::string, GraphDataInfo>::iterator it;
    if((it = _dataInfoMap.find(graphname)) != _dataInfoMap.end())
    {
	if(it->second.pointActionGeode)
	{
	    _graphTransformMap[graphname]->removeChild(it->second.pointActionGeode);
	    it->second.pointActionGeode->removeDrawables(0,it->second.pointActionGeode->getNumDrawables());
	}
	else
	{
	    it->second.pointActionGeode = new osg::Geode();
	    osg::StateSet * stateset = it->second.pointActionGeode->getOrCreateStateSet();
	    stateset->setMode(GL_BLEND,osg::StateAttribute::ON);
	    stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
	    stateset->setAttributeAndModes(_pointActionPoint,osg::StateAttribute::ON);
	}

	it->second.pointActionGeometry = new osg::Geometry();
	it->second.pointActionGeometry->setUseDisplayList(false);
	it->second.pointActionGeode->addDrawable(it->second.pointActionGeometry);
	it->second.pointActionGeode->setCullingActive(false);

	osg::Vec3Array * verts = new osg::Vec3Array(actionMap.size());
	osg::Vec4Array * colors = new osg::Vec4Array(1);
	colors->at(0) = osg::Vec4(1.0,0,0,_pointActionAlpha);
	it->second.pointActionGeometry->setVertexArray(verts);
	it->second.pointActionGeometry->setColorArray(colors);
	it->second.pointActionGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
	it->second.pointActionGeometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS,0,actionMap.size()));

	int count = 0;
	for(std::map<int,PointAction*>::iterator pit = actionMap.begin(); pit != actionMap.end(); ++pit, ++count)
	{
	    verts->at(count) = it->second.data->at(pit->first);
	    verts->at(count).y() -= 0.5;
	}

	_graphTransformMap[graphname]->addChild(it->second.pointActionGeode);

	update();
    }

}

void DataGraph::updatePointAction()
{
    static const float flashingTime = 2.5;
    
    float deltaAlpha = PluginHelper::getLastFrameDuration() / flashingTime;
    if(!_pointActionAlphaDir)
    {
	deltaAlpha *= -1.0;
    }

    _pointActionAlpha += deltaAlpha;
    if(_pointActionAlpha < 0.0)
    {
	_pointActionAlpha = 0.0;
	_pointActionAlphaDir = true;
    }
    else if(_pointActionAlpha > 0.9)
    {
	_pointActionAlpha = 0.9;
	_pointActionAlphaDir = false;
    }

    for(std::map<std::string, GraphDataInfo>::iterator it = _dataInfoMap.begin(); it != _dataInfoMap.end(); ++it)
    {
	if(it->second.pointActionGeometry)
	{
	    osg::Vec4Array * colors = dynamic_cast<osg::Vec4Array*>(it->second.pointActionGeometry->getColorArray());
	    if(colors && colors->size())
	    {
		colors->at(0).w() = _pointActionAlpha;
	    }
	}
    }
}

bool DataGraph::pointClick()
{
    if(_pointActionMap.find(_hoverGraph) != _pointActionMap.end() && _pointActionMap[_hoverGraph].find(_hoverPoint) != _pointActionMap[_hoverGraph].end())
    {
	_pointActionMap[_hoverGraph][_hoverPoint]->action();
	return true;
    }

    return false;
}

void DataGraph::setupMultiGraphDisplayModes()
{
    //shape setup
    _shapeProgram = new osg::Program();
    _shapeProgram->setName("Shape Shader");
    _shapeProgram->addShader(new osg::Shader(osg::Shader::VERTEX,shapeVertSrc));
    _shapeProgram->addShader(new osg::Shader(osg::Shader::FRAGMENT,shapeFragSrc));

    _shapePointSprite = new osg::PointSprite();
    _shapeDepth = new osg::Depth();
    _shapeDepth->setWriteMask(false);

    //point size setup
    _sizeProgram = new osg::Program();
    _sizeProgram->setName("Point Size Shader");
    _sizeProgram->addShader(new osg::Shader(osg::Shader::VERTEX,pointSizeVertSrc));
    _sizeProgram->addShader(new osg::Shader(osg::Shader::FRAGMENT,pointSizeFragSrc));

    _pointSizeUniform = new osg::Uniform(osg::Uniform::FLOAT,"pointSize");
    _pointSizeUniform->set(1.0f);
}

void DataGraph::makeHover()
{
    _hoverTransform = new osg::MatrixTransform();
    _hoverBGScale = new osg::MatrixTransform();
    _hoverBGGeode = new osg::Geode();
    _hoverTextGeode = new osg::Geode();

    _hoverTransform->addChild(_hoverBGScale);
    _hoverTransform->addChild(_hoverTextGeode);
    _hoverBGScale->addChild(_hoverBGGeode);

    osg::Vec4 color(0.0,0.0,0.0,1.0);

    osg::Geometry * geo = new osg::Geometry();
    osg::Vec3Array* verts = new osg::Vec3Array();
    verts->push_back(osg::Vec3(1.0,-3,0));
    verts->push_back(osg::Vec3(1.0,-3,-1.0));
    verts->push_back(osg::Vec3(0,-3,-1.0));
    verts->push_back(osg::Vec3(0,-3,0));

    geo->setVertexArray(verts);

    osg::DrawElementsUInt * ele = new osg::DrawElementsUInt(
	    osg::PrimitiveSet::QUADS,0);

    ele->push_back(0);
    ele->push_back(1);
    ele->push_back(2);
    ele->push_back(3);
    geo->addPrimitiveSet(ele);

    osg::Vec4Array* colors = new osg::Vec4Array;
    colors->push_back(color);

    osg::TemplateIndexArray<unsigned int,osg::Array::UIntArrayType,4,4> *colorIndexArray;
    colorIndexArray = new osg::TemplateIndexArray<unsigned int,
		    osg::Array::UIntArrayType,4,4>;
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);

    geo->setColorArray(colors);
    geo->setColorIndices(colorIndexArray);
    geo->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    _hoverBGGeode->addDrawable(geo);

    _hoverText = makeText("",osg::Vec4(1.0,1.0,1.0,1.0));
    _hoverTextGeode->addDrawable(_hoverText);
    _hoverText->setAlignment(osgText::Text::LEFT_TOP);
    osg::Vec3 pos(0,-4,0);
    _hoverText->setPosition(pos);
}

void DataGraph::makeBar()
{
    _barTransform = new osg::MatrixTransform();
    _barPosTransform = new osg::MatrixTransform();
    _barGeode = new osg::Geode();
    _barGeometry = new osg::Geometry();
    _barTransform->addChild(_barPosTransform);
    _barPosTransform->addChild(_barGeode);
    _barGeode->addDrawable(_barGeometry);

    osg::Geometry * geo = _barGeometry.get();
    osg::Vec3Array* verts = new osg::Vec3Array();
    verts->push_back(osg::Vec3(0,-0.2,0));
    verts->push_back(osg::Vec3(0,-0.2,1.0));

    geo->setVertexArray(verts);

    osg::DrawElementsUInt * ele = new osg::DrawElementsUInt(
	    osg::PrimitiveSet::LINES,0);

    ele->push_back(0);
    ele->push_back(1);
    geo->addPrimitiveSet(ele);

    osg::Vec4Array* colors = new osg::Vec4Array;
    colors->push_back(osg::Vec4(1.0,1.0,0,1.0));

    osg::TemplateIndexArray<unsigned int,osg::Array::UIntArrayType,4,4> *colorIndexArray;
    colorIndexArray = new osg::TemplateIndexArray<unsigned int,
		    osg::Array::UIntArrayType,4,4>;
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);

    geo->setColorArray(colors);
    geo->setColorIndices(colorIndexArray);
    geo->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
}

void DataGraph::update()
{
    float padding = calcPadding();
    float dataWidth = _width - (2.0 * padding);
    float dataHeight = _height - (2.0 * padding);

    //std::cerr << "Update mindispXT: " << _minDisplayXT << " maxdispXT: " << _maxDisplayXT << std::endl;

    osg::Matrix tran,scale;
    for(std::map<std::string, GraphDataInfo>::iterator it = _dataInfoMap.begin(); it != _dataInfoMap.end(); it++)
    {
	float myRangeSize;
	float myRangeCenter;

	if(it->second.xAxisType == TIMESTAMP)
	{
	    time_t range = it->second.xMaxT - it->second.xMinT;
	    time_t totalrange = _maxDisplayXT - _minDisplayXT;
	    myRangeSize = (float)((double)range / (double)totalrange);

	    time_t center = it->second.xMinT + (range / ((time_t)2.0));
	    myRangeCenter = (center - _minDisplayXT) / (double)totalrange;
	}
	else
	{
	    float range = it->second.xMax - it->second.xMin;
	    float totalrange = _maxDisplayX - _minDisplayX;
	    myRangeSize = range / totalrange;

	    float center = it->second.xMin + (range / 2.0);
	    myRangeCenter = (center - _minDisplayX) / totalrange;
	}

	float minxBound = ((0.5 * myRangeSize) - myRangeCenter) / myRangeSize;
	float maxxBound = ((0.5 * myRangeSize) + (1.0 - myRangeCenter)) / myRangeSize;

	//std::cerr << "x bounds min: " << minxBound << " max: " << maxxBound << std::endl;

	//std::cerr << "TotalPoint: " << _dataInfoMap[it->first].data->size() << std::endl;

	// TODO: redo this with binary searches
	int maxpoint = -1, minpoint = -1;
	for(int j = 0; j < _dataInfoMap[it->first].data->size(); j++)
	{
	    if(_dataInfoMap[it->first].data->at(j).x() >= minxBound)
	    {
		minpoint = j;
		break;
	    }
	}

	for(int j = _dataInfoMap[it->first].data->size() - 1; j >= 0; j--)
	{
	    if(_dataInfoMap[it->first].data->at(j).x() <= maxxBound)
	    {
		maxpoint = j;
		break;
	    }
	}

	//std::cerr << "Minpoint: " << minpoint << " Maxpoint: " << maxpoint << std::endl;

	//TODO maybe move this into a subset function call, so there can be different actions based on the display type
	for(int i = 0; i < _dataInfoMap[it->first].pointGeometry->getNumPrimitiveSets(); i++)
	{
	    osg::DrawArrays * da = dynamic_cast<osg::DrawArrays*>(_dataInfoMap[it->first].pointGeometry->getPrimitiveSet(i));
	    if(!da)
	    {
		continue;
	    }

	    if(maxpoint == -1 || minpoint == -1)
	    {
		da->setCount(0);
	    }
	    else
	    {
		da->setFirst(minpoint);
		da->setCount((maxpoint-minpoint)+1);
	    }
	}

	for(int i = 0; i < _dataInfoMap[it->first].connectorGeometry->getNumPrimitiveSets(); i++)
	{
	    osg::DrawArrays * da = dynamic_cast<osg::DrawArrays*>(_dataInfoMap[it->first].connectorGeometry->getPrimitiveSet(i));
	    if(!da)
	    {
		continue;
	    }

	    if(maxpoint == -1 || minpoint == -1)
	    {
		da->setCount(0);
	    }
	    else
	    {
		da->setFirst(minpoint);
		da->setCount((maxpoint-minpoint)+1);
	    }
	}


	//std::cerr << "My range size: " << myRangeSize << " range center: " << myRangeCenter << std::endl;

	osg::Matrix centerm;
	centerm.makeTranslate(osg::Vec3((myRangeCenter - 0.5) * dataWidth,0,0));
	tran.makeTranslate(osg::Vec3(-0.5,0,-0.5));
	scale.makeScale(osg::Vec3(dataWidth*myRangeSize,1.0,dataHeight));
	_graphTransformMap[it->second.name]->setMatrix(tran*scale*centerm);
    }

    if(_dataInfoMap.size() > 1)
    {
	// need this to run when a new graph is added, maybe break into function call
	//if(_multiGraphDisplayMode != _currentMultiGraphDisplayMode)
	{
	    int count = 0;
	    for(std::map<std::string, GraphDataInfo>::iterator it = _dataInfoMap.begin(); it != _dataInfoMap.end(); it++)
	    {
		// revert old mode
		switch(_currentMultiGraphDisplayMode)
		{
		    case MGDM_NORMAL:
			{
			    it->second.connectorGeometry->setColorArray(it->second.colorArray);
			    it->second.connectorGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
			    break;
			}
		    case MGDM_COLOR:
			{
			    it->second.connectorGeometry->setColorArray(it->second.colorArray);
			    it->second.connectorGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
			    break;
			}
		    case MGDM_COLOR_SOLID:
			{
			    it->second.connectorGeometry->setColorArray(it->second.colorArray);
			    it->second.connectorGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
			    it->second.pointGeometry->setColorArray(it->second.colorArray);
			    it->second.pointGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
			    break;
			}
		    case MGDM_COLOR_PT_SIZE:
			{
			    if(!it->second.secondaryColorArray)
			    {
				break;
			    }
			    it->second.connectorGeometry->setColorArray(it->second.colorArray);
			    it->second.connectorGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
			    it->second.pointGeometry->setColorArray(it->second.colorArray);
			    it->second.pointGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

			    it->second.pointGeometry->setVertexAttribArray(4,NULL);
			    osg::StateSet * stateset = it->second.pointGeode->getOrCreateStateSet();
			    stateset->removeAttribute(_sizeProgram);
			    stateset->removeUniform(_pointSizeUniform);
			    stateset->removeMode(GL_VERTEX_PROGRAM_POINT_SIZE);
			    break;
			}
		    case MGDM_SHAPE:
			{
			    osg::StateSet * stateset = it->second.pointGeode->getOrCreateStateSet();
			    stateset->removeTextureAttribute(0,osg::StateAttribute::POINTSPRITE);
			    stateset->removeTextureAttribute(0,osg::StateAttribute::TEXTURE);
			    stateset->removeAttribute(_shapeProgram);
			    it->second.connectorGeode->getOrCreateStateSet()->removeAttribute(_shapeDepth);
			    break;
			}
		    case MGDM_COLOR_SHAPE:
			{
			    it->second.connectorGeometry->setColorArray(it->second.colorArray);
			    it->second.connectorGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
			    osg::StateSet * stateset = it->second.pointGeode->getOrCreateStateSet();
			    stateset->removeTextureAttribute(0,osg::StateAttribute::POINTSPRITE);
			    stateset->removeTextureAttribute(0,osg::StateAttribute::TEXTURE);
			    stateset->removeAttribute(_shapeProgram);
			    it->second.connectorGeode->getOrCreateStateSet()->removeAttribute(_shapeDepth);
			    break;
			}
		    default:
			break;
		}
		count++;
	    }

	    count = 0;
	    for(std::map<std::string, GraphDataInfo>::iterator it = _dataInfoMap.begin(); it != _dataInfoMap.end(); it++)
	    {
		switch(_multiGraphDisplayMode)
		{
		    case MGDM_NORMAL:
			{
			    break;
			}
		    case MGDM_COLOR:
			{
			    //float f = ((float)count) / ((float)_dataInfoMap.size());
			    //osg::Vec4 color = makeColor(f);
			    osg::Vec4 color = ColorGenerator::makeColor(count, _dataInfoMap.size());
			    it->second.singleColorArray->at(0) = color;
			    it->second.connectorGeometry->setColorArray(it->second.singleColorArray);
			    it->second.connectorGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
			    break;
			}
		    case MGDM_COLOR_SOLID:
			{
			    osg::Vec4 color = ColorGenerator::makeColor(count, _dataInfoMap.size());
			    it->second.singleColorArray->at(0) = color;
			    it->second.connectorGeometry->setColorArray(it->second.singleColorArray);
			    it->second.connectorGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
			    it->second.pointGeometry->setColorArray(it->second.singleColorArray);
			    it->second.pointGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
			    break;
			}
		    case MGDM_COLOR_PT_SIZE:
			{
			    if(!it->second.secondaryColorArray)
			    {
				break;
			    }
			    osg::Vec4 color = ColorGenerator::makeColor(count, _dataInfoMap.size());
			    it->second.singleColorArray->at(0) = color;
			    it->second.connectorGeometry->setColorArray(it->second.singleColorArray);
			    it->second.connectorGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
			    it->second.pointGeometry->setColorArray(it->second.singleColorArray);
			    it->second.pointGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);

			    it->second.pointGeometry->setVertexAttribArray(4,it->second.secondaryColorArray);
			    it->second.pointGeometry->setVertexAttribBinding(4,osg::Geometry::BIND_PER_VERTEX);
			    osg::StateSet * stateset = it->second.pointGeode->getOrCreateStateSet();
			    stateset->setAttribute(_sizeProgram);
			    stateset->addUniform(_pointSizeUniform);
			    stateset->setMode(GL_VERTEX_PROGRAM_POINT_SIZE, osg::StateAttribute::ON);
			    
			    break;
			}
		    case MGDM_SHAPE:
			{
			    osg::StateSet * stateset = it->second.pointGeode->getOrCreateStateSet();
			    stateset->setTextureAttributeAndModes(0, _shapePointSprite, osg::StateAttribute::ON);
			    stateset->setTextureAttributeAndModes(0, ShapeTextureGenerator::getOrCreateShapeTexture(count+3,128,128), osg::StateAttribute::ON);
			    stateset->setAttribute(_shapeProgram);
			    it->second.connectorGeode->getOrCreateStateSet()->setAttributeAndModes(_shapeDepth,osg::StateAttribute::ON);
			    break;
			}
		    case MGDM_COLOR_SHAPE:
			{
			    //float f = ((float)count) / ((float)_dataInfoMap.size());
			    //osg::Vec4 color = makeColor(f);
			    osg::Vec4 color = ColorGenerator::makeColor(count, _dataInfoMap.size());
			    it->second.singleColorArray->at(0) = color;
			    it->second.connectorGeometry->setColorArray(it->second.singleColorArray);
			    it->second.connectorGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);

			    osg::StateSet * stateset = it->second.pointGeode->getOrCreateStateSet();
			    stateset->setTextureAttributeAndModes(0, _shapePointSprite, osg::StateAttribute::ON);
			    stateset->setTextureAttributeAndModes(0, ShapeTextureGenerator::getOrCreateShapeTexture(count+3,128,128), osg::StateAttribute::ON);
			    stateset->setAttribute(_shapeProgram);
			    it->second.connectorGeode->getOrCreateStateSet()->setAttributeAndModes(_shapeDepth,osg::StateAttribute::ON);
			    break;
			}
		    default:
			break;
		}
		count++;
	    }

	    _currentMultiGraphDisplayMode = _multiGraphDisplayMode;
	}
    }
    else
    {
	for(std::map<std::string, GraphDataInfo>::iterator it = _dataInfoMap.begin(); it != _dataInfoMap.end(); it++)
	{
	    it->second.singleColorArray->at(0) = osg::Vec4(0.21569,0.49412,0.72157,1.0);
	    it->second.connectorGeometry->setColorArray(it->second.singleColorArray);
	    it->second.connectorGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
	}
    }

    tran.makeTranslate(osg::Vec3(-0.5,0,-0.5));
    scale.makeScale(osg::Vec3(_width,1.0,_height));
    _graphTransform->setMatrix(tran*scale);

    float avglen = (_width + _height) / 2.0;
    _point->setSize(_glScale * avglen * 0.04 * _pointLineScale);
    _pointSizeUniform->set((float)_point->getSize());
    _pointActionPoint->setSize(1.3*_point->getSize());
    //std::cerr << "Point size set to: " << _point->getSize() << std::endl;
    _lineWidth->setWidth(_glScale * avglen * 0.05 * _pointLineScale * _pointLineScale);

    // vroom tile node workaround, who know why
    float tempval;
    _pointSizeUniform->get(tempval);
    _pointSizeUniform->set(tempval*0.31f);

    if(ComController::instance()->isMaster())
    {
	_point->setSize(_point->getSize() * _masterPointScale);
	_pointSizeUniform->set((float)_point->getSize());
	_pointActionPoint->setSize(1.3*_point->getSize());
	_lineWidth->setWidth(_lineWidth->getWidth() * _masterLineScale);
    }

    updateAxis();
    updateBar();
    //updateClip();
    updateBGRanges();
    setLabelDisplayMode(_labelDisplayMode);
}

void DataGraph::updateAxis()
{
    _axisGeode->removeDrawables(0,_axisGeode->getNumDrawables());

    if(!_dataInfoMap.size())
    {
	return;
    }

    float padding = calcPadding();
    float dataWidth = _width - (2.0 * padding);
    float dataHeight = _height - (2.0 * padding);
    float dataXorigin = -(dataWidth / 2.0);
    float dataZorigin = -(dataHeight / 2.0);

    for(int i = 0; i < 2; i++)
    {
	osg::Vec3 startPoint;
	osg::Vec3 dir, tickDir;
	float totalLength;
	float startVal;

	AxisType axisType;
	float minValue, maxValue;
	time_t minTime, maxTime;
	float tickLength = _width * 0.01;	

	osgText::Text::AxisAlignment axisAlign;
	osg::Quat q;

	std::string axisLabel;
	osg::Vec4 textColor;

	// x axis
	if(i == 0)
	{
	    startPoint = osg::Vec3(dataXorigin,0,dataZorigin);
	    startVal = dataXorigin;
	    dir = osg::Vec3(1.0,0,0);
	    tickDir = osg::Vec3(0,0,1.0);
	    totalLength = dataWidth;
	    minValue = _minDisplayX;
	    maxValue = _maxDisplayX;
	    minTime = _minDisplayXT;
	    maxTime = _maxDisplayXT;
	    axisAlign = osgText::Text::XZ_PLANE;
	    textColor = osg::Vec4(0.0,0.0,0.0,1.0);
	    axisLabel = _dataInfoMap.begin()->second.xLabel;
	    axisType = _dataInfoMap.begin()->second.xAxisType;
	    //std::cerr << "MinTime: " << std::string(ctime(&minTime)) << " MaxTime: " << std::string(ctime(&maxTime)) << std::endl;
	}
	// z axis
	else if(i == 1)
	{
	    startPoint = osg::Vec3(dataXorigin,0,dataZorigin);
	    startVal = dataZorigin;
	    dir = osg::Vec3(0,0,1.0);
	    tickDir = osg::Vec3(1.0,0,0);
	    totalLength = dataHeight;
	    axisAlign = osgText::Text::USER_DEFINED_ROTATION;
	    q.makeRotate(M_PI/2.0,osg::Vec3(1.0,0,0));
	    q = q * osg::Quat(-M_PI/2.0,osg::Vec3(0,1.0,0));

	    if(_dataInfoMap.size() == 1)
	    {
		float max = _dataInfoMap.begin()->second.zMax;
		float min = _dataInfoMap.begin()->second.zMin;
		float dataRange = max - min;
		minValue = min + (_minDisplayZ * dataRange);
		maxValue = min + (_maxDisplayZ * dataRange);
		textColor = osg::Vec4(0.0,0.0,0.0,1.0);
		axisLabel = _dataInfoMap.begin()->second.zLabel;
		axisType = _dataInfoMap.begin()->second.zAxisType;
	    }
	    else
	    {
		bool useNormalized = true;

		bool validHover = false;
		int graphCount = 0;
		std::map<std::string,GraphDataInfo>::iterator it;

		for(it = _dataInfoMap.begin(); it != _dataInfoMap.end(); it++)
		{
		    if(it->first == _hoverGraph)
		    {
			validHover = true;
			break;
		    }
		    graphCount++;
		}

		if(validHover)
		{
		    switch(_currentMultiGraphDisplayMode)
		    {
			case MGDM_COLOR:
			case MGDM_COLOR_SOLID:
			case MGDM_COLOR_PT_SIZE:
			case MGDM_COLOR_SHAPE:
			{
			    float max = it->second.zMax;
			    float min = it->second.zMin;
			    float dataRange = max - min;
			    minValue = min + (_minDisplayZ * dataRange);
			    maxValue = min + (_maxDisplayZ * dataRange);
			    textColor = it->second.singleColorArray->at(0);
			    axisLabel = it->second.zLabel;
			    axisType = it->second.zAxisType;
			    useNormalized = false;
			    break;
			}
			default:
			    break;
		    }
		}

		if(useNormalized)
		{
		    minValue = _minDisplayZ;
		    maxValue = _maxDisplayZ;
		    textColor = osg::Vec4(0.0,0.0,0.0,1.0);
		    axisLabel = "Normalized Value";
		    axisType = LINEAR;
		}
	    }
	}

	osg::Geometry * geometry = new osg::Geometry();
	geometry->setUseDisplayList(false);
	geometry->setUseVertexBufferObjects(true);

	osg::Vec3Array * points = new osg::Vec3Array();
	geometry->setVertexArray(points);

	osg::Vec4 color(0.0,0.0,0.0,1.0);

	osg::Vec4Array * colorArray = new osg::Vec4Array(1);
	colorArray->at(0) = color;
	geometry->setColorArray(colorArray);
	geometry->setColorBinding(osg::Geometry::BIND_OVERALL);

	points->push_back(startPoint);
	points->push_back(startPoint + dir * totalLength);

	switch(axisType)
	{
	    case TIMESTAMP:
	    {
		enum markInterval
		{
		    YEAR,
		    MONTH,
		    DAY,
		    HOUR,
		    MINUTE,
		    SECOND
		};

		std::stringstream lowerTextss;

		markInterval mi;
		int intervalMult = 1;

		double totalTime = difftime(maxTime,minTime);

		if(totalTime < 20.0)
		{
		    mi = SECOND;
		}
		else
		{
		    totalTime /= 60.0;
		    if(totalTime < 20.0)
		    {
			mi = MINUTE;
		    }
		    else
		    {
			totalTime /= 60.0;
			if(totalTime < 20.0)
			{
			    mi = HOUR;
			}
			else
			{
			    totalTime /= 24.0;
			    if(totalTime < 20.0)
			    {
				mi = DAY;
			    }
			    else
			    {
				totalTime /= 30.0;
				if(totalTime < 20.0)
				{
				    mi = MONTH;
				    if(totalTime > 10.0)
				    {
					//intervalMult = 2;
				    }
				}
				else
				{
				    mi = YEAR;
				    if(totalTime / 12.0 > 10)
				    {
					intervalMult = 2;
				    }
				    //std::cerr << "Setting tick value to YEAR, totalTime: " << totalTime / 12.0 << std::endl;
				}
			    }
			}
		    }
		}

		struct tm starttm, endtm;
		endtm = *localtime(&maxTime);
		starttm = *localtime(&minTime);

		switch(mi)
		{
		    case YEAR:
		    {
			break;
		    }
		    case MONTH:
		    {
			/*char tempC[1024];
			strftime(tempC,1023,"%Y",&starttm);
			lowerTextss << tempC;
			if(starttm.tm_year != endtm.tm_year)
			{
			    strftime(tempC,1023,"%Y",&endtm);
			    lowerTextss << " - " << tempC;
			}*/
			break;
		    }
		    case DAY:
		    {
			/*char tempC[1024];
			strftime(tempC,1023,"%b %Y",&starttm);
			lowerTextss << tempC;
			if(starttm.tm_mon != endtm.tm_mon)
			{
			    strftime(tempC,1023,"%b %Y",&endtm);
			    lowerTextss << " - " << tempC;
			}*/
			break;
		    }
		    case HOUR:
		    {
			char tempC[1024];
			strftime(tempC,1023,"%b %d, %Y",&starttm);
			lowerTextss << tempC;
			if(starttm.tm_mday != endtm.tm_mday)
			{
			    strftime(tempC,1023,"%b %d, %Y",&endtm);
			    lowerTextss << " - " << tempC;
			}
			break;
		    }
		    case MINUTE:
		    {
			char tempC[1024];
			strftime(tempC,1023,"%b %d, %Y",&starttm);
			lowerTextss << tempC;
			if(starttm.tm_hour != endtm.tm_hour)
			{
			    strftime(tempC,1023,"%b %d, %Y",&endtm);
			    lowerTextss << " - " << tempC;
			}
			break;	
		    }
		    case SECOND:
		    {
			char tempC[1024];
			strftime(tempC,1023,"%b %d, %Y",&starttm);
			lowerTextss << tempC;
			if(starttm.tm_min != endtm.tm_min)
			{
			    strftime(tempC,1023,"%b %d, %Y",&endtm);
			    lowerTextss << " - " << tempC;
			}
			break;
		    }
		    default:
			break;
		}


		struct tm currentStep;
		double currentValue;

		currentStep = *localtime(&minTime);
		switch(mi)
		{
		    case YEAR:
		    {
			currentStep.tm_isdst = 0;
			currentStep.tm_sec = currentStep.tm_min = currentStep.tm_hour = currentStep.tm_mon = 0;
			currentStep.tm_mday = 1;
			currentStep.tm_year++;

			currentValue = difftime(mktime(&currentStep),minTime);
			currentValue /= difftime(maxTime,minTime);
			currentValue *= totalLength;

			break;
		    }
		    case MONTH:
		    {
			currentStep.tm_isdst = 0;
			currentStep.tm_sec = currentStep.tm_min = currentStep.tm_hour = 0;
			currentStep.tm_mday = 1;
			
			currentStep.tm_mon += intervalMult;
			/*while(currentStep.tm_mon >= 12)
			{
			    currentStep.tm_year++;
			    currentStep.tm_mon -= 12;
			}*/

			currentValue = difftime(mktime(&currentStep),minTime);
			currentValue /= difftime(maxTime,minTime);
			currentValue *= totalLength;
			break;
		    }
		    case DAY:
		    {
			currentStep.tm_isdst = 0;
			currentStep.tm_sec = currentStep.tm_min = currentStep.tm_hour = 0;
			currentStep.tm_mday++;
			// needed for some reason or the tick is off
			//currentStep.tm_hour--;

			currentValue = difftime(mktime(&currentStep),minTime);
			currentValue /= difftime(maxTime,minTime);
			currentValue *= totalLength;
			break;
		    }
		    case HOUR:
		    {
			//std::cerr << "Min time: " << asctime(&currentStep) << std::endl;;
			currentStep.tm_sec = currentStep.tm_min = 0;
			currentStep.tm_hour++;

			//std::cerr << "Current time: " << asctime(&currentStep) << std::endl;

			currentValue = difftime(mktime(&currentStep),minTime);
			currentValue /= difftime(maxTime,minTime);
			//std::cerr << "currentVal ratio: " << currentValue << std::endl;
			currentValue *= totalLength;

			break;
		    }
		    case MINUTE:
		    {
			currentStep.tm_sec = 0;
			currentStep.tm_min++;

			currentValue = difftime(mktime(&currentStep),minTime);
			currentValue /= difftime(maxTime,minTime);
			currentValue *= totalLength;
			break;
		    }
		    case SECOND:
		    {
			currentStep.tm_sec++;

			currentValue = difftime(mktime(&currentStep),minTime);
			currentValue /= difftime(maxTime,minTime);
			currentValue *= totalLength;
			break;
		    }
		    default:
			currentValue = totalLength + 1.0;
			std::cerr << "Unimplemented timestamp tick case." << std::endl;
			break;
		}

		while(currentValue < totalLength)
		{
		    points->push_back(startPoint + dir * currentValue);
		    points->push_back(startPoint + tickDir * tickLength + dir * currentValue);

		    std::stringstream ss;

		    switch(mi)
		    {
			case YEAR:
			{
			    char tlabel[256];
			    strftime(tlabel,255,"%Y",&currentStep);
			    ss << tlabel;
			    break;
			}
			case MONTH:
			{
			    char tlabel[256];
			    strftime(tlabel,255,"%m/%y",&currentStep);
			    ss << tlabel;
			    break;
			}
			case DAY:
			{
			    char tlabel[256];
			    strftime(tlabel,255,"%m/%d/%y",&currentStep);
			    ss << tlabel;
			    break;
			}
			case HOUR:
			{
			    char tlabel[256];
			    strftime(tlabel,255,"%H:00",&currentStep);
			    ss << tlabel;
			    break;
			}
			case MINUTE:
			{
			    char tlabel[256];
			    strftime(tlabel,255,"%H:%M",&currentStep);
			    ss << tlabel;
			    break;
			}
			case SECOND:
			{
			    char tlabel[256];
			    strftime(tlabel,255,"%H:%M:%S",&currentStep);
			    ss << tlabel;
			    break;
			}
			default:
			    break;
		    }

		    osgText::Text * text = makeText(ss.str(),textColor);

		    float targetSize = padding * 0.27;
		    osg::BoundingBox bb = text->getBound();
		    text->setCharacterSize(targetSize / (bb.zMax() - bb.zMin()));
		    text->setAxisAlignment(axisAlign);
		    if(axisAlign == osgText::Text::USER_DEFINED_ROTATION)
		    {
			text->setRotation(q);
		    }

		    text->setPosition(startPoint + -tickDir * (padding * 0.5 * 0.3) + dir * currentValue + osg::Vec3(0,-1,0));

		    _axisGeode->addDrawable(text);

		    switch(mi)
		    {
			case YEAR:
			    currentStep.tm_year++;
			    currentValue = difftime(mktime(&currentStep),minTime);
			    currentValue /= difftime(maxTime,minTime);
			    currentValue *= totalLength;
			    break;
			case MONTH:
			{
			    currentStep.tm_mon += intervalMult;
			    /*while(currentStep.tm_mon >= 12)
			    {
				currentStep.tm_year++;
				currentStep.tm_mon -= 12;
			    }*/

			    currentValue = difftime(mktime(&currentStep),minTime);
			    currentValue /= difftime(maxTime,minTime);
			    currentValue *= totalLength;
			    break;
			}
			case DAY:
			{
			    currentStep.tm_mday++;
			    currentValue = difftime(mktime(&currentStep),minTime);
			    currentValue /= difftime(maxTime,minTime);
			    currentValue *= totalLength;
			    break;
			}
			case HOUR:
			{
			    currentStep.tm_hour++;
			    currentValue = difftime(mktime(&currentStep),minTime);
			    currentValue /= difftime(maxTime,minTime);
			    currentValue *= totalLength;
			    break;
			}
			case MINUTE:
			{
			    currentStep.tm_min++;
			    currentValue = difftime(mktime(&currentStep),minTime);
			    currentValue /= difftime(maxTime,minTime);
			    currentValue *= totalLength;
			    break;
			}
			case SECOND:
			{
			    currentStep.tm_sec++;
			    currentValue = difftime(mktime(&currentStep),minTime);
			    currentValue /= difftime(maxTime,minTime);
			    currentValue *= totalLength;
			    break;
			}
			default:
			    break;
		    }
		}

		if(lowerTextss.str().size())
		{
		    osgText::Text * text = makeText(lowerTextss.str(),textColor);

		    float targetSize = padding * 0.67;
		    osg::BoundingBox bb = text->getBound();
		    float size1 = targetSize / (bb.zMax() - bb.zMin());
		    float size2 = totalLength / (bb.xMax() - bb.xMin());
		    text->setCharacterSize(std::min(size1,size2));
		    text->setAxisAlignment(axisAlign);
		    if(axisAlign == osgText::Text::USER_DEFINED_ROTATION)
		    {
			text->setRotation(q);
		    }

		    text->setPosition(startPoint + -tickDir * (padding * 0.65) + dir * 0.5 * totalLength + osg::Vec3(0,-1,0));

		    _axisGeode->addDrawable(text);

		}

		break;
	    }
	    case LINEAR:
	    {
		// find tick interval
		float rangeDif = maxValue - minValue;
		int power = (int)log10(rangeDif);
		float interval = pow(10.0, power);

		while(rangeDif / interval < 2)
		{
		    interval /= 10.0;
		}

		while(rangeDif / interval > 30)
		{
		    interval *= 10.0;
		}

		if(rangeDif / interval < 4)
		{
		    interval /= 2;
		}

		float tickValue = ((float)((int)(minValue/interval)))*interval;
		if(tickValue < minValue)
		{
		    tickValue += interval;
		}
    
		float value = (((tickValue - minValue) / (maxValue - minValue)) * totalLength);
		while(value <= totalLength)
		{
		    points->push_back(startPoint + dir * value);
		    points->push_back(startPoint + tickDir * tickLength + dir * value);

		    std::stringstream ss;
		    ss << tickValue;

		    osgText::Text * text = makeText(ss.str(),textColor);

		    float targetSize = padding * 0.27;
		    osg::BoundingBox bb = text->getBound();
		    text->setCharacterSize(targetSize / (bb.zMax() - bb.zMin()));
		    text->setAxisAlignment(axisAlign);
		    if(axisAlign == osgText::Text::USER_DEFINED_ROTATION)
		    {
			text->setRotation(q);
		    }

		    text->setPosition(startPoint + -tickDir * (padding * 0.5 * 0.3) + dir * value + osg::Vec3(0,-1,0));

		    _axisGeode->addDrawable(text);

		    tickValue += interval;
		    value = (((tickValue - minValue) / (maxValue - minValue)) * totalLength);
		}

		if(!axisLabel.empty())
		{
		    osgText::Text * text = makeText(axisLabel,textColor);

		    float targetSize = padding * 0.67;
		    osg::BoundingBox bb = text->getBound();
		    text->setCharacterSize(targetSize / (bb.zMax() - bb.zMin()));
		    text->setAxisAlignment(axisAlign);
		    if(axisAlign == osgText::Text::USER_DEFINED_ROTATION)
		    {
			text->setRotation(q);
		    }

		    text->setPosition(startPoint + -tickDir * (padding * (0.3 + 0.5 * 0.7)) + dir * (totalLength / 2.0) + osg::Vec3(0,-1,0));

		    _axisGeode->addDrawable(text);
		}
		break;
	    }
	    default:
		break;
	}

	geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES,0,points->size()));

	_axisGeode->addDrawable(geometry);
    }

    if(_dataInfoMap.size() == 1)
    {
	osgText::Text * text = makeText(_dataInfoMap.begin()->second.name,osg::Vec4(0.0,0.0,0.0,1.0));

	float targetHeight = padding * 0.95;
	float targetWidth = _width - (2.0 * padding);
	osg::BoundingBox bb = text->getBound();
	float hsize = targetHeight / (bb.zMax() - bb.zMin());
	float wsize = targetWidth / (bb.xMax() - bb.xMin());
	text->setCharacterSize(std::min(hsize,wsize));
	text->setAxisAlignment(osgText::Text::XZ_PLANE);

	text->setPosition(osg::Vec3(0,-1,(_height-padding)/2.0));

	_axisGeode->addDrawable(text);
    }
    else
    {
	static bool sizeCalibrated = false;
	static float spacerSize;

	if(!sizeCalibrated)
	{
	    osg::ref_ptr<osgText::Text> spacerText1 = makeText(": - :",osg::Vec4(0.0,0.0,0.0,1.0));
	    osg::ref_ptr<osgText::Text> spacerText2 = makeText("::",osg::Vec4(0.0,0.0,0.0,1.0));

	    float size1, size2;

	    osg::BoundingBox bb = spacerText1->getBound();
	    size1 = bb.xMax() - bb.xMin();
	    bb = spacerText2->getBound();
	    size2 = bb.xMax() - bb.xMin();

	    spacerSize = size1 - size2;
	    sizeCalibrated = true;
	}

	std::stringstream titless;

	for(std::map<std::string, GraphDataInfo>::iterator it = _dataInfoMap.begin(); it != _dataInfoMap.end();)
	{
	    titless << it->second.name;
	    it++;
	    if(it != _dataInfoMap.end())
	    {
		titless << " - ";
	    }
	}

	osg::ref_ptr<osgText::Text> text = makeText(titless.str(),osg::Vec4(0.0,0.0,0.0,1.0));
	
	float targetHeight = padding * 0.95;
	float targetWidth = _width - (2.0 * padding);
	osg::BoundingBox bb = text->getBound();
	float hsize = targetHeight / (bb.zMax() - bb.zMin());
	float wsize = targetWidth / (bb.xMax() - bb.xMin());

	float csize = std::min(hsize,wsize);

	bool defaultTitle = true;

	switch(_currentMultiGraphDisplayMode)
	{
	    case MGDM_COLOR:
	    case MGDM_COLOR_SOLID:
	    case MGDM_COLOR_PT_SIZE:
	    case MGDM_COLOR_SHAPE:
	    {
		float spSize = csize * spacerSize;
		float position = -((bb.xMax() - bb.xMin()) * csize) / 2.0;
		for(std::map<std::string,GraphDataInfo>::iterator it = _dataInfoMap.begin(); it != _dataInfoMap.end();)
		{
		    osgText::Text * ttext = makeText(it->second.name,it->second.singleColorArray->at(0));
		    ttext->setCharacterSize(csize);
		    ttext->setAxisAlignment(osgText::Text::XZ_PLANE);
		    ttext->setAlignment(osgText::Text::LEFT_CENTER);
		    ttext->setPosition(osg::Vec3(position,-1,(_height-padding)/2.0));
		    osg::BoundingBox tbb = ttext->getBound();
		    position += (tbb.xMax() - tbb.xMin());
		    _axisGeode->addDrawable(ttext);
		    it++;
		    if(it != _dataInfoMap.end())
		    {
			ttext = makeText("-",osg::Vec4(0,0,0,1));
			ttext->setCharacterSize(csize);
			ttext->setAxisAlignment(osgText::Text::XZ_PLANE);
			ttext->setAlignment(osgText::Text::CENTER_CENTER);
			ttext->setPosition(osg::Vec3(position + (spSize / 2.0),-1,(_height-padding)/2.0));
			_axisGeode->addDrawable(ttext);
			position += spSize;
		    }
		}
		defaultTitle = false;
		break;
	    }
	    default:
		break;
	}

	if(defaultTitle)
	{
	    text->setCharacterSize(std::min(hsize,wsize));
	    text->setAxisAlignment(osgText::Text::XZ_PLANE);

	    text->setPosition(osg::Vec3(0,-1,(_height-padding)/2.0));

	    _axisGeode->addDrawable(text);
	}
    }
}

void DataGraph::updateClip()
{
    float padding = calcPadding();
    float dataWidth = _width - (2.0 * padding);
    float dataHeight = _height - (2.0 * padding);
    float halfWidth = dataWidth / 2.0;
    float halfHeight = dataHeight / 2.0;

    osg::Vec3 point, normal;
    osg::Plane plane;

    point = osg::Vec3(-halfWidth,0,0);
    normal = osg::Vec3(1.0,0,0);
    plane = osg::Plane(normal,point);
    _clipNode->getClipPlane(0)->setClipPlane(plane);

    point = osg::Vec3(halfWidth,0,0);
    normal = osg::Vec3(-1.0,0,0);
    plane = osg::Plane(normal,point);
    _clipNode->getClipPlane(1)->setClipPlane(plane);

    point = osg::Vec3(0,0,halfHeight);
    normal = osg::Vec3(0,0,-1.0);
    plane = osg::Plane(normal,point);
    _clipNode->getClipPlane(2)->setClipPlane(plane);

    point = osg::Vec3(0,0,-halfHeight);
    normal = osg::Vec3(0,0,1.0);
    plane = osg::Plane(normal,point);
    _clipNode->getClipPlane(3)->setClipPlane(plane);

    _clipNode->setLocalStateSetModes(); 
}

void DataGraph::updateBGRanges()
{
    _bgRangesGeode->removeDrawables(0,_bgRangesGeode->getNumDrawables());

    osg::Geometry * geom = new osg::Geometry();
    osg::Vec3Array * verts = new osg::Vec3Array();
    osg::Vec4Array * colors = new osg::Vec4Array();
    geom->setVertexArray(verts);
    geom->setColorArray(colors);
    geom->setUseDisplayList(false);
    geom->setUseVertexBufferObjects(true);
    
    float padding = calcPadding();
    float wpadding = padding / _width;
    float hpadding = padding / _height;
    float dataWidth = (_width - (2.0 * padding)) / _width;
    float dataHeight = (_height - (2.0 * padding)) / _height;

    if(getNumGraphs() != 1 || !_bgRanges.size())
    {
	osg::Vec4 defaultColor(0.4,0.4,0.4,1.0);
	verts->push_back(osg::Vec3(wpadding,0.5,hpadding));
	verts->push_back(osg::Vec3(wpadding+dataWidth,0.5,hpadding));
	verts->push_back(osg::Vec3(wpadding+dataWidth,0.5,hpadding+dataHeight));
	verts->push_back(osg::Vec3(wpadding,0.5,hpadding+dataHeight));
	colors->push_back(defaultColor);
	geom->setColorBinding(osg::Geometry::BIND_OVERALL);
	geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,4));
    }
    else
    {
	for(int i = 0; i < _bgRanges.size(); ++i)
	{
	    verts->push_back(osg::Vec3(wpadding,0.5,hpadding+(_bgRanges[i].first*dataHeight)));
	    verts->push_back(osg::Vec3(wpadding+dataWidth,0.5,hpadding+(_bgRanges[i].first*dataHeight)));
	    verts->push_back(osg::Vec3(wpadding+dataWidth,0.5,hpadding+(_bgRanges[i].second*dataHeight)));
	    verts->push_back(osg::Vec3(wpadding,0.5,hpadding+(_bgRanges[i].second*dataHeight)));
	    colors->push_back(_bgRangesColors[i]);
	    colors->push_back(_bgRangesColors[i]);
	    colors->push_back(_bgRangesColors[i]);
	    colors->push_back(_bgRangesColors[i]);
	}
	geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
	geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,_bgRanges.size()*4));
    }

    _bgRangesGeode->addDrawable(geom);
}

void DataGraph::updateBar()
{
    float padding = calcPadding();
    float dataWidth = _width - (2.0 * padding);
    float dataHeight = _height - (2.0 * padding);

    osg::Matrix tran, scale;
    tran.makeTranslate(osg::Vec3(-0.5,0,-0.5));
    scale.makeScale(osg::Vec3(dataWidth,1.0,dataHeight));

    _barTransform->setMatrix(tran*scale);
}

float DataGraph::calcPadding()
{
    float minD = std::min(_width,_height);

    return 0.07 * minD;
}

osgText::Text * DataGraph::makeText(std::string text, osg::Vec4 color)
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

osg::Vec4 DataGraph::makeColor(float f)
{
    if(f < 0)
    {
        f = 0;
    }
    else if(f > 1.0)
    {
        f = 1.0;
    }

    osg::Vec4 color;
    color.w() = 1.0;

    if(f <= 0.33)
    {
        float part = f / 0.33;
        float part2 = 1.0 - part;

        color.x() = part2;
        color.y() = part;
        color.z() = 0;
    }
    else if(f <= 0.66)
    {
        f = f - 0.33;
        float part = f / 0.33;
        float part2 = 1.0 - part;

        color.x() = 0;
        color.y() = part2;
        color.z() = part;
    }
    else if(f <= 1.0)
    {
        f = f - 0.66;
        float part = f / 0.33;
        float part2 = 1.0 - part;

        color.x() = part;
        color.y() = 0;
        color.z() = part2;
    }

    //std::cerr << "Color x: " << color.x() << " y: " << color.y() << " z: " << color.z() << std::endl;

    return color;
}

