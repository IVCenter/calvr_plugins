#include "GraphKeyObject.h"
#include "GraphLayoutObject.h"

#include <cvrKernel/CalVR.h>
#include <cvrUtil/OsgMath.h>

using namespace cvr;

GraphKeyObject::GraphKeyObject(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : LayoutLineObject(name,navigation,movable,clip,contextMenu,showBounds)
{
    setBoundsCalcMode(SceneObject::MANUAL);

    _geode = new osg::Geode();
    _bgGeom = new osg::Geometry();
    _boxGeom = new osg::Geometry();

    _geode->setCullingActive(false);
    _boxGeom->setUseDisplayList(false);
    _boxGeom->setUseVertexBufferObjects(true);
    _bgGeom->setUseDisplayList(false);
    _bgGeom->setUseVertexBufferObjects(true);

    osg::Vec3Array * verts = new osg::Vec3Array(4);
    osg::Vec4Array * colors = new osg::Vec4Array(1);

    _width = _height = 1000.0;

    verts->at(0) = osg::Vec3(-_width/2.0,1,_height/2.0);
    verts->at(1) = osg::Vec3(-_width/2.0,1,-_height/2.0);
    verts->at(2) = osg::Vec3(_width/2.0,1,-_height/2.0);
    verts->at(3) = osg::Vec3(_width/2.0,1,_height/2.0);

    colors->at(0) = osg::Vec4(0.9,0.9,0.9,1.0);

    _bgGeom->setVertexArray(verts);
    _bgGeom->setColorArray(colors);
    _bgGeom->setColorBinding(osg::Geometry::BIND_OVERALL);
    _bgGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,4));

    _geode->addDrawable(_bgGeom);
    _geode->addDrawable(_boxGeom);

    addChild(_geode);

    osg::BoundingBox bb(-(_width*0.5),-2,-(_height*0.5),_width*0.5,0,_height*0.5);
    setBoundingBox(bb);

    _font = osgText::readFontFile(CalVR::instance()->getHomeDir() + "/resources/arial.ttf");

    osg::StateSet * stateset = _geode->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
}

GraphKeyObject::~GraphKeyObject()
{
}

void GraphKeyObject::setKeys(std::vector<osg::Vec4> & colors, std::vector<std::string> & labels)
{
    if(!colors.size() || colors.size() != labels.size())
    {
	return;
    }

    // reset geometry
    for(int i = 0; i < _textList.size(); ++i)
    {
	_geode->removeDrawable(_textList[i]);
    }

    _boxGeom->setVertexArray(NULL);
    _boxGeom->setColorArray(NULL);
    _boxGeom->removePrimitiveSet(0,_boxGeom->getNumPrimitiveSets());

    _rangeList.clear();

    _colors = colors;
    _labels = labels;

    osg::Vec3Array * verts = new osg::Vec3Array(_colors.size()*4);
    osg::Vec4Array * colorArray = new osg::Vec4Array(_colors.size()*4);

    for(int i = 0; i < _colors.size(); ++i)
    {
	colorArray->at((i*4)+0) = _colors[i];
	colorArray->at((i*4)+1) = _colors[i];
	colorArray->at((i*4)+2) = _colors[i];
	colorArray->at((i*4)+3) = _colors[i];
    }

    _boxGeom->setVertexArray(verts);
    _boxGeom->setColorArray(colorArray);
    _boxGeom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    _boxGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,_colors.size()*4));

    for(int i = 0; i < _labels.size(); ++i)
    {
	osgText::Text * text = new osgText::Text();
	text->setCharacterSize(1.0);
	text->setAlignment(osgText::Text::LEFT_CENTER);
	text->setColor(osg::Vec4(0,0,0,1));
	text->setBackdropColor(osg::Vec4(0,0,0,0));
	text->setAxisAlignment(osgText::Text::XZ_PLANE);
	text->setText(_labels[i]);
	if(_font)
	{
	    text->setFont(_font);
	}
	_textList.push_back(text);
	_geode->addDrawable(text);
    }

    update();
}

void GraphKeyObject::setSize(float width, float height)
{
    _width = width;
    _height = height;

    osg::BoundingBox bb(-(_width*0.5),-2,-(_height*0.5),_width*0.5,0,_height*0.5);
    setBoundingBox(bb);

    update();
}

bool GraphKeyObject::eventCallback(InteractionEvent * ie)
{
    TrackedButtonInteractionEvent * tie = ie->asTrackedButtonEvent();
    if(tie)
    {
	if(tie->getButton() == 0 && (tie->getInteraction() == BUTTON_DOWN || tie->getInteraction() == BUTTON_DOUBLE_CLICK))
	{
	    osg::Vec3 point1, point2(0,1000.0,0);
	    point1 = point1 * tie->getTransform() * getWorldToObjectMatrix();
	    point2 = point2 * tie->getTransform() * getWorldToObjectMatrix();

	    osg::Vec3 planePoint, planeNormal(0,-1,0), intersect;
	    float w;

	    if(linePlaneIntersectionRef(point1,point2,planePoint,planeNormal,intersect,w))
	    {
		for(int i = 0; i < _rangeList.size(); ++i)
		{
		    if(intersect.x() >= _rangeList[i].first && intersect.x() <= _rangeList[i].second)
		    {
			std::string group = _labels[i];
			std::vector<std::string> emptyList;

			GraphLayoutObject * layout = dynamic_cast<GraphLayoutObject*>(_parent);
			if(layout)
			{
			    layout->selectPatients(group,emptyList);
			}

			return true;
		    }
		}
	    }
	}
    }

    return false;
}

void GraphKeyObject::update()
{
    if(!_colors.size())
    {
	return;
    }

    _rangeList.clear();

    float space = 0.01*_width;
    float boxSize = 0.85*_height;

    float maxTextSpace = _width - (3.0*space+boxSize)*((float)_colors.size());

    float textSpace = 0.0;
    float maxTextHeight = FLT_MIN;

    for(int i = 0; i < _textList.size(); ++i)
    {
	_textList[i]->setCharacterSize(1.0);
	osg::BoundingBox bb = _textList[i]->getBound();

	textSpace += bb.xMax() - bb.xMin();

	float textHeight = bb.zMax() - bb.zMin();
	if(textHeight > maxTextHeight)
	{
	    maxTextHeight = textHeight;
	}
    }

    float csize1, csize2;
    csize1 = maxTextSpace / textSpace;
    csize2 = boxSize / maxTextHeight;

    float csize = std::min(csize1,csize2);
    
    float actualTextSpace = 0.0;
    for(int i = 0; i < _textList.size(); ++i)
    {
	_textList[i]->setCharacterSize(csize);
	osg::BoundingBox bb = _textList[i]->getBound();

	actualTextSpace += bb.xMax() - bb.xMin();
    }

    float totalWidth = (3.0*space+boxSize)*((float)_colors.size()) + actualTextSpace;
    
    float position = (-_width / 2.0) + (_width - totalWidth) / 2.0;
    position += space;

    osg::Vec3Array * verts = dynamic_cast<osg::Vec3Array*>(_boxGeom->getVertexArray());
    int vertIndex = 0;

    if(verts)
    {
	for(int i = 0; i < _textList.size(); ++i)
	{
	    float start,end;
	    start = position;

	    verts->at(vertIndex+0) = osg::Vec3(position,0,boxSize/2.0);
	    verts->at(vertIndex+1) = osg::Vec3(position,0,-boxSize/2.0);
	    verts->at(vertIndex+2) = osg::Vec3(position+boxSize,0,-boxSize/2.0);
	    verts->at(vertIndex+3) = osg::Vec3(position+boxSize,0,boxSize/2.0);
	    vertIndex += 4;
	    position += boxSize + space;

	    _textList[i]->setPosition(osg::Vec3(position,0,0));
	    osg::BoundingBox bb = _textList[i]->getBound();

	    end = position + bb.xMax() - bb.xMin();
	    _rangeList.push_back(std::pair<float,float>(start,end));

	    position += bb.xMax() - bb.xMin() + space + space;
	}
	verts->dirty();
    }

    verts = dynamic_cast<osg::Vec3Array*>(_bgGeom->getVertexArray());

    if(verts)
    {
	verts->at(0) = osg::Vec3(-_width/2.0,1,_height/2.0);
	verts->at(1) = osg::Vec3(-_width/2.0,1,-_height/2.0);
	verts->at(2) = osg::Vec3(_width/2.0,1,-_height/2.0);
	verts->at(3) = osg::Vec3(_width/2.0,1,_height/2.0);
	verts->dirty();
    }
}
