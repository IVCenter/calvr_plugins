#include "ColorSelector.h"

#include <kernel/InteractionManager.h>
#include <kernel/PluginHelper.h>
#include <kernel/NodeMask.h>
#include <util/Intersection.h>
#include <util/LocalToWorldVisitor.h>

#include <osg/Material>
#include <osg/PolygonMode>
#include <osg/Geometry>

#include <iostream>
#include <cmath>
#include <algorithm>

#ifdef WIN32
#define M_PI 3.141592653589793238462643
#endif

using namespace cvr;

float acos360(float x, float y,bool degrees)
{
    float r = x * x + y * y;
    r = sqrt(r);

    float val;
    if(x >= 0)
    {
        if(y >= 0)
        {
            val = acos( x / r );
        }
        else
        {
            val = 2*M_PI - acos( x / r );
        }
    }
    else
    {
        if(y >= 0)
        {
            val = M_PI - acos( -x / r );
        }
        else
        {
            val = M_PI + acos( -x / r );
        }
    }
    if(degrees)
    {
        return (val * 180.0) / M_PI;
    }
    else
    {
        return val;
    }
}

ColorSelector::ColorSelector(osg::Vec4 color, osg::Vec3 pos, float scale)
{
    _scale = scale;
    _position = pos;
    _color = color;
    _visible = false;
    _sphereRad = 50.0;
    _moving = false;

    _root = new osg::MatrixTransform();
    setMatrix();
    createGeometry();
}

ColorSelector::~ColorSelector()
{
}

bool ColorSelector::buttonEvent(int type, const osg::Matrix & mat)
{
    if(type == BUTTON_DOWN)
    {
	osg::Vec3 start, end(0,10000,0);
	start = start * mat;
	end = end * mat;
	std::vector<IsectInfo> isec = getObjectIntersection(PluginHelper::getScene(), start, end);
	bool found = false;
	for(int i = 0; i < isec.size(); i++)
	{
	    if(isec[i].geode == _sphereGeode)
	    {
		found = true;
		break;
	    }
	}
	if(!found)
	{
	    return false;
	}

	_moving = true;
	_pointerSpaceCenter = osg::Vec3(0,0,0);
	_pointerSpaceCenter = _pointerSpaceCenter * getLocalToWorldMatrix(_sphereGeode) * osg::Matrix::inverse(mat);

	return true;
    }
    else if(_moving && (type == BUTTON_DRAG || type == BUTTON_UP))
    {
	//osg::Matrix sphere2world = getLocalToWorldMatrix(_sphereGeode);
	osg::Matrix world2root = getLocalToWorldMatrix(_root.get());
	world2root = osg::Matrix::inverse(world2root);
	
	osg::Matrix m = _sphereTransform->getMatrix();
	osg::Vec3 newPoint;
	newPoint = _pointerSpaceCenter * mat * world2root;

	bool undef;
	osg::Vec3 newColor = xyz2hcl(newPoint,undef);
	//std::cerr << "New Color h: " << newColor.x() << " c: " << newColor.y() << " l: " << newColor.z() << std::endl;
	newPoint = hcl2xyz(newColor);
	newColor = hsl2rgb(newColor,undef);
	//std::cerr << "New Color r: " << newColor.x() << " g: " << newColor.y() << " b: " << newColor.z() << std::endl;

	_color.x() = newColor.x();
	_color.y() = newColor.y();
	_color.z() = newColor.z();
	setColor(_color);

	m.setTrans(newPoint);
	_sphereTransform->setMatrix(m);
	//_sphereTransform->setMatrix(sphere2world * _lastPointerInv * mat * world2root);
	//_lastPointerInv = osg::Matrix::inverse(mat);

	if(type == BUTTON_UP)
	{
	    _moving = false;
	}
	return true;
    }
    return false;
}

void ColorSelector::setColor(osg::Vec4 color)
{
    _color = color;
    _sphereDrawable->setColor(_color);
}

osg::Vec4 ColorSelector::getColor()
{
    return _color;
}

void ColorSelector::setVisible(bool v)
{
    if(v == _visible)
    {
	return;
    }

    if(v)
    {
	PluginHelper::getScene()->addChild(_root);
	//PluginHelper::getObjectsRoot()->addChild(_root);
    }
    else
    {
	PluginHelper::getScene()->removeChild(_root);
	//PluginHelper::getObjectsRoot()->removeChild(_root);
    }

    _visible = v;
}

bool ColorSelector::isVisible()
{
    return _visible;
}

void ColorSelector::setPosition(osg::Vec3 pos)
{
    _position = pos;
    setMatrix();
}

osg::Vec3 ColorSelector::getPosition()
{
    return _position;
}

void ColorSelector::setScale(float scale)
{
    _scale = scale;
    setMatrix();
}

float ColorSelector::getScale()
{
    return _scale;
}

void ColorSelector::setMatrix()
{
    osg::Matrix scale, trans;
    scale.makeScale(osg::Vec3(_scale,_scale,_scale));
    trans.makeTranslate(_position);
    _root->setMatrix(scale*trans);
}

osg::Vec3 ColorSelector::hsl2rgb(osg::Vec3 hsl, bool undefh)
{
    hsl.y() = (1.0 - fabs(2.0 * hsl.z() - 1.0)) * hsl.y();
    return hcl2rgb(hsl,undefh);
}

osg::Vec3 ColorSelector::hcl2rgb(osg::Vec3 hcl,bool undefh)
{
    static const float hdiv = 60.0 * M_PI / 180.0;
    float hprime = hcl.x() / hdiv;

    osg::Vec3 rgb;
    if(undefh)
    {
	rgb = osg::Vec3(0,0,0);
    }
    else
    {
	float x = hcl.y() * (1.0 - fabs(fmod(hprime,2.0f) - 1.0));
	int hpi = (int)hprime;

	switch(hpi)
	{
	    case 0:
		rgb = osg::Vec3(hcl.y(),x,0);
		break;
	    case 1:
		rgb = osg::Vec3(x,hcl.y(),0);
		break;
	    case 2:
		rgb = osg::Vec3(0,hcl.y(),x);
		break;
	    case 3:
		rgb = osg::Vec3(0,x,hcl.y());
		break;
	    case 4:
		rgb = osg::Vec3(x,0,hcl.y());
		break;
	    case 5:
		rgb = osg::Vec3(hcl.y(),0,x);
		break;
	    default:
		break;
	}
    }

    float m = hcl.z() - 0.5 * hcl.y();
    rgb.x() = rgb.x() + m;
    rgb.y() = rgb.y() + m;
    rgb.z() = rgb.z() + m;

    return rgb;
}

osg::Vec3 ColorSelector::rgb2hcl(osg::Vec4 color)
{
    float mincval,maxcval;
    mincval = std::min(color.x(),color.y());
    mincval = std::min(mincval,color.z());
    maxcval = std::max(color.x(),color.y());
    maxcval = std::max(maxcval,color.z());

    osg::Vec3 hcl;

    hcl.z() = 0.5 * (maxcval + mincval);
    hcl.y() = maxcval - mincval;
    float hprime = 0;

    if(hcl.y() == 0)
    {
	hprime = 0;
    }
    else if(maxcval == color.x())
    {
	hprime = fmod((color.y() - color.z()) / hcl.y(),6.0f);
    }
    else if(maxcval == color.y())
    {
	hprime = ((color.z() - color.x()) / hcl.y()) + 2.0;
    }
    else if(maxcval == color.z())
    {
	hprime = ((color.x() - color.y()) / hcl.y()) + 4.0;
    }
    hcl.x() = hprime * (M_PI * 60.0 / 180.0);

    return hcl;
}

osg::Vec3 ColorSelector::hcl2xyz(osg::Vec3 hcl)
{
    osg::Vec3 xyz;
    xyz.x() = (1.0 - fabs(2.0 * hcl.z() - 1.0)) * 500.0 * hcl.y();
    xyz.z() = ( 2.0 * (hcl.z() - 0.5)) * 500.0;
    osg::Matrix rot;
    rot.makeRotate(hcl.x(),osg::Vec3(0,0,1.0));
    xyz = xyz * rot;

    return xyz;
}

osg::Vec3 ColorSelector::xyz2hcl(osg::Vec3 xyz, bool & undefh)
{
    osg::Vec3 hcl;
    hcl.z() = (xyz.z() / 1000.0) + 0.5;
    hcl.z() = std::max(hcl.z(),0.0f);
    hcl.z() = std::min(hcl.z(),1.0f);

    float maxc = (1.0 - fabs(2.0 * hcl.z() - 1.0)) * 500.0;

    if(maxc == 0.0)
    {
	hcl.y() = 0;
    }
    else
    {
	float dist = (xyz - osg::Vec3(0,0,xyz.z())).length();
	if(dist >= maxc)
	{
	    hcl.y() = 1.0;
	}
	else
	{
	    hcl.y() = dist / maxc;
	}
    }

    if(hcl.y() == 0.0)
    {
	undefh = true;
	hcl.x() = 0;
    }
    else
    {
	undefh = false;
	hcl.x() = acos360(xyz.x(),xyz.y(),false);
    }

    return hcl;
}

void ColorSelector::createGeometry()
{
    _mainGeode = new osg::Geode();
    _root->addChild(_mainGeode);

    osg::Material * mat = new osg::Material();

    osg::StateSet * stateset;

    _mainGeode->setNodeMask(_mainGeode->getNodeMask() & ~(INTERSECT_MASK));

    static const float inc = 60.0 * M_PI / 180.0;

    for(float f = 0; f < 1.9 * M_PI; f += inc)
    {
	osg::Matrix mtran,mrot;
	mtran.makeTranslate(osg::Vec3(500,0,0));
	mrot.makeRotate(f,osg::Vec3(0,0,1));
	osg::Vec3 pos;
	pos = pos * mtran * mrot;

	osg::Sphere * sphere = new osg::Sphere(pos,_sphereRad);
	osg::Vec3 color(f,1.0,0.5);
	color = hsl2rgb(color);
	osg::Vec4 color4(color,1.0);

	osg::ShapeDrawable * sd = new osg::ShapeDrawable(sphere);
	sd->setColor(color4);
	_mainGeode->addDrawable(sd);

	stateset = sd->getOrCreateStateSet();
	//stateset->setAttributeAndModes(mat,osg::StateAttribute::ON);
    }

    osg::Sphere * sphere = new osg::Sphere(osg::Vec3(0,0,500),_sphereRad);
    osg::ShapeDrawable * sd = new osg::ShapeDrawable(sphere);
    sd->setColor(osg::Vec4(1.0,1.0,1.0,1.0));
    _mainGeode->addDrawable(sd);

    sphere = new osg::Sphere(osg::Vec3(0,0,-500),_sphereRad);
    sd = new osg::ShapeDrawable(sphere);
    sd->setColor(osg::Vec4(0.0,0.0,0.0,1.0));
    _mainGeode->addDrawable(sd);

    sphere = new osg::Sphere(osg::Vec3(0,0,-500),_sphereRad * 1.1);
    sd = new osg::ShapeDrawable(sphere);
    sd->setColor(osg::Vec4(0.0,0.0,0.0,1.0));
    _mainGeode->addDrawable(sd);

    stateset = sd->getOrCreateStateSet();
    osg::PolygonMode * pm = new osg::PolygonMode(osg::PolygonMode::FRONT_AND_BACK,osg::PolygonMode::LINE);
    stateset->setAttributeAndModes(pm,osg::StateAttribute::ON);

    //TODO: create lines
    
    osg::Geometry * lineGeometry = new osg::Geometry();
    osg::DrawArrays * ring = new osg::DrawArrays(osg::PrimitiveSet::LINE_STRIP,0,0);
    osg::Vec3Array * verts = new osg::Vec3Array(0);
    osg::Vec4Array * colors = new osg::Vec4Array(0);
    lineGeometry->setVertexArray(verts);
    lineGeometry->setColorArray(colors);
    lineGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    lineGeometry->addPrimitiveSet(ring);
    int count = 0;

    for(float f = 0; f <= 2.0001 * M_PI; f += 0.2)
    {
	osg::Matrix mtran,mrot;
	mtran.makeTranslate(osg::Vec3(500,0,0));
	mrot.makeRotate(f,osg::Vec3(0,0,1));
	osg::Vec3 pos;
	pos = pos * mtran * mrot;

	osg::Vec3 color(f,1.0,0.5);
	color = hsl2rgb(color);
	osg::Vec4 color4(color,1.0);

	verts->push_back(pos);
	colors->push_back(color4);

	count++;
    }

    ring->setCount(count);
    _mainGeode->addDrawable(lineGeometry);

    stateset = lineGeometry->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    lineGeometry = new osg::Geometry();
    ring = new osg::DrawArrays(osg::PrimitiveSet::LINES,0,0);
    verts = new osg::Vec3Array(0);
    colors = new osg::Vec4Array(0);
    lineGeometry->setVertexArray(verts);
    lineGeometry->setColorArray(colors);
    lineGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    lineGeometry->addPrimitiveSet(ring);
    count = 0;

    for(float f = 0; f < 1.9 * M_PI; f += inc)
    {
	osg::Matrix mrot;
	mrot.makeRotate(f,osg::Vec3(0,0,1));

	float maxc = 0.0;
	osg::Vec3 lastPos(0,0,-500);
	lastPos = lastPos * mrot;
	
	osg::Vec3 tcolor;
	osg::Vec4 lastColor;
	tcolor = osg::Vec3(f,0,0);
	tcolor = hsl2rgb(tcolor);
	lastColor = osg::Vec4(tcolor,1.0);

	for(float g = 0.05; g <= 1.01; g += 0.05)
	{
	    maxc = (1.0 - fabs(2.0 * g - 1.0)) * 500.0;
	    osg::Vec3 pos(maxc,0,(g * 1000.0) - 500.0);
	    pos = pos * mrot;
	    tcolor = osg::Vec3(f,1.0,g);
	    tcolor = hsl2rgb(tcolor);
	    osg::Vec4 color(tcolor,1.0);

	    verts->push_back(lastPos);
	    verts->push_back(pos);

	    colors->push_back(lastColor);
	    colors->push_back(color);

	    lastColor = color;
	    lastPos = pos;
	    
	    count += 2;
	}
    }

    ring->setCount(count);
    _mainGeode->addDrawable(lineGeometry);

    stateset = lineGeometry->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    _sphereTransform = new osg::MatrixTransform();
    _root->addChild(_sphereTransform);
    _sphereGeode = new osg::Geode();
    _sphereTransform->addChild(_sphereGeode);

    sphere = new osg::Sphere(osg::Vec3(0,0,0),_sphereRad * 1.2);
    _sphereDrawable = new osg::ShapeDrawable(sphere);
    _sphereDrawable->setColor(_color);
    _sphereGeode->addDrawable(_sphereDrawable);

    osg::Vec3 hcl = rgb2hcl(_color);

    //std::cerr << "h: " << hcl.x() << " c: " << hcl.y() << " l: " << hcl.z() << std::endl;

    osg::Vec3 pos = hcl2xyz(hcl);
    //std::cerr << "pos x: " << pos.x() << " y: " << pos.y() << " z: " << pos.z() << std::endl;

    osg::Matrix m;
    m.makeTranslate(pos);
    _sphereTransform->setMatrix(m);

    osg::TessellationHints * hint = new osg::TessellationHints();
    hint->setDetailRatio(0.25);

    sphere = new osg::Sphere(osg::Vec3(0,0,0),_sphereRad * 1.2 * 1.1);
    sd = new osg::ShapeDrawable(sphere, hint);
    sd->setColor(osg::Vec4(1.0,1.0,1.0,1.0));
    _sphereGeode->addDrawable(sd);

    stateset = sd->getOrCreateStateSet();
    stateset->setAttributeAndModes(pm,osg::StateAttribute::ON);
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
}
