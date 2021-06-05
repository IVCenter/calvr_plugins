#include "Selection3DTool.h"
#include "HelmsleyVolume.h"
#include <cvrConfig/ConfigManager.h>

#include <sstream>
#include <iomanip>

using namespace cvr;

void Selection3DTool::init()
{
	std::string vert = HelmsleyVolume::loadShaderFile("selection3D.vert");
	std::string frag = HelmsleyVolume::loadShaderFile("selection3D.frag");

	if (vert.empty() || frag.empty())
	{
		std::cerr << "Helmsley Volume shaders not found!" << std::endl;
		return;
	}

	_start = osg::Vec3(0, 0, 0);
	_end = osg::Vec3(0, 0, 0);
	_ustart = new osg::Uniform("Start", _start);
	_uend = new osg::Uniform("End", _end);

	osg::ref_ptr<osg::Drawable> box = new osg::ShapeDrawable(new osg::Box(osg::Vec3(0, 0, 0.0), 1, 1, 1));
	_stateset = box->getOrCreateStateSet();
	_stateset->setMode(GL_LIGHTING, osg::StateAttribute::ON);
	_stateset->addUniform(_ustart);
	_stateset->addUniform(_uend);
	_stateset->setRenderingHint(osg::StateSet::OPAQUE_BIN);
	//stateset->setAttribute(new osg::CullFace(osg::CullFace::FRONT), osg::StateAttribute::ON);
	_stateset->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
	_stateset->setRenderBinDetails(0, "RenderBin");
	_stateset->setMode(GL_CULL_FACE, osg::StateAttribute::OFF | osg::StateAttribute::OVERRIDE);
	//_stateset->setMode(GL_BLEND, osg::StateAttribute::ON);


	osg::Program* program = new osg::Program();
	program->setName("Selection");
	program->addShader(new osg::Shader(osg::Shader::VERTEX, vert));
	program->addShader(new osg::Shader(osg::Shader::FRAGMENT, frag));

	_stateset->setAttributeAndModes(program, osg::StateAttribute::ON);


	osg::ref_ptr<osg::Geode> g = new osg::Geode();
	_ruler = new osg::MatrixTransform();

	g->addDrawable(box);
	_ruler->addChild(g);
	this->addChild(_ruler);
}

void Selection3DTool::setStart(osg::Vec3 v)
{
	_start = v;
	_ustart->set(v);
	update();
}

void Selection3DTool::setEnd(osg::Vec3 v)
{
	_end = v;
	_uend->set(v);
	update();
}
 

void Selection3DTool::activate()
{
	_ruler->setNodeMask(0xFFFFFFFF);
 }

void Selection3DTool::deactivate()
{
	_ruler->setNodeMask(0);
 }

void Selection3DTool::update()
{
	osg::Vec3 midpoint = (_start + _end) / 2.0;
	midpoint += osg::Vec3(0, 0, 25.0);
 
	//parent scene object is volume scene object
	float scale = _parent->getScale();
	setTransform(osg::Matrix::inverse(_parent->getTransform()));


	osg::Vec3 forward = (_end - _start);
	float length = forward.length();
	//forward.normalize();


   

	osg::Matrix m = osg::Matrix();
	/*m.set(right.x(), right.y(), right.z(), _start.x(),
		forward.x() * length, forward.y() * length, forward.z() * length, _start.y(),
		up.x(), up.y(), up.z(), _start.z(),
		0, 0, 0, 1);*/
	m.makeLookAt(_start, osg::Vec3(_start.x(), 1,_start.z()), osg::Vec3(0, 0, 1));
	m = osg::Matrix::inverse(m);

	m.preMultScale(osg::Vec3(std::abs(forward.x()), std::abs(forward.z()), std::abs(forward.y())));
	//m.preMultScale(osg::Vec3(length, length, length));
	//m.preMultScale(osg::Vec3(forward.z(), forward.z(), forward.z()));

	//m.postMultTranslate(_start);


	_ruler->setMatrix(m);
}

float Selection3DTool::getLength()
{
	return (_end - _start).length();
}