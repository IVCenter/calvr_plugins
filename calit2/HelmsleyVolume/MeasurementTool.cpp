#include "MeasurementTool.h"
#include "HelmsleyVolume.h"
#include <cvrConfig/ConfigManager.h>
#include <osg/CullFace>

#include <sstream>
#include <iomanip>


MeasurementTool::MeasurementTool()
{
	init();
}

MeasurementTool::MeasurementTool(const MeasurementTool & group,
	const osg::CopyOp & copyop)
	: Group(group, copyop)
{
	init();
}

MeasurementTool::~MeasurementTool()
{
}

void MeasurementTool::init()
{
	std::string vert = HelmsleyVolume::loadShaderFile("ruler.vert");
	std::string frag = HelmsleyVolume::loadShaderFile("ruler.frag");

	if (vert.empty() || frag.empty())
	{
		std::cerr << "Helmsley Volume shaders not found!" << std::endl;
		return;
	}

	_start = osg::Vec3(0, 0, 0);
	_end = osg::Vec3(0, 0, 0);
	_ustart = new osg::Uniform("Start", _start);
	_uend = new osg::Uniform("End", _end);

	_text = new osgText::Text();
	_text->setCharacterSize(20);
	_text->setAlignment(osgText::TextBase::CENTER_BOTTOM);
	_text->setAutoRotateToScreen(true);
	_text->setColor(osg::Vec4(1, 1, 1, 1));
	_text->setEnableDepthWrites(false);
	_text->getOrCreateStateSet()->setRenderBinDetails(INT_MAX, "RenderBin");


	osg::ref_ptr<osg::Drawable> box = new osg::ShapeDrawable(new osg::Box(osg::Vec3(0, 0, -0.5), 20, 2, 1));
	osg::StateSet* stateset = box->getOrCreateStateSet();
	stateset->setMode(GL_LIGHTING, osg::StateAttribute::ON);
	stateset->addUniform(_ustart);
	stateset->addUniform(_uend);
	//stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
	stateset->setAttribute(new osg::CullFace(osg::CullFace::BACK));
	stateset->setMode(GL_CULL_FACE, osg::StateAttribute::ON);
	//stateset->setMode(GL_BLEND, osg::StateAttribute::ON);


	osg::Program* program = new osg::Program();
	program->setName("Ruler");
	program->addShader(new osg::Shader(osg::Shader::VERTEX, vert));
	program->addShader(new osg::Shader(osg::Shader::FRAGMENT, frag));

	stateset->setAttributeAndModes(program, osg::StateAttribute::ON);


	osg::ref_ptr<osg::Geode> g = new osg::Geode();
	_ruler = new osg::MatrixTransform();

	g->addDrawable(box);
	_ruler->addChild(g);
	this->addChild(_ruler);
	this->addChild(_text);
}

void MeasurementTool::setStart(osg::Vec3 v)
{
	_start = v;
	_ustart->set(v);
	update();
}

void MeasurementTool::setEnd(osg::Vec3 v)
{
	_end = v;
	_uend->set(v);
	update();
}

void MeasurementTool::setText(std::string s)
{
	_text->setText(s);
}

void MeasurementTool::update()
{
	osg::Vec3 midpoint = (_start + _end) / 2.0;
	midpoint += osg::Vec3(0, 0, 25.0);
	_text->setPosition(midpoint);

	float dist = FLT_MAX;
	cvr::SceneObject* closest = nullptr;
	float scale = 1.0f;

	for (int i = 0; i < HelmsleyVolume::instance()->getSceneObjects().size(); ++i)
	{
		cvr::SceneObject* so = HelmsleyVolume::instance()->getSceneObjects()[i];
		const osg::BoundingBox bb = so->getOrComputeBoundingBox();

		float distance = (bb.center() - midpoint).length();
		if (distance < dist)
		{
			dist = distance;
			closest = so;
		}
	}

	if (closest)
	{
		scale = closest->getScale();
	}

	
	osg::Vec3 forward = (_end - _start);
	float length = forward.length();
	//forward.normalize();


	std::stringstream stream;
	stream << std::fixed << std::setprecision(2) << length / scale;
	setText(stream.str() + "mm");


	osg::Matrix m = osg::Matrix();
	//m.set(right.x(), right.y(), right.z(), _start.x(),
		//forward.x() * length, forward.y() * length, forward.z() * length, _start.y(),
		//up.x(), up.y(), up.z(), _start.z(),
		//0, 0, 0, 1);
	m.makeLookAt(_start, _end, osg::Vec3(0, 1, 0));
	m = osg::Matrix::inverse(m);
	m.preMultScale(osg::Vec3(1, 1, length));
	//m.postMultTranslate(_start);
	_ruler->setMatrix(m);
}

float MeasurementTool::getLength()
{
	return (_end - _start).length();
}