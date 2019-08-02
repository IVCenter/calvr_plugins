#include "VolumeGroup.h"
#include "ImageLoader.hpp"
#include <osg/Shader>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/CVRViewer.h>

std::string VolumeGroup::loadShaderFile(std::string filename) const
{
	std::ifstream file(filename);
	if (!file) return "";

	std::stringstream sstr;
	sstr << file.rdbuf();

	file.close();

	return sstr.str();
}

VolumeGroup::VolumeGroup()
{
	init();
}

VolumeGroup::VolumeGroup(const VolumeGroup & group,
	const osg::CopyOp & copyop)
	: Group(group, copyop)
{
	init();
}

VolumeGroup::~VolumeGroup()
{
}

void VolumeGroup::init()
{
	
	std::string shaderDir = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.ShaderDir");
	std::string vert = loadShaderFile(shaderDir + "test.vert");
	std::string frag = loadShaderFile(shaderDir + "test.frag");
	std::string compute = loadShaderFile(shaderDir + "volume.glsl");

	if (vert.empty() || frag.empty() || compute.empty())
	{
		std::cerr << "Helmsley Volume shaders not found!" << std::endl;
		return;
	}

	_program = new osg::Program;
	_program->setName("Volume");
	_program->addShader(new osg::Shader(osg::Shader::VERTEX, vert));
	_program->addShader(new osg::Shader(osg::Shader::FRAGMENT, frag));


	_computeProgram = new osg::Program;
	_computeProgram->setName("Bake");
	_computeProgram->addShader(new osg::Shader(osg::Shader::COMPUTE, compute));
	
	//_computeState = new osg::StateSet();
	_pat = new osg::PositionAttitudeTransform();
	_cube = new osg::ShapeDrawable(new osg::Box(osg::Vec3(0, 0, 0), 1, 1, 1));
	osg::Vec4Array* colors = new osg::Vec4Array;
	colors->push_back(osg::Vec4(1, 1, 1, 1));
	_cube->setColorArray(colors);
	_cube->setColorBinding(osg::Geometry::BIND_OVERALL);

	osg::StateSet* states = _cube->getOrCreateStateSet();
	states->setAttribute(new osg::CullFace(osg::CullFace::FRONT));
	states->setMode(GL_CULL_FACE, osg::StateAttribute::ON);
	states->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

	states->setAttributeAndModes(_program.get(), osg::StateAttribute::ON);

	osg::ref_ptr<osg::Geode> g = new osg::Geode();
	g->addChild(_cube);
	_pat->addChild(g);
	this->addChild(_pat);

	std::vector<osg::Camera*> cameras = std::vector<osg::Camera*>();
	cvr::CVRViewer::instance()->getCameras(cameras);
	for (int i = 0; i < cameras.size(); ++i)
	{
		cameras[i]->getGraphicsContext()->getState()->setUseModelViewAndProjectionUniforms(true);
	}

	_cube->setDrawCallback(new VolumeDrawCallback(this));
	_cube->setUseDisplayList(false);
	
	_MVP = new osg::Uniform("MVP", osg::Matrix());
	_ViewToObject = new osg::Uniform("ViewToObject", osg::Matrix());
	_InverseProjection = new osg::Uniform("InverseProjection", osg::Matrix());
	_CameraPosition = new osg::Uniform("CameraPosition", osg::Vec3());
	_ViewInverse = new osg::Uniform("ViewInverse", osg::Matrix());
	states->addUniform(_MVP);
	states->addUniform(_ViewToObject);
	states->addUniform(_InverseProjection);
	states->addUniform(_CameraPosition);
	states->addUniform(_ViewInverse);


	_PlanePoint = new osg::Uniform("PlanePoint", osg::Vec3());
	_PlaneNormal = new osg::Uniform("PlaneNormal", osg::Vec3());
	_StepSize = new osg::Uniform("StepSize", 0.0f);

	states->addUniform(_PlanePoint);
	states->addUniform(_PlaneNormal);
	states->addUniform(_StepSize);


	osg::ref_ptr<osg::Uniform> vol = new osg::Uniform("Volume", 0);
	osg::ref_ptr<osg::Uniform> dt = new osg::Uniform("DepthTexture", 1);

	states->addUniform(vol);
	states->addUniform(dt);
	
}

void VolumeGroup::loadVolume(std::string path)
{
	osg::Vec3 size = osg::Vec3(0,0,0);
	_volume = ImageLoader::LoadVolume(path, size);
	if (!_volume)
	{
		std::cerr << "Volume could not be loaded" << std::endl;
		return;
	}
	_pat->setScale(size);
	
	osg::StateSet* states = _cube->getOrCreateStateSet();
	osg::Texture3D* texture = new osg::Texture3D;
	texture->setImage(_volume.get());
	texture->setFilter(osg::Texture3D::FilterParameter::MIN_FILTER, osg::Texture3D::FilterMode::LINEAR);
	texture->setFilter(osg::Texture3D::FilterParameter::MAG_FILTER, osg::Texture3D::FilterMode::LINEAR);
	states->setTextureAttribute(0, texture, osg::StateAttribute::ON);
	states->setTextureMode(0, GL_TEXTURE_3D, osg::StateAttribute::ON);
	
	//states->setTextureMode(0, GL_TEXTURE_GEN_S, osg::StateAttribute::ON);
	//states->setTextureMode(0, GL_TEXTURE_GEN_T, osg::StateAttribute::ON);
}