#include "VolumeDrawable.h"
#include <iostream>
#include <osg/Shader>
#include <fstream>
#include <sstream>
#include <cvrConfig/ConfigManager.h>
#include <osg/DispatchCompute>
#include <osg/BindImageTexture>
#include <osg/Texture3D>
#include <osg/CullFace>
#include <gl/GL.h>

#pragma comment (lib, "OpenGL32.lib")

std::string VolumeDrawable::loadShaderFile(std::string filename) const
{
	std::ifstream file(filename);
	if (!file) return "";

	std::stringstream sstr;
	sstr << file.rdbuf();

	file.close();
	
	return sstr.str();
}

VolumeDrawable::VolumeDrawable()
{
	init();
}

VolumeDrawable::VolumeDrawable(const VolumeDrawable & drawable,
	const osg::CopyOp & copyop)
	: Drawable(drawable, copyop)
{
	init();
}

VolumeDrawable::~VolumeDrawable()
{
}

void VolumeDrawable::init()
{
	std::string shaderDir = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.ShaderDir");
	std::string vert = loadShaderFile(shaderDir + "volume.vert");
	std::string frag = loadShaderFile(shaderDir + "volume.frag");
	std::string compute = loadShaderFile(shaderDir + "volume.glsl");

	if (vert.empty() || frag.empty() || compute.empty())
	{
		return;
	}

	_program = new osg::Program;
	_program->setName("Volume");
	_program->addShader(new osg::Shader(osg::Shader::VERTEX, vert));
	_program->addShader(new osg::Shader(osg::Shader::FRAGMENT, frag));

	_volumeState = new osg::StateSet();
	_volumeState->setAttribute(_program);


	_computeProgram = new osg::Program;
	_computeProgram->setName("Bake");
	_computeProgram->addShader(new osg::Shader(osg::Shader::COMPUTE, compute));

	//_computeState = new osg::StateSet();
	_volumeState = new osg::StateSet();

	_volumeState->setAttribute(new osg::CullFace(osg::CullFace::FRONT));
	_volumeState->setMode(GL_CULL_FACE, osg::StateAttribute::ON);
	_volumeState->setAttribute(_program);

	osg::ref_ptr<osg::Uniform> mvp = new osg::Uniform("MVP", osg::Matrix());
	osg::ref_ptr<osg::Uniform> vto = new osg::Uniform("ViewToObject", osg::Matrix());
	osg::ref_ptr<osg::Uniform> ip = new osg::Uniform("InverseProjection", osg::Matrix());
	osg::ref_ptr<osg::Uniform> cp = new osg::Uniform("CameraPosition", osg::Vec3());

	_volumeState->addUniform(mvp);
	_volumeState->addUniform(vto);
	_volumeState->addUniform(ip);
	_volumeState->addUniform(cp);


	osg::ref_ptr<osg::Uniform> pp = new osg::Uniform("PlanePoint", osg::Vec3());
	osg::ref_ptr<osg::Uniform> pn = new osg::Uniform("PlaneNormal", osg::Vec3());
	osg::ref_ptr<osg::Uniform> ss = new osg::Uniform("StepSize", 0.0f);

	_volumeState->addUniform(pp);
	_volumeState->addUniform(pn);
	_volumeState->addUniform(ss);


	osg::ref_ptr<osg::Uniform> vol = new osg::Uniform("Volume", 0);
	osg::ref_ptr<osg::Uniform> dt = new osg::Uniform("DepthTexture", 1);

	_volumeState->addUniform(vol);
	_volumeState->addUniform(dt);
}

void VolumeDrawable::drawImplementation(osg::RenderInfo &renderInfo) const 
{
	std::cout << "drawing volume" << std::endl;

	//_program->apply();
	//osg::State & state = *renderInfo.getState();

	//state.pushStateSet(_volumeState);

	glDrawArrays(GL_TRIANGLES, 0, 12 * 3);
}

osg::BoundingSphere VolumeDrawable::computeBound() const
{
	osg::BoundingSphere sphere = osg::BoundingSphereImpl<osg::Vec3f>(osg::Vec3f(0, 0, 0), 1.0f);
	return sphere;
}

osg::BoundingBox VolumeDrawable::computeBoundingBox() const
{
	osg::BoundingBox box = osg::BoundingBoxImpl<osg::Vec3f>(-1, -1, -1, 1, 1, 1);
	return box;
}



static const GLfloat g_vertex_buffer_data[] = {
	-1.0f,-1.0f,-1.0f, // triangle 1 : begin
	-1.0f,-1.0f, 1.0f,
	-1.0f, 1.0f, 1.0f, // triangle 1 : end
	1.0f, 1.0f,-1.0f, // triangle 2 : begin
	-1.0f,-1.0f,-1.0f,
	-1.0f, 1.0f,-1.0f, // triangle 2 : end
	1.0f,-1.0f, 1.0f,
	-1.0f,-1.0f,-1.0f,
	1.0f,-1.0f,-1.0f,
	1.0f, 1.0f,-1.0f,
	1.0f,-1.0f,-1.0f,
	-1.0f,-1.0f,-1.0f,
	-1.0f,-1.0f,-1.0f,
	-1.0f, 1.0f, 1.0f,
	-1.0f, 1.0f,-1.0f,
	1.0f,-1.0f, 1.0f,
	-1.0f,-1.0f, 1.0f,
	-1.0f,-1.0f,-1.0f,
	-1.0f, 1.0f, 1.0f,
	-1.0f,-1.0f, 1.0f,
	1.0f,-1.0f, 1.0f,
	1.0f, 1.0f, 1.0f,
	1.0f,-1.0f,-1.0f,
	1.0f, 1.0f,-1.0f,
	1.0f,-1.0f,-1.0f,
	1.0f, 1.0f, 1.0f,
	1.0f,-1.0f, 1.0f,
	1.0f, 1.0f, 1.0f,
	1.0f, 1.0f,-1.0f,
	-1.0f, 1.0f,-1.0f,
	1.0f, 1.0f, 1.0f,
	-1.0f, 1.0f,-1.0f,
	-1.0f, 1.0f, 1.0f,
	1.0f, 1.0f, 1.0f,
	-1.0f, 1.0f, 1.0f,
	1.0f,-1.0f, 1.0f
};