#include "VolumeGroup.h"
#include "ImageLoader.hpp"
#include <osg/Shader>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/ScreenConfig.h>
#include <cvrKernel/CVRViewer.h>
#include <osg/Texture2D>
#include <osg/Notify>
#include <osg/BlendFunc>
#include <osg/MatrixTransform>

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

osg::Matrix VolumeGroup::getObjectToWorldMatrix()
{
	osg::Matrix mat = osg::Matrix::identity();
	if(_pat)
	{
		_pat->computeLocalToWorldMatrix(mat, NULL);
	}
	return mat;
}

osg::Matrix VolumeGroup::getWorldToObjectMatrix()
{
	osg::Matrix mat = osg::Matrix::identity();
	if (_pat)
	{
		_pat->computeWorldToLocalMatrix(mat, NULL);
	}
	return mat;
}

void VolumeGroup::init()
{
	_computeUniforms = std::map<std::string, osg::ref_ptr<osg::Uniform>>();

	
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
	//osg::ShaderDefines d = _program->getShaderDefines();
	//d.insert("SAMPLECOUNT");
	//_program->setShaderDefines(d);
	
	_pat = new osg::PositionAttitudeTransform();
	_cube = new osg::ShapeDrawable(new osg::Box(osg::Vec3(0, 0, 0), 1, 1, 1));
	osg::Vec4Array* colors = new osg::Vec4Array;
	colors->push_back(osg::Vec4(1, 1, 1, 1));
	_cube->setColorArray(colors);
	_cube->setColorBinding(osg::Geometry::BIND_OVERALL);

	osg::StateSet* states = _cube->getOrCreateStateSet();
	states->setAttribute(new osg::CullFace(osg::CullFace::FRONT));
	states->setMode(GL_CULL_FACE, osg::StateAttribute::ON);
	states->setMode(GL_BLEND, osg::StateAttribute::ON);
	states->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

	states->setAttributeAndModes(_program.get(), osg::StateAttribute::ON);

	osg::ref_ptr<osg::Geode> g = new osg::Geode();
	g->addChild(_cube);
	_pat->addChild(g);
	this->addChild(_pat);


	/*
	osg::ref_ptr<osg::Texture2D> tex = new osg::Texture2D;
	tex->setName("TEX");
	tex->setTextureSize(1920, 1080);
	tex->setSourceFormat(GL_DEPTH_COMPONENT);
	tex->setSourceType(GL_FLOAT);
	tex->setInternalFormat(GL_DEPTH_COMPONENT24);
	tex->setFilter(osg::Texture2D::MIN_FILTER,
		osg::Texture2D::LINEAR);
	tex->setFilter(osg::Texture2D::MAG_FILTER,
		osg::Texture2D::LINEAR);

	osg::ref_ptr<osg::Texture2D> tex2 = new osg::Texture2D;
	tex2->setName("TEX2");
	tex2->setTextureSize(1920, 1080);
	tex2->setInternalFormat(GL_RGBA32F_ARB);
	tex2->setSourceFormat(GL_RGBA);
	tex2->setSourceType(GL_FLOAT);
	tex2->setFilter(osg::Texture2D::MIN_FILTER,
		osg::Texture2D::LINEAR);
	tex2->setFilter(osg::Texture2D::MAG_FILTER,
		osg::Texture2D::LINEAR);

	states->setTextureAttribute(1, tex2, osg::StateAttribute::ON);
	states->setTextureMode(1, GL_TEXTURE_2D, osg::StateAttribute::ON);
	*/

	
	std::vector<osg::Camera*> cameras = std::vector<osg::Camera*>();
	cvr::CVRViewer::instance()->getCameras(cameras);
	for (int i = 0; i < cameras.size(); ++i)
	{
		cameras[i]->getGraphicsContext()->getState()->setUseModelViewAndProjectionUniforms(true);
		/*
		osg::Camera::BufferAttachmentMap bam = cameras[i]->getBufferAttachmentMap();
		if (!bam.count(osg::Camera::DEPTH_BUFFER))
		{
			//cameras[i]->setRenderOrder(osg::Camera::PRE_RENDER, 0);
			cameras[i]->setRenderTargetImplementation(osg::Camera::RenderTargetImplementation::FRAME_BUFFER_OBJECT);

			//std::cout << cameras[i]->getName() << std::endl;
			//tex->setResizeNonPowerOfTwoHint(false);
			//tex->setUseHardwareMipMapGeneration(false);

			cameras[i]->attach(osg::Camera::DEPTH_BUFFER, tex.get());


			//tex2->setResizeNonPowerOfTwoHint(false);
			//tex2->setUseHardwareMipMapGeneration(false);

			cameras[i]->attach(osg::Camera::COLOR_BUFFER, tex2.get());


		}
		*/
	}

	_cube->setDrawCallback(new VolumeDrawCallback(this));
	_cube->setUseDisplayList(false);
	

	_PlanePoint = new osg::Uniform("PlanePoint", osg::Vec3(0.f, -2.f, 0.f));
	_PlaneNormal = new osg::Uniform("PlaneNormal", osg::Vec3(0.f, 1.f, 0.f));
	_StepSize = new osg::Uniform("StepSize", .00135f);

	states->addUniform(_PlanePoint);
	states->addUniform(_PlaneNormal);
	states->addUniform(_StepSize);

	osg::ref_ptr<osg::Uniform> vol = new osg::Uniform("Volume", (int)0);
	osg::ref_ptr<osg::Uniform> dt = new osg::Uniform("DepthTexture", 1);

	states->addUniform(vol);
	states->addUniform(dt);
	


	_computeProgram = new osg::Program;
	_computeProgram->setName("Bake");
	_computeProgram->addShader(new osg::Shader(osg::Shader::COMPUTE, compute));
	osg::ShaderDefines cd = _computeProgram->getShaderDefines();
	cd.insert("LIGHT_DIRECTIONAL");
	_computeProgram->setShaderDefines(cd);

	
	_computeNode = new osg::DispatchCompute(0, 0, 0);
	_computeNode->setDataVariance(osg::Object::DYNAMIC);
	_computeNode->setDrawCallback(new ComputeDrawCallback);
	osg::StateSet* computeStates = _computeNode->getOrCreateStateSet();
	computeStates->setAttributeAndModes(_computeProgram.get(), osg::StateAttribute::ON);
	states->setRenderBinDetails(-1, "RenderBin");
	//this->addChild(_computeNode);


	_computeUniforms["Exposure"] = new osg::Uniform("Exposure", 1.5f);
	_computeUniforms["Density"] = new osg::Uniform("Density", .5f);
	_computeUniforms["Threshold"] = new osg::Uniform("Threshold", .2f);

	_computeUniforms["LightDensity"] = new osg::Uniform("LightDensity", 300.f);
	_computeUniforms["LightPosition"] = new osg::Uniform("LightPosition", osg::Vec3(0, 0.1f, 0));
	_computeUniforms["LightDirection"] = new osg::Uniform("LightDirection", osg::Vec3(1.0f, 0, 0));
	_computeUniforms["LightAngle"] = new osg::Uniform("LightAngle", .5f);
	_computeUniforms["LightAmbient"] = new osg::Uniform("LightAmbient", .2f);
	_computeUniforms["LightIntensity"] = new osg::Uniform("LightIntensity", 100.0f);

	_computeUniforms["WorldScale"] = new osg::Uniform("WorldScale", osg::Vec3(1, 1, 1));
	_computeUniforms["TexelSize"] = new osg::Uniform("TexelSize", osg::Vec3(1.0f / 512.0f, 1.0f / 512.0f, 1.0f / 128.0f));

	_computeUniforms["volume"] = new osg::Uniform("volume", (int)0);
	_computeUniforms["baked"] = new osg::Uniform("baked", (int)1);

	for (std::map<std::string, osg::ref_ptr<osg::Uniform>>::iterator it = _computeUniforms.begin(); it != _computeUniforms.end(); ++it)
	{
		computeStates->addUniform(it->second);
	}

}

void VolumeGroup::loadVolume(std::string path)
{
	//osg::setNotifyLevel(osg::NotifySeverity::DEBUG_INFO);
	osg::Vec3 size = osg::Vec3(0,0,0);
	osg::Image* i = ImageLoader::LoadVolume(path, size);
	if (!i)
	{
		std::cerr << "Volume could not be loaded" << std::endl;
		return;
	}
	_pat->setScale(size);
	

	_volume = new osg::Texture3D;
	_volume->setImage(i);
	_volume->setTextureSize(i->s(), i->t(), i->r());
	_volume->setFilter(osg::Texture3D::FilterParameter::MIN_FILTER, osg::Texture3D::FilterMode::LINEAR);
	_volume->setFilter(osg::Texture3D::FilterParameter::MAG_FILTER, osg::Texture3D::FilterMode::LINEAR);
	_volume->setInternalFormat(GL_RG16);
	_volume->setName("VOLUME");
	_volume->setResizeNonPowerOfTwoHint(false);

	OSG_NOTICE << "Volume texture size: " << (int)(_volume->getTextureWidth()) << ", " << (int)(_volume->getTextureHeight()) << ", " << (int)(_volume->getTextureDepth()) << std::endl;


	_computeNode->setComputeGroups((i->s() + 7) / 8, (i->t() + 7) / 8, (i->r() + 7) / 8);
	_computeUniforms["TexelSize"]->set(osg::Vec3(1.0f / (float)i->s(), 1.0f / (float)i->t(), 1.0f / (float)i->r()));
	
	osg::StateSet* states = _computeNode->getOrCreateStateSet();
	//states->setTextureAttribute(0, _volume, osg::StateAttribute::ON);
	//states->setTextureMode(0, GL_TEXTURE_3D, osg::StateAttribute::ON);
	states->setTextureAttributeAndModes(0, _volume.get());
	//states->setAttributeAndModes(imagbinding.get());

	setDirty();
	precompute();

	//osg::setNotifyLevel(osg::NotifySeverity::NOTICE);

}

void VolumeGroup::precompute()
{

	if (!_baked)
	{
		_baked = new osg::Texture3D;
		osg::ref_ptr<osg::Image> bimage = new osg::Image();
		bimage->allocateImage(_volume->getTextureWidth(), _volume->getTextureHeight(), _volume->getTextureDepth(), GL_RG, GL_UNSIGNED_SHORT);
		_baked->setImage(bimage);
		_baked->setTextureSize(_volume->getTextureWidth(), _volume->getTextureHeight(), _volume->getTextureDepth());
		_baked->setFilter(osg::Texture3D::FilterParameter::MIN_FILTER, osg::Texture3D::FilterMode::LINEAR);
		_baked->setFilter(osg::Texture3D::FilterParameter::MAG_FILTER, osg::Texture3D::FilterMode::LINEAR);
		_baked->setInternalFormat(GL_RG16);
		_baked->setResizeNonPowerOfTwoHint(false);

		_baked->setName("BAKED");
		_baked->setTextureSize(_volume->getTextureWidth(), _volume->getTextureHeight(), _volume->getTextureDepth());

		osg::StateSet* states = _computeNode->getOrCreateStateSet();
		//states->setTextureAttribute(1, _baked.get(), osg::StateAttribute::ON);
		//states->setTextureMode(1, GL_TEXTURE_3D, osg::StateAttribute::ON);
		states->setTextureAttributeAndModes(1, _baked.get());
		//states->setAttributeAndModes(imagbinding.get());


		osg::ref_ptr<osg::BindImageTexture> imagbinding = new osg::BindImageTexture(1, _baked.get(), osg::BindImageTexture::READ_WRITE, GL_RG16, 0, GL_TRUE);
		osg::ref_ptr<osg::BindImageTexture> imagbinding2 = new osg::BindImageTexture(0, _volume.get(), osg::BindImageTexture::READ_ONLY, GL_RG16);

		states->setAttributeAndModes(imagbinding);
		states->setAttributeAndModes(imagbinding2);



		states = _cube->getOrCreateStateSet();
		//states->setTextureAttribute(0, _baked.get(), osg::StateAttribute::ON);
		//states->setTextureMode(0, GL_TEXTURE_3D, osg::StateAttribute::ON);
		states->setTextureAttributeAndModes(0, _baked.get());




		this->addChild(_computeNode);
	}
	
	/*
	if (_dirty)
	{
		this->addChild(_computeNode);
		setDirty(false);
	}
	else
	{
		this->removeChild(_computeNode);
	}
	*/
}