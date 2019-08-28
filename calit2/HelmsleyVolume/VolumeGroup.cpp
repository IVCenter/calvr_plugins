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
#include <osg/Texture3D>
#include <osg/Texture>
#include <osg/Notify>
#include <osg/BlendFunc>
#include <osg/MatrixTransform>
#include <osg/Depth>

std::string VolumeGroup::loadShaderFile(std::string filename)
{
	std::ifstream file(filename.c_str());
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
	_computeUniforms = std::map<std::string, osg::ref_ptr<osg::Uniform> >();
	_dirty = std::map<osg::GraphicsContext*, bool>();
	
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
	_computeNode->setDrawCallback(new ComputeDrawCallback(this));
	osg::StateSet* computeStates = _computeNode->getOrCreateStateSet();
	computeStates->setAttributeAndModes(_computeProgram.get(), osg::StateAttribute::ON);
	computeStates->setRenderBinDetails(-1, "RenderBin");
	//this->addChild(_computeNode);


	//_computeUniforms["Exposure"] = new osg::Uniform("Exposure", 1.5f);
	_computeUniforms["OpacityCenter"] = new osg::Uniform("OpacityCenter", 1.0f);
	_computeUniforms["OpacityWidth"] = new osg::Uniform("OpacityWidth", 1.0f);
	_computeUniforms["OpacityMult"] = new osg::Uniform("OpacityMult", 1.0f);
	_computeUniforms["ContrastBottom"] = new osg::Uniform("ContrastBottom", 0.0f);
	_computeUniforms["ContrastTop"] = new osg::Uniform("ContrastTop", 1.0f);

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

	for (std::map<std::string, osg::ref_ptr<osg::Uniform> >::iterator it = _computeUniforms.begin(); it != _computeUniforms.end(); ++it)
	{
		computeStates->addUniform(it->second);
	}

}

void VolumeGroup::loadVolume(std::string path, osg::Vec3 size)
{
	//osg::setNotifyLevel(osg::NotifySeverity::DEBUG_INFO);
	osg::Vec3 s = osg::Vec3(0,0,0);
	osg::Image* i = ImageLoader::LoadVolume(path, s);
	if (!i)
	{
		std::cerr << "Volume could not be loaded" << std::endl;
		return;
	}
	//_pat->setScale(size);

	std::cout << size.x() << ", " << size.y() << ", " << size.z() << std::endl;

	_pat->setScale(size);


	_volume = new osg::Texture3D;
	_volume->setImage(i);
	_volume->setTextureSize(i->s(), i->t(), i->r());
	_volume->setFilter(osg::Texture3D::MIN_FILTER, osg::Texture3D::LINEAR);
	_volume->setFilter(osg::Texture3D::MAG_FILTER, osg::Texture3D::LINEAR);
	_volume->setInternalFormat(GL_RG16);
	_volume->setName("VOLUME");
	_volume->setResizeNonPowerOfTwoHint(false);

	OSG_NOTICE << "Volume texture size: " << (int)(_volume->getTextureWidth()) << ", " << (int)(_volume->getTextureHeight()) << ", " << (int)(_volume->getTextureDepth()) << std::endl;


	_computeNode->setComputeGroups((i->s() + 7) / 8, (i->t() + 7) / 8, (i->r() + 7) / 8);
	_computeUniforms["TexelSize"]->set(osg::Vec3(1.0f / (float)i->s(), 1.0f / (float)i->t(), 1.0f / (float)i->r()));
	
	osg::StateSet* states = _computeNode->getOrCreateStateSet();
	states->setTextureAttribute(0, _volume, osg::StateAttribute::ON);
	states->setTextureMode(0, GL_TEXTURE_3D, osg::StateAttribute::ON);

	//Set dirty on all graphics contexts
	std::vector<osg::Camera*> cameras = std::vector<osg::Camera*>();
	cvr::CVRViewer::instance()->getCameras(cameras);
	for (int i = 0; i < cameras.size(); ++i)
	{
		setDirty(cameras[i]->getGraphicsContext());
	}

	precompute();


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
		_baked->setFilter(osg::Texture3D::MIN_FILTER, osg::Texture3D::LINEAR);
		_baked->setFilter(osg::Texture3D::MAG_FILTER, osg::Texture3D::LINEAR);
		_baked->setInternalFormat(GL_RG16);
		_baked->setResizeNonPowerOfTwoHint(false);

		_baked->setName("BAKED");
		_baked->setTextureSize(_volume->getTextureWidth(), _volume->getTextureHeight(), _volume->getTextureDepth());

		osg::StateSet* states = _computeNode->getOrCreateStateSet();
		states->setTextureAttribute(1, _baked, osg::StateAttribute::ON);
		states->setTextureMode(1, GL_TEXTURE_3D, osg::StateAttribute::ON);


		osg::ref_ptr<osg::BindImageTexture> imagbinding = new osg::BindImageTexture(0, _volume, osg::BindImageTexture::READ_ONLY, GL_RG16, 0, GL_TRUE);
		osg::ref_ptr<osg::BindImageTexture> imagbinding2 = new osg::BindImageTexture(1, _baked, osg::BindImageTexture::READ_WRITE, GL_RG16, 0, GL_TRUE);

		states->setAttributeAndModes(imagbinding);
		states->setAttributeAndModes(imagbinding2);



		states = _cube->getOrCreateStateSet();
		states->setTextureAttribute(0, _baked, osg::StateAttribute::ON);
		states->setTextureMode(0, GL_TEXTURE_3D, osg::StateAttribute::ON);




		this->addChild(_computeNode);
	}
}