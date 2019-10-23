#include "VolumeGroup.h"
#include "ImageLoader.hpp"
#include "HelmsleyVolume.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/ScreenConfig.h>
#include <cvrKernel/CVRViewer.h>

#include <osg/Shader>
#include <osg/Texture2D>
#include <osg/Texture3D>
#include <osg/Texture>
#include <osg/Notify>
#include <osg/BlendFunc>
#include <osg/MatrixTransform>
#include <osg/Depth>

#include <osgDB/ReadFile>

void VolumeDrawCallback::drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
{
	if (!group)
	{
		std::cerr << "group doesn't exist!" << std::endl;
		return;
	}
	const osg::GLExtensions* ext = renderInfo.getState()->get<osg::GLExtensions>();

	if (!cvr::ScreenBase::resolveBuffers(renderInfo.getCurrentCamera(), group->_resolveFBO, renderInfo.getState(), GL_DEPTH_BUFFER_BIT))
	{
		std::cout << "Depth buffer could not be resolved" << std::endl;
	}

	const osg::Viewport* currview = renderInfo.getState()->getCurrentViewport();
	const osg::Viewport* totalview = renderInfo.getCurrentCamera()->getViewport();
	osg::Vec4 relativeView = osg::Vec4(
		currview->width() / totalview->width(),
		currview->height() / totalview->height(),
		currview->x() / totalview->width(),
		currview->y() / totalview->height()
	);
	group->_RelativeViewport->set(relativeView);

	group->_RelativeViewport->apply(ext, group->_program->getPCP(*renderInfo.getState())->getUniformLocation("RelativeViewport"));

	drawable->drawImplementation(renderInfo);
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
	_computeUniforms.clear();
	_dirty.clear();
}

osg::Matrix VolumeGroup::getObjectToWorldMatrix()
{
	osg::Matrix mat = osg::Matrix::identity();
	if(_transform)
	{
		_transform->computeLocalToWorldMatrix(mat, NULL);
	}
	return mat;
}

osg::Matrix VolumeGroup::getWorldToObjectMatrix()
{
	osg::Matrix mat = osg::Matrix::identity();
	if (_transform)
	{
		_transform->computeWorldToLocalMatrix(mat, NULL);
	}
	return mat;
}

void VolumeGroup::init()
{
	_hasMask = false;
	_computeUniforms = std::map<std::string, osg::ref_ptr<osg::Uniform> >();
	_dirty = std::map<osg::GraphicsContext*, bool>();
	
	std::string vert = HelmsleyVolume::loadShaderFile("volume.vert");
	std::string frag = HelmsleyVolume::loadShaderFile("volume.frag");
	std::string compute = HelmsleyVolume::loadShaderFile("volume.glsl");

	if (vert.empty() || frag.empty() || compute.empty())
	{
		std::cerr << "Helmsley Volume shaders not found!" << std::endl;
		return;
	}


	//Set up depth buffer fbo
	_resolveFBO = new osg::FrameBufferObject();
	_depthTexture = new osg::Texture2D();
	_depthTexture->setTextureSize(2048, 2048);
	_depthTexture->setResizeNonPowerOfTwoHint(false);
	_depthTexture->setSourceFormat(GL_DEPTH_COMPONENT);
	//_depthTexture->setSourceType(GL_UNSIGNED_INT);
	_depthTexture->setInternalFormat(GL_DEPTH_COMPONENT32);
	_depthTexture->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::NEAREST);
	_depthTexture->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::NEAREST);
	_depthTexture->setWrap(osg::Texture::WRAP_R, osg::Texture::CLAMP_TO_EDGE);
	_depthTexture->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
	_resolveFBO->setAttachment(osg::Camera::DEPTH_BUFFER, osg::FrameBufferAttachment(_depthTexture));

	_colorTexture = new osg::Texture2D();
	_colorTexture->setTextureSize(2048, 2048);
	_colorTexture->setResizeNonPowerOfTwoHint(false);
	_colorTexture->setInternalFormat(GL_RGBA);
	_colorTexture->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::NEAREST);
	_colorTexture->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::NEAREST);
	_colorTexture->setWrap(osg::Texture::WRAP_R, osg::Texture::CLAMP_TO_EDGE);
	_colorTexture->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
	_resolveFBO->setAttachment(osg::Camera::COLOR_BUFFER0, osg::FrameBufferAttachment(_colorTexture));



	_program = new osg::Program;
	_program->setName("Volume");
	_program->addShader(new osg::Shader(osg::Shader::VERTEX, vert));
	_program->addShader(new osg::Shader(osg::Shader::FRAGMENT, frag));
	
	_transform = new osg::MatrixTransform();
	_cube = new osg::ShapeDrawable(new osg::Box(osg::Vec3(0, 0, 0), 1, 1, 1));
	osg::Vec4Array* colors = new osg::Vec4Array;
	colors->push_back(osg::Vec4(1, 1, 1, 1));
	_cube->setColorArray(colors);
	_cube->setColorBinding(osg::Geometry::BIND_OVERALL);

	_cube->setDrawCallback(new VolumeDrawCallback(this));
	_cube->setDataVariance(osg::Object::DYNAMIC);
	_cube->setUseDisplayList(false);

	osg::StateSet* states = _cube->getOrCreateStateSet();
	_side = new osg::CullFace(osg::CullFace::FRONT);
	states->setAttribute(_side);
	states->setMode(GL_CULL_FACE, osg::StateAttribute::ON);
	states->setMode(GL_BLEND, osg::StateAttribute::ON);
	states->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
	states->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);
	//states->setRenderBinDetails(INT_MAX, "RenderBin");
	states->setAttributeAndModes(_program.get(), osg::StateAttribute::ON);

	osg::ref_ptr<osg::Geode> g = new osg::Geode();
	g->addChild(_cube);
	_transform->addChild(g);
	this->addChild(_transform);
	

	_PlanePoint = new osg::Uniform("PlanePoint", osg::Vec3(0.f, -2.f, 0.f));
	_PlaneNormal = new osg::Uniform("PlaneNormal", osg::Vec3(0.f, 1.f, 0.f));
	_StepSize = new osg::Uniform("StepSize", .00135f);
	_RelativeViewport = new osg::Uniform("RelativeViewport", osg::Vec4(1, 1, 0, 0));

	states->addUniform(_PlanePoint);
	states->addUniform(_PlaneNormal);
	states->addUniform(_StepSize);
	//states->addUniform(_RelativeViewport);

	osg::ref_ptr<osg::Uniform> vol = new osg::Uniform("Volume", (int)0);
	osg::ref_ptr<osg::Uniform> dt = new osg::Uniform("DepthTexture", (int)1);

	states->addUniform(vol);
	states->addUniform(dt);
	
	states->setTextureAttribute(1, _depthTexture, osg::StateAttribute::ON);
	states->setTextureMode(1, GL_TEXTURE_2D, osg::StateAttribute::ON);



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

void VolumeGroup::loadVolume(std::string path, std::string maskpath)
{
	//osg::setNotifyLevel(osg::NotifySeverity::DEBUG_INFO);
	//osg::Vec3 s = osg::Vec3(0,0,0);
	osg::Matrix m = osg::Matrix::identity();
	osg::ref_ptr<osg::Image> i = ImageLoader::LoadVolume(path, m);
	if (!i)
	{
		std::cerr << "Volume could not be loaded" << std::endl;
		return;
	}
	//_transform->setScale(s);
	_transform->setMatrix(m);
	osg::Vec3 scale;
	osg::Vec3 translation;
	osg::Quat rot;
	osg::Quat so;
	m.decompose(translation, rot, scale, so);
	if (scale.x() < 0)
	{
		flipCull();
	}
	if (scale.y() < 0)
	{
		flipCull();
	}
	if (scale.z() < 0)
	{
		flipCull();
	}
	std::cout << "Scale: " << scale.x() << ", " << scale.y() << ", " << scale.z() << std::endl;

	if (maskpath.compare("") != 0)
	{
		loadMask(maskpath, i);
		_hasMask = true;
		//_computeNode->getOrCreateStateSet()->setDefine("MASK", true);
		//_computeNode->getOrCreateStateSet()->setDefine("COLON", true);
	}

	//std::cout << size.x() << ", " << size.y() << ", " << size.z() << std::endl;

	//_pat->setScale(size);

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

void VolumeGroup::loadMask(std::string path, osg::Image* volume)
{
	unsigned int width = volume->s();
	unsigned int height = volume->t();
	unsigned int depth = volume->r();
	uint16_t * volumeData = (uint16_t*)volume->data();
	int i = 0;
	for (i = 0; i < depth; ++i)
	{
		std::string maskpath = path + "\\" + std::to_string(depth - (i + 1)) + ".png";
		osg::ref_ptr<osg::Image> mask = osgDB::readImageFile(maskpath);
		mask->flipVertical();
		unsigned int bytesize = mask->getPixelSizeInBits() / 8;
		unsigned char* maskData = mask->data();

		uint16_t* slice = volumeData + 2 * i * width * height;
		for (unsigned int y = 0; y < height; ++y)
		{
			for (unsigned int x = 0; x < width; ++x)
			{
				unsigned int volpixel = 2 * (x + y * width);
				unsigned int maskpixel = bytesize * (x + y * width);
				slice[volpixel + 1] = 256 * (uint16_t)(maskData[maskpixel]);
			}
		}
	}
}

void VolumeGroup::precompute()
{

	if (!_baked)
	{
		_baked = new osg::Texture3D;
		osg::ref_ptr<osg::Image> bimage = new osg::Image();
		bimage->allocateImage(_volume->getTextureWidth(), _volume->getTextureHeight(), _volume->getTextureDepth(), GL_RGBA, GL_UNSIGNED_BYTE);
		_baked->setImage(bimage);
		_baked->setTextureSize(_volume->getTextureWidth(), _volume->getTextureHeight(), _volume->getTextureDepth());
		_baked->setFilter(osg::Texture3D::MIN_FILTER, osg::Texture3D::LINEAR);
		_baked->setFilter(osg::Texture3D::MAG_FILTER, osg::Texture3D::LINEAR);
		_baked->setInternalFormat(GL_RGBA8);
		_baked->setResizeNonPowerOfTwoHint(false);

		_baked->setName("BAKED");
		_baked->setTextureSize(_volume->getTextureWidth(), _volume->getTextureHeight(), _volume->getTextureDepth());

		osg::StateSet* states = _computeNode->getOrCreateStateSet();
		states->setTextureAttribute(1, _baked, osg::StateAttribute::ON);
		states->setTextureMode(1, GL_TEXTURE_3D, osg::StateAttribute::ON);


		osg::ref_ptr<osg::BindImageTexture> imagbinding = new osg::BindImageTexture(0, _volume, osg::BindImageTexture::READ_ONLY, GL_RG16, 0, GL_TRUE);
		osg::ref_ptr<osg::BindImageTexture> imagbinding2 = new osg::BindImageTexture(1, _baked, osg::BindImageTexture::READ_WRITE, GL_RGBA8, 0, GL_TRUE);

		states->setAttributeAndModes(imagbinding);
		states->setAttributeAndModes(imagbinding2);



		states = _cube->getOrCreateStateSet();
		states->setTextureAttribute(0, _baked, osg::StateAttribute::ON);
		states->setTextureMode(0, GL_TEXTURE_3D, osg::StateAttribute::ON);




		this->addChild(_computeNode);
	}
}

void VolumeGroup::flipCull()
{
	if (_side->getMode() == osg::CullFace::FRONT)
	{
		_side->setMode(osg::CullFace::BACK);
	}
	else
	{
		_side->setMode(osg::CullFace::FRONT);
	}
}