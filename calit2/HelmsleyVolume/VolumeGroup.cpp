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

#define ISOVALUE 0.08f
#define MCFACTOR 2
#define CONFIGLUTSIZE 4096


void VolumeDrawCallback::drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
{
	if (!group)
	{
		std::cerr << "group doesnt not xist!" << std::endl;
		return;
	}
	const osg::GLExtensions* ext = renderInfo.getState()->get<osg::GLExtensions>();

	if (!cvr::ScreenBase::resolveBuffers(renderInfo.getCurrentCamera(), group->_resolveFBO, renderInfo.getState(), GL_DEPTH_BUFFER_BIT))
	{
		//std::cout << "Depth buffer could not be resolved" << std::endl;
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

	//checkStatuses();


}

void VolumeDrawCallback::checkStatuses() const{
	if (group->_statusDirty) {
		if (((MarchingCubeCallback*)group->_marchingCubeNode->getDrawCallback())->ready[0] == 1) {
			group->readyMCUI();
		}
	}
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
	_minMaxShader = HelmsleyVolume::loadShaderFile("minMax.comp");
	_excessShader = HelmsleyVolume::loadShaderFile("excess.comp");
	_histShader = HelmsleyVolume::loadShaderFile("hist.comp");
	_clipShader = HelmsleyVolume::loadShaderFile("clipHist1.comp");
	_clipShader2 = HelmsleyVolume::loadShaderFile("clipHist2.comp");
	_lerpShader = HelmsleyVolume::loadShaderFile("lerp.comp");
	_totalHistShader = HelmsleyVolume::loadShaderFile("nonClaheHist.comp");
	_marchingCubesShader = HelmsleyVolume::loadShaderFile("marchingcube.glsl");

	if (vert.empty() || frag.empty() || compute.empty())
	{
		std::cerr << "Helsey Volume shaders not found!" << std::endl;
		return;
	}


	//Set up depth buffer fbo
	_resolveFBO = new osg::FrameBufferObject();
	_depthTexture = new osg::Texture2D();
	_depthTexture->setTextureSize(2048, 2048);
	_depthTexture->setResizeNonPowerOfTwoHint(false);
	_depthTexture->setSourceFormat(GL_DEPTH_COMPONENT);
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
	states->setAttributeAndModes(_program.get(), osg::StateAttribute::ON);

	osg::ref_ptr<osg::Geode> g = new osg::Geode();
	g->addChild(_cube);
	g->getOrCreateStateSet()->setRenderBinDetails(7, "RenderBin");
 
	_transform->addChild(g);
	this->addChild(_transform);
	

	_PlanePoint = new osg::Uniform("PlanePoint", osg::Vec3(0.f, -2.f, 0.f));
	_PlaneNormal = new osg::Uniform("PlaneNormal", osg::Vec3(0.f, 1.f, 0.f));
	_StepSize = new osg::Uniform("StepSize", .00150f);
	_testScale = new osg::Uniform("TestScale", .001f);
	_maxSteps = new osg::Uniform("MaxSteps", .98f);
	_RelativeViewport = new osg::Uniform("RelativeViewport", osg::Vec4(1, 1, 0, 0));

	states->addUniform(_PlanePoint);
	states->addUniform(_PlaneNormal);
	states->addUniform(_StepSize);
	states->addUniform(_testScale);
	states->addUniform(_maxSteps);
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
	computeStates->setRenderBinDetails(6, "RenderBin");

	_computeUniforms["OpacityCenter"] = new osg::Uniform(osg::Uniform::FLOAT, "OpacityCenter", 10);
	_computeUniforms["OpacityCenter"]->setElement(0, .5f);
	_computeUniforms["OpacityWidth"] = new osg::Uniform(osg::Uniform::FLOAT, "OpacityWidth", 10);
	_computeUniforms["OpacityWidth"]->setElement(0, 1.0f);
	_computeUniforms["OpacityTopWidth"] = new osg::Uniform(osg::Uniform::FLOAT, "OpacityTopWidth", 10);
	_computeUniforms["OpacityTopWidth"]->setElement(0, 1.0f);
	_computeUniforms["OpacityMult"] = new osg::Uniform(osg::Uniform::FLOAT, "OpacityMult", 10);;
	_computeUniforms["OpacityMult"]->setElement(0, 1.0f);
	_computeUniforms["Lowest"] = new osg::Uniform(osg::Uniform::FLOAT, "Lowest", 10);
	_computeUniforms["Lowest"]->setElement(0, 0.0f);
	_computeUniforms["TriangleCount"] = new osg::Uniform("TriangleCount", 1.0f);
	_computeUniforms["ContrastBottom"] = new osg::Uniform("ContrastBottom", 0.0f);
	_computeUniforms["ContrastTop"] = new osg::Uniform("ContrastTop", 1.0f);
	_computeUniforms["TrueContrast"] = new osg::Uniform("TrueContrast", 1.0f);
	_computeUniforms["ContrastCenter"] = new osg::Uniform("ContrastCenter", 0.5f);
	_computeUniforms["leftColor"] = new osg::Uniform("leftColor", osg::Vec3(1.0f, 0.0f, 0.0f));
	_computeUniforms["rightColor"] = new osg::Uniform("rightColor", osg::Vec3(1.0f, 1.0f, 1.0f));
	_computeUniforms["Brightness"] = new osg::Uniform("Brightness", 0.5f);
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

	////////////////centerline
	_centerLineGeodes = new std::vector<osg::ref_ptr<osg::Geode>>();
	_colonCoords = FileSelector::loadCenterLine(path, FileSelector::OrganEnum::COLON);
	if (!_colonCoords->empty()) {
		Line* colonLine = new Line(_colonCoords, osg::Vec4(UI_BLUE_COLOR, 1.0));
		_centerLineGeodes->push_back(colonLine->getGeode());
		_transform->addChild(_centerLineGeodes->at(_centerLineGeodes->size()-1));
		_centerLineGeodes->at(_centerLineGeodes->size() - 1)->setNodeMask(0);
	}

    _illeumCoords = FileSelector::loadCenterLine(path, FileSelector::OrganEnum::ILLEUM);
	if (!_illeumCoords->empty()) {
		Line* illeumLine = new Line(_illeumCoords, osg::Vec4(UI_PURPLE_COLOR, 1.0));
		_centerLineGeodes->push_back(illeumLine->getGeode());
		_transform->addChild(_centerLineGeodes->at(_centerLineGeodes->size() - 1));
		_centerLineGeodes->at(_centerLineGeodes->size() - 1)->setNodeMask(0);
	}

	osg::Matrix m = osg::Matrix::identity();
	osg::ref_ptr<osg::Image> i = ImageLoader::LoadVolume(path, m);
	if (!i)
	{
		std::cerr << "Volume couldt be loaded" << std::endl;
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

	if (maskpath.compare("") != 0)
	{
		loadMask(maskpath, i);
		_hasMask = true;
	}
	else {
		_hasMask = false;
		
	}
 	


	_volume = new osg::Texture3D;
	_volume->setImage(i);
	_volume->setTextureSize(i->s(), i->t(), i->r());
	
	
	_volume->setFilter(osg::Texture3D::MIN_FILTER, osg::Texture3D::LINEAR);
	_volume->setFilter(osg::Texture3D::MAG_FILTER, osg::Texture3D::LINEAR);
	_volume->setInternalFormat(GL_RG16);
	_volume->setName("VOLUME");
	_volume->setResizeNonPowerOfTwoHint(false);

	//_claheVolume = new osg::Texture3D;
	//_claheVolume->setTextureSize(i->s(), i->t(), i->r());
	//osg::ref_ptr<osg::Image> bimage = new osg::Image();
	//bimage->allocateImage(_volume->getTextureWidth(), _volume->getTextureHeight(), _volume->getTextureDepth(), GL_RG, GL_UNSIGNED_SHORT);
	//_claheVolume->setImage(bimage);
	//_claheVolume->setFilter(osg::Texture3D::MIN_FILTER, osg::Texture3D::NEAREST);
	//_claheVolume->setFilter(osg::Texture3D::MAG_FILTER, osg::Texture3D::NEAREST);
	//_claheVolume->setInternalFormat(GL_RG16);
	//_claheVolume->setName("CLAHEVOLUME");
	//_claheVolume->setResizeNonPowerOfTwoHint(false);

	//OSG_NOTICE << "Volume texture size: " << (int)(_volume->getTextureWidth()) << ", " << (int)(_volume->getTextureHeight()) << ", " << (int)(_volume->getTextureDepth()) << std::endl;


	_computeNode->setComputeGroups((i->s() + 7) / 8, (i->t() + 7) / 8, (i->r() + 7) / 8);
	_computeUniforms["TexelSize"]->set(osg::Vec3(1.0f / (float)i->s(), 1.0f / (float)i->t(), 1.0f / (float)i->r()));
	osg::StateSet* states = _computeNode->getOrCreateStateSet();
	states->setTextureAttribute(0, _volume, osg::StateAttribute::ON);
	states->setTextureMode(0, GL_TEXTURE_3D, osg::StateAttribute::ON);
	/*states->setTextureAttribute(5, _claheVolume, osg::StateAttribute::ON);
	states->setTextureMode(5, GL_TEXTURE_3D, osg::StateAttribute::ON);*/
	////////////////////////////////////////clahe/////////////////////////
	_volDims = osg::Vec3i(i->s(), i->t(), i->r());
	_volArea = _volDims.x() * _volDims.y() * _volDims.z();
	//_sizeSB = osg::Vec3i(_volDims.x() / _numSB_3D.x(), _volDims.y() / _numSB_3D.y(), _volDims.z() / _numSB_3D.z());
	//float tempClipValue = 1.1f * (_sizeSB.x() * _sizeSB.y() * _sizeSB.z()) / _numGrayVals;
	//_minClipValue = unsigned int(tempClipValue + 0.5f);
	
	
	//Set dirty on all graphics contexts
	std::vector<osg::Camera*> cameras = std::vector<osg::Camera*>();/////////////uncomment
	cvr::CVRViewer::instance()->getCameras(cameras);
	for (int i = 0; i < cameras.size(); ++i)
	{
		setDirty(cameras[i]->getGraphicsContext());
	}


	precompute();
}


osg::ref_ptr<osg::AtomicCounterBufferBinding> VolumeGroup::precompMinMax() {
	osg::ref_ptr<osg::Program> prog = new osg::Program;
	prog->addShader(new osg::Shader(osg::Shader::COMPUTE, _minMaxShader));

	sourceNode = new osg::DispatchCompute((_volDims.x() + 7) / 8, (_volDims.y() + 7) / 8, (_volDims.z() + 7) / 8);
	sourceNode->getOrCreateStateSet()->setAttributeAndModes(prog.get());
	sourceNode->setDataVariance(osg::Object::DYNAMIC);

	MinMaxCallback* accallback = new MinMaxCallback(this);
	sourceNode->setDrawCallback(accallback);
	osg::StateSet* states = sourceNode->getOrCreateStateSet();
	states->setTextureAttribute(0, _volume, osg::StateAttribute::ON);
	states->setTextureMode(0, GL_TEXTURE_3D, osg::StateAttribute::ON);
	states->setRenderBinDetails(1, "RenderBin");
	states->addUniform(new osg::Uniform("VolumeDims", _volDims.x(), _volDims.y(), _volDims.z()));

	this->addChild(sourceNode);

	return setupMinmaxSSBO();
}

osg::ref_ptr<osg::AtomicCounterBufferBinding> VolumeGroup::setupMinmaxSSBO() {
	osg::ref_ptr<osg::UIntArray> dati = new osg::UIntArray;
	dati->push_back(0);//#pixels
	dati->push_back(INT_MAX);//Min
	dati->push_back(0);//Max

	osg::ref_ptr<osg::BufferObject> acbo = new osg::AtomicCounterBufferObject;
	acbo->setBufferData(0, dati);
	osg::ref_ptr<osg::AtomicCounterBufferBinding> acbb = new osg::AtomicCounterBufferBinding(1, acbo->getBufferData(0), 0, sizeof(GLuint) * 3);
	dati.release();
 	acbo.release();

	((MinMaxCallback*)sourceNode->getDrawCallback())->_acbb = acbb;
	sourceNode->getOrCreateStateSet()->setAttributeAndModes(acbb, osg::StateAttribute::ON);

	sourceNode->getOrCreateStateSet()->addUniform(new osg::Uniform("numBins", _numGrayVals));

	return acbb;
 }



std::pair< osg::ref_ptr<osg::ShaderStorageBufferBinding>, osg::ref_ptr<osg::ShaderStorageBufferBinding>> VolumeGroup::precompHist() {

	osg::ref_ptr<osg::Program> prog2 = new osg::Program;
	osg::Shader* shader = new osg::Shader(osg::Shader::COMPUTE, _histShader);
	prog2->addShader(shader);
	

	_histNode = new osg::DispatchCompute((_volDims.x() + 7) / 8, (_volDims.y() + 7) / 8, (_volDims.z() + 7) / 8);
	_histNode->getOrCreateStateSet()->setAttributeAndModes(prog2.get());
	_histNode->setDataVariance(osg::Object::DYNAMIC);

	CLAHEHistCallback* shaderStorageCallback = new CLAHEHistCallback(this);
	_histNode->setDrawCallback(shaderStorageCallback);

	osg::StateSet* states = _histNode->getOrCreateStateSet();
	states->setRenderBinDetails(2, "RenderBin");
	states->setTextureAttribute(0, _volume, osg::StateAttribute::ON);
	states->setTextureMode(0, GL_TEXTURE_3D, osg::StateAttribute::ON);
	
	states->addUniform(new osg::Uniform("VolumeDims", _volDims.x(), _volDims.y(), _volDims.z()));
	states->addUniform(new osg::Uniform("numSB", _numSB_3D.x(), _numSB_3D.y(), _numSB_3D.z()));
 
	this->addChild(_histNode);
	return setupHistSSBO();
}

std::pair< osg::ref_ptr<osg::ShaderStorageBufferBinding>, osg::ref_ptr<osg::ShaderStorageBufferBinding>> VolumeGroup::setupHistSSBO() {
	osg::ref_ptr<osg::UIntArray> hist = new osg::UIntArray(_histSize);
 	osg::ref_ptr<osg::UIntArray> histMaxVals = new osg::UIntArray(_numHist);


	osg::ref_ptr<osg::ShaderStorageBufferObject> histBuffer = new osg::ShaderStorageBufferObject;
	osg::ref_ptr<osg::ShaderStorageBufferObject> histMaxBuffer = new osg::ShaderStorageBufferObject;

	hist->setBufferObject(histBuffer);
	histMaxVals->setBufferObject(histMaxBuffer);

	osg::ref_ptr<osg::ShaderStorageBufferBinding> ssbbHist = new osg::ShaderStorageBufferBinding(1, histBuffer->getBufferData(0), 0, sizeof(GLuint) * _histSize);
	hist.release();
	histBuffer.release();
	osg::ref_ptr<osg::ShaderStorageBufferBinding> ssbbHistMax = new osg::ShaderStorageBufferBinding(2, histMaxBuffer->getBufferData(0), 0, sizeof(GLuint) * _numHist);
	histMaxVals.release();
	histMaxBuffer.release();

	_histNode->getOrCreateStateSet()->setAttributeAndModes(ssbbHist, osg::StateAttribute::ON);
	_histNode->getOrCreateStateSet()->setAttributeAndModes(ssbbHistMax, osg::StateAttribute::ON);
	_histNode->getOrCreateStateSet()->addUniform(new osg::Uniform("NUM_OUT_BINS", _numGrayVals)); 
	
	((CLAHEHistCallback*)_histNode->getDrawCallback())->_ssbb = ssbbHist;
	((CLAHEHistCallback*)_histNode->getDrawCallback())->_buffersize = _histSize;
	((CLAHEHistCallback*)_histNode->getDrawCallback())->_ssbb2 = ssbbHistMax;
	((CLAHEHistCallback*)_histNode->getDrawCallback())->_buffersize2 = _numHist;

	std::pair< t_ssbb, t_ssbb> ssbbs;
	ssbbs.first = ssbbHist;
	ssbbs.second = ssbbHistMax;

	return ssbbs;
}

osg::ref_ptr<osg::ShaderStorageBufferBinding> VolumeGroup::precompExcess(t_ssbb ssbbHist, t_ssbb ssbbHistMax) {
	////////////////////////////////////////////////////////////////////////////
	// Calculate the excess pixels based on the clipLimit

	// buffer for the pixels to re-distribute
	
	// calculate the minClipValue

	//Set up Compute Shader
	osg::ref_ptr<osg::Program> excessShader = new osg::Program;
	excessShader->addShader(new osg::Shader(osg::Shader::COMPUTE, _excessShader));

	/*int width = 4096;
	int count = (_histSize + 63) / 64;
	GLuint dispatchWidth = count / (width*width);
	GLuint dispatchHeight = (count / width) % width;
	GLuint dispatchDepth = count % width;*/
	_excessNode = new osg::DispatchCompute(_histSize+63/64, 1, 1);	
	//_excessNode = new osg::DispatchCompute(dispatchWidth, dispatchHeight, dispatchDepth);
	
	_excessNode->getOrCreateStateSet()->setAttributeAndModes(excessShader.get());	
	_excessNode->setDataVariance(osg::Object::DYNAMIC);

	osg::StateSet* states = _excessNode->getOrCreateStateSet();
	states->setRenderBinDetails(3, "RenderBin");
 	/*states->addUniform(new osg::Uniform("clipLimit", _clipLimit3D));
	states->addUniform(new osg::Uniform("minClipValue", _minClipValue));*/
	states->addUniform(new osg::Uniform("blocksDims", _numSB_3D.x(), _numSB_3D.y(), _numSB_3D.z()));

	ExcessSSB* shaderStorageCallback = new ExcessSSB(this);
	_excessNode->setDrawCallback(shaderStorageCallback);
		
	this->addChild(_excessNode);

	return setupExcessSSBO(ssbbHist, ssbbHistMax);
}

osg::ref_ptr<osg::ShaderStorageBufferBinding> VolumeGroup::setupExcessSSBO(t_ssbb ssbbHist, t_ssbb ssbbHistMax) {
	osg::ref_ptr<osg::UIntArray> excess = new osg::UIntArray(_numHist);
	osg::ref_ptr<osg::ShaderStorageBufferObject> excessBuffer = new osg::ShaderStorageBufferObject;
	excess->setBufferObject(excessBuffer);
	osg::ref_ptr<osg::ShaderStorageBufferBinding> ssbbExcess = new osg::ShaderStorageBufferBinding(3, excessBuffer->getBufferData(0), 0, sizeof(GLuint) * _numHist);
	excess.release();
	excessBuffer.release();

	/*int width = 4096;
	int count = (_histSize + 63) / 64;
	GLuint dispatchWidth = count / (width * width);
	GLuint dispatchHeight = (count / width) % width;
	GLuint dispatchDepth = count % width;
	_excessNode->setComputeGroups(dispatchWidth, dispatchHeight, dispatchDepth);*/

	_excessNode->getOrCreateStateSet()->setAttributeAndModes(ssbbExcess, osg::StateAttribute::ON);
	_excessNode->getOrCreateStateSet()->setAttributeAndModes(ssbbHist, osg::StateAttribute::ON);
	_excessNode->getOrCreateStateSet()->setAttributeAndModes(ssbbHistMax, osg::StateAttribute::ON);
	_excessNode->getOrCreateStateSet()->addUniform(new osg::Uniform("NUM_BINS", _numGrayVals));
	_excessNode->getOrCreateStateSet()->addUniform(new osg::Uniform("clipLimit", _clipLimit3D));
	
	((ExcessSSB*)_excessNode->getDrawCallback())->_buffersize = _numHist;
	((ExcessSSB*)_excessNode->getDrawCallback())->_ssbb = ssbbExcess;

	return ssbbExcess;
}
 
void VolumeGroup::precompHistClip(t_ssbb ssbbHist, t_ssbb ssbbHistMax, t_ssbb ssbbExcess, t_acbb acbbminmax) {
	

	osg::ref_ptr<osg::Program> clipShader = new osg::Program;
	osg::Shader* shader = new osg::Shader(osg::Shader::COMPUTE, _clipShader);
	clipShader->addShader(shader);
	
	int width = 4096;
	int count = (_histSize + 63) / 64;
	GLuint dispatchWidth = count / (width * width);
	GLuint dispatchHeight = (count / width) % width;
	GLuint dispatchDepth = count % width;
	_clipHist1Node = new osg::DispatchCompute(dispatchWidth, dispatchHeight, dispatchDepth);


	_clipHist1Node->getOrCreateStateSet()->setAttributeAndModes(clipShader.get());

	_clipHist1Node->setDataVariance(osg::Object::DYNAMIC);

	osg::StateSet* states = _clipHist1Node->getOrCreateStateSet();
	states->setRenderBinDetails(4, "RenderBin");
	/*states->addUniform(new osg::Uniform("clipLimit", _clipLimit3D));*/
	states->addUniform(new osg::Uniform("minClipValue", _minClipValue));
	
	//Clipshader2 Used in Callback
	osg::ref_ptr<osg::Program> clipShader2Prog = new osg::Program;
	osg::ref_ptr<osg::Shader> clipShader2 = new osg::Shader(osg::Shader::COMPUTE, _clipShader2);
	clipShader2Prog->addShader(clipShader2);
	
	
	Clip1SSB* shaderStorageCallback = new Clip1SSB(this);
	shaderStorageCallback->_clipshader2 = clipShader2Prog.get();

	_clipHist1Node->setDrawCallback(shaderStorageCallback);

	setupClip(ssbbHist, ssbbHistMax, ssbbExcess, acbbminmax);

	this->addChild(_clipHist1Node);

}

void VolumeGroup::setupClip(t_ssbb ssbbHist, t_ssbb ssbbHistMax, t_ssbb ssbbExcess, t_acbb acbbminmax) {
	int width = 4096;
	int count = (_histSize + 63) / 64;
	GLuint dispatchWidth = count / (width * width);//TODO : look into this
	GLuint dispatchHeight = (count / width) % width;
	GLuint dispatchDepth = count % width;
	_clipHist1Node->setComputeGroups(dispatchWidth, dispatchHeight, dispatchDepth);


 	_clipHist1Node->getOrCreateStateSet()->setAttributeAndModes(ssbbExcess, osg::StateAttribute::ON);
	_clipHist1Node->getOrCreateStateSet()->setAttributeAndModes(ssbbHist, osg::StateAttribute::ON);
	_clipHist1Node->getOrCreateStateSet()->setAttributeAndModes(ssbbHistMax, osg::StateAttribute::ON);
	_clipHist1Node->getOrCreateStateSet()->addUniform(new osg::Uniform("NUM_BINS", _numGrayVals));
	_clipHist1Node->getOrCreateStateSet()->addUniform(new osg::Uniform("clipLimit", _clipLimit3D));


	((Clip1SSB*)_clipHist1Node->getDrawCallback())->_buffersize = _numHist;
	((Clip1SSB*)_clipHist1Node->getDrawCallback())->_ssbbExcess = ssbbExcess;
	((Clip1SSB*)_clipHist1Node->getDrawCallback())->_ssbbHist = ssbbHist;
	((Clip1SSB*)_clipHist1Node->getDrawCallback())->_ssbbHistMax = ssbbHistMax;
	((Clip1SSB*)_clipHist1Node->getDrawCallback())->numPixels = -1;
	((Clip1SSB*)_clipHist1Node->getDrawCallback())->_acbbminMax = acbbminmax;
	((Clip1SSB*)_clipHist1Node->getDrawCallback())->volDims = _volDims;
	((Clip1SSB*)_clipHist1Node->getDrawCallback())->_numGrayVals = _numGrayVals;
	((Clip1SSB*)_clipHist1Node->getDrawCallback())->_clipLimit = _clipLimit3D;
}

void VolumeGroup::precompLerp(t_ssbb ssbbHist) {
	

	osg::ref_ptr<osg::Program> lerpProgram = new osg::Program;
	lerpProgram->addShader(new osg::Shader(osg::Shader::COMPUTE, _lerpShader));

	_lerpNode = new osg::DispatchCompute((_volDims.x() + 7) / 8, (_volDims.y() + 7) / 8, (_volDims.z() + 7) / 8);
	osg::StateSet* states = _lerpNode->getOrCreateStateSet();
	

	states->setAttributeAndModes(lerpProgram.get());
	_lerpNode->setDataVariance(osg::Object::DYNAMIC);
	states->setTextureAttribute(0, _volume, osg::StateAttribute::ON);
	states->setTextureMode(0, GL_TEXTURE_3D, osg::StateAttribute::ON);
	states->setTextureAttribute(5, _claheVolume, osg::StateAttribute::ON);
	states->setTextureMode(5, GL_TEXTURE_3D, osg::StateAttribute::ON);
	states->setRenderBinDetails(5, "RenderBin");
	states->addUniform(new osg::Uniform("numSB", _numSB_3D.x(), _numSB_3D.y(), _numSB_3D.z()));

	LerpSSB* shaderStorageCallback = new LerpSSB(this);
	shaderStorageCallback->_claheDirty = ((ComputeDrawCallback*)_computeNode->getDrawCallback())->_claheDirty;

	_lerpNode->setDrawCallback(shaderStorageCallback);

	setupLerp(ssbbHist);
	
	this->addChild(_lerpNode);
	
}

void VolumeGroup::setupLerp(t_ssbb ssbbHist) {
	osg::StateSet* states = _lerpNode->getOrCreateStateSet();
	states->setAttributeAndModes(ssbbHist, osg::StateAttribute::ON);
	states->addUniform(new osg::Uniform("NUM_IN_BINS", _numGrayVals));
	states->addUniform(new osg::Uniform("NUM_OUT_BINS", _numGrayVals));

	((LerpSSB*)_lerpNode->getDrawCallback())->_buffersize = _histSize;
	((LerpSSB*)_lerpNode->getDrawCallback())->_ssbb = ssbbHist;
}

void VolumeGroup::genClahe() {
	if (!_clahePrecomped) {
		_claheVolume = new osg::Texture3D;
		_claheVolume->setTextureSize(_volume->getTextureWidth(), _volume->getTextureHeight(), _volume->getTextureDepth());
		osg::ref_ptr<osg::Image> bimage = new osg::Image();
		bimage->allocateImage(_volume->getTextureWidth(), _volume->getTextureHeight(), _volume->getTextureDepth(), GL_RG, GL_UNSIGNED_SHORT);
		_claheVolume->setImage(bimage);
		_claheVolume->setFilter(osg::Texture3D::MIN_FILTER, osg::Texture3D::NEAREST);
		_claheVolume->setFilter(osg::Texture3D::MAG_FILTER, osg::Texture3D::NEAREST);
		_claheVolume->setInternalFormat(GL_RG16);
		_claheVolume->setName("CLAHEVOLUME");
		_claheVolume->setResizeNonPowerOfTwoHint(false);

		osg::StateSet* states = _computeNode->getOrCreateStateSet();
		states->setTextureAttribute(5, _claheVolume, osg::StateAttribute::ON);
		states->setTextureMode(5, GL_TEXTURE_3D, osg::StateAttribute::ON);

		//////////////////////////////////////clahe/////////////////////////
		//_volDims = osg::Vec3i(_volume->getTextureWidth(), _volume->getTextureHeight(), _volume->getTextureDepth());
		_sizeSB = osg::Vec3i(_volDims.x() / _numSB_3D.x(), _volDims.y() / _numSB_3D.y(), _volDims.z() / _numSB_3D.z());
		float tempClipValue = 1.1f * (_sizeSB.x() * _sizeSB.y() * _sizeSB.z()) / _numGrayVals;
		_minClipValue = unsigned int(tempClipValue + 0.5f);



		//////////////////////MinMax//////////////
		auto acbbMinMax = precompMinMax();
		//////////////////////////Hist///////////////////////////
		auto ssbbPair = precompHist();
		//////////////////////////Excess////////////////////////
		auto ssbbexcess = precompExcess(ssbbPair.first, ssbbPair.second);
		///////////////////////////HistClip///////////////////////////
		precompHistClip(ssbbPair.first, ssbbPair.second, ssbbexcess, acbbMinMax);
		///////////////////////////Lerp///////////////////////
		precompLerp(ssbbPair.first);

		//release refptrs
		acbbMinMax.release();
		ssbbPair.first.release();
		ssbbPair.second.release();
		ssbbexcess.release();


		osg::ref_ptr<osg::BindImageTexture> imagbinding = new osg::BindImageTexture(5, _claheVolume, osg::BindImageTexture::READ_WRITE, GL_RG16, 0, GL_TRUE);
		states->setAttributeAndModes(imagbinding);
		///////////////////////clahe///////////////////
		states = sourceNode->getOrCreateStateSet();
		imagbinding = new osg::BindImageTexture(0, _volume, osg::BindImageTexture::READ_ONLY, GL_RG16, 0, GL_TRUE);
		states->setAttributeAndModes(imagbinding);

		states = _histNode->getOrCreateStateSet();
		imagbinding = new osg::BindImageTexture(0, _volume, osg::BindImageTexture::READ_ONLY, GL_RG16, 0, GL_TRUE);
		states->setAttributeAndModes(imagbinding);

		states = _lerpNode->getOrCreateStateSet();
		imagbinding = new osg::BindImageTexture(0, _volume, osg::BindImageTexture::READ_ONLY, GL_RG16, 0, GL_TRUE);
		states->setAttributeAndModes(imagbinding);
		imagbinding = new osg::BindImageTexture(5, _claheVolume, osg::BindImageTexture::READ_WRITE, GL_RG16, 0, GL_TRUE);
		states->setAttributeAndModes(imagbinding);

	



		_clahePrecomped = true;
	}


	((MinMaxCallback*)sourceNode->getDrawCallback())->_acbb.release();
	auto minmaxSSBB = setupMinmaxSSBO();

	((CLAHEHistCallback*)_histNode->getDrawCallback())->_ssbb.release();
	((CLAHEHistCallback*)_histNode->getDrawCallback())->_ssbb2.release();
	auto ssbbs = setupHistSSBO();

	((ExcessSSB*)_excessNode->getDrawCallback())->_ssbb.release();
 	auto excessSSBB = setupExcessSSBO(ssbbs.first, ssbbs.second);

	((Clip1SSB*)_clipHist1Node->getDrawCallback())->_acbbminMax.release();
	((Clip1SSB*)_clipHist1Node->getDrawCallback())->_ssbbExcess.release();
	((Clip1SSB*)_clipHist1Node->getDrawCallback())->_ssbbHist.release();
	((Clip1SSB*)_clipHist1Node->getDrawCallback())->_ssbbHistMax.release();
	setupClip(ssbbs.first, ssbbs.second, excessSSBB, minmaxSSBB);

	((LerpSSB*)_lerpNode->getDrawCallback())->_ssbb.release();
	setupLerp(ssbbs.first);

	//dirty callbacks
	((MinMaxCallback*)sourceNode->getDrawCallback())->stop[0] = 0;
	((CLAHEHistCallback*)_histNode->getDrawCallback())->stop[0] = 0;
	((ExcessSSB*)_excessNode->getDrawCallback())->stop[0] = 0;
	((Clip1SSB*)_clipHist1Node->getDrawCallback())->stop[0] = 0;
	((LerpSSB*)_lerpNode->getDrawCallback())->stop[0] = 0;

	((LerpSSB*)_lerpNode->getDrawCallback())->_claheDirty[0] = 1;




	
	//dirty vol
	_claheAvailable = true;
	this->setDirtyAll();
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
		std::string maskpath = path + "/" + std::to_string(depth - (i + 1)) + ".png";
		osg::ref_ptr<osg::Image> mask = osgDB::readImageFile(maskpath);
		mask->flipVertical();
		unsigned int bytesize = mask->getPixelSizeInBits() / 8;
		//throw bytesize;
		unsigned char* maskData = mask->data();

		uint16_t* slice = volumeData + (int)2 * i * width * height;
		for (unsigned int y = 0; y < height; ++y)
		{
			for (unsigned int x = 0; x < width; ++x)
			{
				unsigned int volpixel = 2 * (x + y * width);
				unsigned int maskpixel = bytesize * (x + y * width);
				//upper 8 bits are green, bottom 8 are red. use 1-hot encoding

				if (bytesize <= 1 && (uint8_t)(maskData[maskpixel]) != 0) //Binary mask, only care about colon
				{
					slice[volpixel + 1] = 4;
				}
				else //multi-organ mask
				{
					slice[volpixel + 1] = (uint16_t)(maskData[maskpixel]) + (uint16_t)(maskData[maskpixel + 1]) * 256;
				}
				if (slice[volpixel + 1] != 0)
				{
					//std::cout << slice[volpixel + 1] << std::endl;
				}
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
		//bimage->allocateImage(_volume->getTextureWidth(), _volume->getTextureHeight(), _volume->getTextureDepth(), GL_RGBA, GL_UNSIGNED_BYTE);
		bimage->allocateImage(_volume->getTextureWidth(), _volume->getTextureHeight(), _volume->getTextureDepth(), GL_RGBA, GL_UNSIGNED_INT);
		_baked->setImage(bimage);
		_baked->setTextureSize(_volume->getTextureWidth(), _volume->getTextureHeight(), _volume->getTextureDepth());
		_baked->setFilter(osg::Texture3D::MIN_FILTER, osg::Texture3D::LINEAR);
		_baked->setFilter(osg::Texture3D::MAG_FILTER, osg::Texture3D::LINEAR);
		//_baked->setInternalFormat(GL_RGBA8);
		_baked->setInternalFormat(GL_RGBA16);
		_baked->setResizeNonPowerOfTwoHint(false);
		
		_baked->setName("BAKED");
		_baked->setTextureSize(_volume->getTextureWidth(), _volume->getTextureHeight(), _volume->getTextureDepth());

		osg::StateSet* states = _computeNode->getOrCreateStateSet();
		states->setTextureAttribute(1, _baked, osg::StateAttribute::ON);
		states->setTextureMode(1, GL_TEXTURE_3D, osg::StateAttribute::ON);
		osg::ref_ptr<osg::BindImageTexture> imagbinding = new osg::BindImageTexture(0, _volume, osg::BindImageTexture::READ_ONLY, GL_RG16, 0, GL_TRUE);
		osg::ref_ptr<osg::BindImageTexture> imagbinding2 = new osg::BindImageTexture(1, _baked, osg::BindImageTexture::READ_WRITE, GL_RGBA16, 0, GL_TRUE);
		//osg::ref_ptr<osg::BindImageTexture> imagbinding25 = new osg::BindImageTexture(5, _claheVolume, osg::BindImageTexture::READ_WRITE, GL_RG16, 0, GL_TRUE);

		
		states->setAttributeAndModes(imagbinding);
		states->setAttributeAndModes(imagbinding2);
		//states->setAttributeAndModes(imagbinding25);
		/////////////////////////clahe///////////////////
		//states = sourceNode->getOrCreateStateSet();
		//osg::ref_ptr<osg::BindImageTexture> imagbinding3 = new osg::BindImageTexture(0, _volume, osg::BindImageTexture::READ_ONLY, GL_RG16, 0, GL_TRUE);
		//states->setAttributeAndModes(imagbinding3);
		//
		//states = _histNode->getOrCreateStateSet();
		//osg::ref_ptr<osg::BindImageTexture> imagbinding4 = new osg::BindImageTexture(0, _volume, osg::BindImageTexture::READ_ONLY, GL_RG16, 0, GL_TRUE);
		//states->setAttributeAndModes(imagbinding4);

		//states = _lerpNode->getOrCreateStateSet();
		//osg::ref_ptr<osg::BindImageTexture> imagbinding5 = new osg::BindImageTexture(0, _volume, osg::BindImageTexture::READ_ONLY, GL_RG16, 0, GL_TRUE);
		//osg::ref_ptr<osg::BindImageTexture> imagbinding6 = new osg::BindImageTexture(5, _claheVolume, osg::BindImageTexture::READ_WRITE, GL_RG16, 0, GL_TRUE);
		//states->setAttributeAndModes(imagbinding5);
		//states->setAttributeAndModes(imagbinding6);

	


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

void VolumeGroup::precompTotalHistogram() {

	//Hist Buffer
	unsigned int numBins = 255;
	osg::ref_ptr<osg::UIntArray> hist = new osg::UIntArray(numBins);
	osg::ref_ptr<osg::ShaderStorageBufferObject> histBuffer = new osg::ShaderStorageBufferObject;	
	hist->setBufferObject(histBuffer);
	osg::ref_ptr<osg::ShaderStorageBufferBinding> ssbbHist = new osg::ShaderStorageBufferBinding(6, histBuffer->getBufferData(0), 0, sizeof(GLuint) * numBins);
	hist.release();
	histBuffer.release();
 
	//Max Hist Value
	osg::ref_ptr<osg::UIntArray> dati = new osg::UIntArray(1);
	osg::ref_ptr<osg::BufferObject> acbo = new osg::AtomicCounterBufferObject;
	acbo->setBufferData(0, dati);
	osg::ref_ptr<osg::AtomicCounterBufferBinding> acbb = new osg::AtomicCounterBufferBinding(7, acbo->getBufferData(0), 0, sizeof(GLuint));
	dati.release();
	acbo.release();


	

	osg::ref_ptr<osg::Program> prog2 = new osg::Program;
	osg::Shader* shader = new osg::Shader(osg::Shader::COMPUTE, _totalHistShader);
	prog2->addShader(shader);


	_totalHistNode = new osg::DispatchCompute((_volDims.x() + 7) / 8, (_volDims.y() + 7) / 8, (_volDims.z() + 7) / 8);
	_totalHistNode->getOrCreateStateSet()->setAttributeAndModes(prog2.get());
	_totalHistNode->getOrCreateStateSet()->setAttributeAndModes(ssbbHist, osg::StateAttribute::ON);
	_totalHistNode->getOrCreateStateSet()->setAttributeAndModes(acbb, osg::StateAttribute::ON);
	_totalHistNode->setDataVariance(osg::Object::DYNAMIC);

	TotalHistCallback* shaderStorageCallback = new TotalHistCallback(this);
	_totalHistNode->setDrawCallback(shaderStorageCallback);
	shaderStorageCallback->_ssbb = ssbbHist;
	shaderStorageCallback->_acbb = acbb;
	/*shaderStorageCallback->_buffersize = numBins;*/
  


	osg::StateSet* states = _totalHistNode->getOrCreateStateSet();
	states->setRenderBinDetails(9, "RenderBin");
	/*states->setTextureAttribute(5, _claheVolume, osg::StateAttribute::ON);
	states->setTextureMode(5, GL_TEXTURE_3D, osg::StateAttribute::ON);*/
	/*states->setTextureAttribute(1, _baked, osg::StateAttribute::ON);
	states->setTextureMode(1, GL_TEXTURE_3D, osg::StateAttribute::ON);*/
	states->setTextureAttribute(0, _volume, osg::StateAttribute::ON);
	states->setTextureMode(0, GL_TEXTURE_3D, osg::StateAttribute::ON);

	states->addUniform(new osg::Uniform("VolumeDims", _volDims.x(), _volDims.y(), _volDims.z()));
 	states->addUniform(new osg::Uniform("NUM_BINS", numBins));

	this->addChild(_totalHistNode);

	

}

void VolumeGroup::precompMarchingCubes() {

	//TriangleIDs buffer
	
 	//std::cout << "voldims: " << _volDims.x() << "   " << _volDims.y() << "   " << _volDims.z() << std::endl;
	//Vec Count Value
	osg::ref_ptr<osg::UIntArray> dati = new osg::UIntArray;
	dati->push_back(0);
	osg::ref_ptr<osg::BufferObject> acbo = new osg::AtomicCounterBufferObject;
	acbo->setBufferData(0, dati);
	osg::ref_ptr<osg::AtomicCounterBufferBinding> acbb = new osg::AtomicCounterBufferBinding(4, acbo->getBufferData(0), 0, sizeof(GLuint));
	dati.release();
	acbo.release();

	//ConfigLookup buffer
	osg::ref_ptr<osg::IntArray> configLUTArray = new osg::IntArray(CONFIGLUTSIZE);
	memcpy((void*)&(configLUTArray->front()), &_luts.triangulation[0], CONFIGLUTSIZE *sizeof(int));
	osg::ref_ptr<osg::ShaderStorageBufferObject> ssboCL = new osg::ShaderStorageBufferObject;	
	configLUTArray->setBufferObject(ssboCL);
	osg::ref_ptr<osg::ShaderStorageBufferBinding> ssbbCL = new osg::ShaderStorageBufferBinding(2, ssboCL->getBufferData(0), 0, sizeof(GLint) * 4096);
	configLUTArray.release();
	ssboCL.release();

	//TriangleVertices buffer
	int newStructSize = _volDims.x() / MCFACTOR * _volDims.y() / MCFACTOR * _volDims.z() / MCFACTOR;
	unsigned int triangleStructMaxSize = newStructSize*45;//Maximum_Index(max edge index(15)*Vertice_Count(3) + Float_Count(9) = 51
	_va = new osg::Vec3Array(triangleStructMaxSize/3u);
	for (int i = 0; i < triangleStructMaxSize/3; i++) {
		_va->at(i).x() = std::numeric_limits<float>::max();
		_va->at(i).y() = std::numeric_limits<float>::max();
		_va->at(i).z() = std::numeric_limits<float>::max();
	}

	
	geo = new osg::Geometry;	
	geo->setUseVertexBufferObjects(true);

	osg::VertexBufferObject* vbo = new osg::VertexBufferObject;
	vbo->setArray(0, _va);
	osg::Vec3Array* va = new osg::Vec3Array;
	va->setVertexBufferObject(vbo);

	geo->setVertexArray(_va);
 
	//WEIRD SHAPE
	//osg::ref_ptr<osg::ShaderStorageBufferBinding> ssbbTV = new osg::ShaderStorageBufferBinding(3,geo->getVertexArray()->getBufferObject()->getBufferData(0), 0, sizeof(GLfloat) * triangleStructMaxSize);
	osg::ref_ptr<osg::ShaderStorageBufferBinding> ssbbTV = new osg::ShaderStorageBufferBinding(3, vbo->getArray(0), 0, sizeof(GLfloat) * triangleStructMaxSize);
	
	
	
	//verticesArray.release();
	

	osg::ref_ptr<osg::Program> prog = new osg::Program;
	osg::Shader* shader = new osg::Shader(osg::Shader::COMPUTE, _marchingCubesShader);
	prog->addShader(shader);


	_marchingCubeNode = new osg::DispatchCompute((_volDims.x() + 7) / 8, (_volDims.y() + 7) / 8, (_volDims.z() + 7) / 8);
	_marchingCubeNode->getOrCreateStateSet()->setAttributeAndModes(prog.get());
 	_marchingCubeNode->getOrCreateStateSet()->setAttributeAndModes(ssbbCL, osg::StateAttribute::ON);
	_marchingCubeNode->getOrCreateStateSet()->setAttributeAndModes(ssbbTV, osg::StateAttribute::ON);
	_marchingCubeNode->getOrCreateStateSet()->setAttributeAndModes(acbb, osg::StateAttribute::ON);
	_marchingCubeNode->setDataVariance(osg::Object::DYNAMIC);

	MarchingCubeCallback* shaderStorageCallback = new MarchingCubeCallback(this);
	_marchingCubeNode->setDrawCallback(shaderStorageCallback);
	shaderStorageCallback->_ssbb = ssbbTV;
 	shaderStorageCallback->_buffersize = triangleStructMaxSize;
 	shaderStorageCallback->_volDims = _volDims;
 	shaderStorageCallback->_geomToPass = geo;
 	shaderStorageCallback->_acbb = acbb;
 	shaderStorageCallback->_VA = _va;

	
  


	osg::StateSet* states = _marchingCubeNode->getOrCreateStateSet();
	states->setRenderBinDetails(8, "RenderBin");
	/*states->setTextureAttribute(5, _claheVolume, osg::StateAttribute::ON);
	states->setTextureMode(5, GL_TEXTURE_3D, osg::StateAttribute::ON);*/
	/*states->setTextureAttribute(1, _baked, osg::StateAttribute::ON);
	states->setTextureMode(1, GL_TEXTURE_3D, osg::StateAttribute::ON);*/

	states->setTextureAttribute(0, _volume, osg::StateAttribute::ON);
	states->setTextureMode(0, GL_TEXTURE_3D, osg::StateAttribute::ON);

	states->addUniform(new osg::Uniform("VolumeDims", _volDims.x(), _volDims.y(), _volDims.z()));
 	states->addUniform(new osg::Uniform("IsoLevel",ISOVALUE));
 	states->addUniform(new osg::Uniform("McFactor", MCFACTOR));

	this->addChild(_marchingCubeNode);

 	

}

osg::ref_ptr<osg::Geometry> VolumeGroup::getMCGeom() {
	return ((MarchingCubeCallback*)_marchingCubeNode->getDrawCallback())->_geomToPass;
}

unsigned int VolumeGroup::getMCVertCount() {
	return ((MarchingCubeCallback*)_marchingCubeNode->getDrawCallback())->vertexCount[0];
}

osg::ref_ptr<osg::Vec3Array> VolumeGroup::getVA() {
	return ((MarchingCubeCallback*)_marchingCubeNode->getDrawCallback())->_VA;
}

osg::ref_ptr<osg::ShaderStorageBufferBinding> VolumeGroup::getMCSSBB() {
	return ((MarchingCubeCallback*)_marchingCubeNode->getDrawCallback())->_ssbb;
}


void VolumeGroup::intializeMC() {
	if (!_mcPrecomped) {
		precompMarchingCubes();

		osg::StateSet* states = _marchingCubeNode->getOrCreateStateSet();
		//osg::ref_ptr<osg::BindImageTexture> imagbinding7 = new osg::BindImageTexture(5, _claheVolume, osg::BindImageTexture::READ_ONLY, GL_RG16, 0, GL_TRUE);
		osg::ref_ptr<osg::BindImageTexture> imagbinding8 = new osg::BindImageTexture(0, _volume, osg::BindImageTexture::READ_ONLY, GL_RG16, 0, GL_TRUE);
		//osg::ref_ptr<osg::BindImageTexture> imagbinding7 = new osg::BindImageTexture(1, _baked, osg::BindImageTexture::READ_ONLY, GL_RGBA16, 0, GL_TRUE);
		states->setAttributeAndModes(imagbinding8);

		_mcPrecomped = true;
	}
	((MarchingCubeCallback*)_marchingCubeNode->getDrawCallback())->stop[0] = 0;
	this->setDirtyAll();
	this->_UIDirty = true;
}

bool VolumeGroup::toggleMC() {
	if (_mcIsReady) {
		if (_mcrGeode == nullptr) {
			
			MarchingCubesRender* mcr = new MarchingCubesRender(_mcVertices, _volDims, getMCGeom(), getMCVertCount(), getVA(), getMCSSBB());
 
			_mcrGeode = mcr->getGeode();
		}
		if (_transform->removeChild(_mcrGeode) == false) {
			_transform->addChild(_mcrGeode);
			return true;
		}
		return false;
	}
	return false;
}

void VolumeGroup::readyMCUI() {
	_mcIsReady = true;
}

unsigned int VolumeGroup::getHistMax() {
	return ((TotalHistCallback*)_totalHistNode->getDrawCallback())->histMax[0];
}
osg::ref_ptr<osg::ShaderStorageBufferBinding> VolumeGroup::getHistBB() {
	return ((TotalHistCallback*)_totalHistNode->getDrawCallback())->_ssbb;
}

void Clip1SSB::drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
{
	if (group->isDirty(renderInfo.getCurrentCamera()->getGraphicsContext()) && stop[0] != 1)
	{

		
 		osg::ref_ptr<osg::UIntArray> uintArray = new osg::UIntArray(_buffersize);

		drawable->drawImplementation(renderInfo);
		renderInfo.getState()->get<osg::GLExtensions>()->glMemoryBarrier(GL_ALL_BARRIER_BITS);

		osg::GLBufferObject* glBufferObject = _ssbbExcess->getBufferData()->getBufferObject()->getOrCreateGLBufferObject(renderInfo.getState()->getContextID());
 
		GLint previousID = 1;
		glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &previousID);

		if ((GLuint)previousID != glBufferObject->getGLObjectID())
			glBufferObject->_extensions->glBindBuffer(GL_SHADER_STORAGE_BUFFER, glBufferObject->getGLObjectID());

		GLubyte* data = (GLubyte*)glBufferObject->_extensions->glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY_ARB);
 		if (data)
		{
			size_t size = osg::minimum<int>(_ssbbExcess->getSize(), uintArray->getTotalDataSize());
			memcpy((void*)&(uintArray->front()), data + _ssbbExcess->getOffset(), size);
			glBufferObject->_extensions->glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
		}

		if ((GLuint)previousID != glBufferObject->getGLObjectID())
			glBufferObject->_extensions->glBindBuffer(GL_SHADER_STORAGE_BUFFER, previousID);


		unsigned int value = uintArray->front();
 

 		
		uint32_t* stepSize = new uint32_t[_buffersize];
		memset(stepSize, 0, _buffersize * sizeof(uint32_t));
		bool computePass2 = false;
		for (unsigned int i = 0; i < _buffersize; i++) {
			if (uintArray->at(i) == 0) {
				stepSize[i] = 0;
			}
			else {
				stepSize[i] = std::max(_numGrayVals / uintArray->at(i), 1u);
				computePass2 = true;
			}
		}

		
		renderInfo.getState()->get<osg::GLExtensions>()->glMemoryBarrier(GL_ALL_BARRIER_BITS);
		//////////////////////computepass 2 ///////////////////
		osg::Vec3 numSB = osg::Vec3(4, 4, 2);
		
		unsigned int histSize = 32 * _numGrayVals;
	
		GLuint ssbbExcessID = _ssbbExcess->getBufferData()->getBufferObject()->getOrCreateGLBufferObject(renderInfo.getState()->getContextID())->getGLObjectID();
		GLuint histID = _ssbbHist->getBufferData()->getBufferObject()->getOrCreateGLBufferObject(renderInfo.getState()->getContextID())->getGLObjectID();
		GLuint histMaxID = _ssbbHistMax->getBufferData()->getBufferObject()->getOrCreateGLBufferObject(renderInfo.getState()->getContextID())->getGLObjectID();


		


		if (computePass2) {

			GLuint stepSizeBuffer;
			renderInfo.getState()->get<osg::GLExtensions>()->glGenBuffers(1, &stepSizeBuffer);
			renderInfo.getState()->get<osg::GLExtensions>()->glBindBuffer(GL_SHADER_STORAGE_BUFFER, stepSizeBuffer);
			renderInfo.getState()->get<osg::GLExtensions>()->glBufferData(GL_SHADER_STORAGE_BUFFER, _buffersize * sizeof(uint32_t), stepSize, GL_STREAM_READ);
			renderInfo.getState()->get<osg::GLExtensions>()->glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

			osg::State* state = renderInfo.getState();
			 
			GLint id;
			glGetIntegerv(GL_CURRENT_PROGRAM, &id);
		 
			std::cout << "test id.. " << id << std::endl;
 			//renderInfo.getState()->get<osg::GLExtensions>()->glUseProgram(0);

		 
			_clipshader2->apply(*state);


		 
			glGetIntegerv(GL_CURRENT_PROGRAM, &id);
			 
			std::cout << "test id.. " << id << std::endl;

			renderInfo.getState()->get<osg::GLExtensions>()->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, histID);
			renderInfo.getState()->get<osg::GLExtensions>()->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, histMaxID);
			renderInfo.getState()->get<osg::GLExtensions>()->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssbbExcessID);
			renderInfo.getState()->get<osg::GLExtensions>()->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, stepSizeBuffer);

			const unsigned int numOutGrayVals = _numGrayVals;
			const float clipLimit = _clipLimit;


			float tempClipValue = 1.1f * (numSB.x() * numSB.y() * numSB.z()) / numOutGrayVals;
			std::cout << "temp clip... " << tempClipValue << std::endl;
			unsigned int minClipValue = unsigned int(tempClipValue + 0.5f);



			renderInfo.getState()->get<osg::GLExtensions>()->glUniform1ui(
				renderInfo.getState()->get<osg::GLExtensions>()->glGetUniformLocation(id, "NUM_BINS"), numOutGrayVals);
			renderInfo.getState()->get<osg::GLExtensions>()->glUniform1f(
				renderInfo.getState()->get<osg::GLExtensions>()->glGetUniformLocation(id, "clipLimit"), clipLimit);
			renderInfo.getState()->get<osg::GLExtensions>()->glUniform1ui(
				renderInfo.getState()->get<osg::GLExtensions>()->glGetUniformLocation(id, "minClipValue"), minClipValue);

			///////////////DEBUGGING
			{
				osg::ref_ptr<osg::UIntArray> uintArray = new osg::UIntArray(32*_numGrayVals);
				osg::GLBufferObject* glBufferObject = _ssbbHist->getBufferData()->getBufferObject()->getOrCreateGLBufferObject(renderInfo.getState()->getContextID());
				//std::cout << glBufferObject << std::endl;

				GLint previousID = 1;
				glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &previousID);

				if ((GLuint)previousID != glBufferObject->getGLObjectID())
					glBufferObject->_extensions->glBindBuffer(GL_SHADER_STORAGE_BUFFER, glBufferObject->getGLObjectID());

				GLubyte* data = (GLubyte*)glBufferObject->_extensions->glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY_ARB);
				//std::cout << data << std::endl;
				if (data)
				{
					size_t size = osg::minimum<int>(_ssbbHist->getSize(), uintArray->getTotalDataSize());
					memcpy((void*)&(uintArray->front()), data + _ssbbHist->getOffset(), size);
					glBufferObject->_extensions->glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
				}

				if ((GLuint)previousID != glBufferObject->getGLObjectID())
					glBufferObject->_extensions->glBindBuffer(GL_SHADER_STORAGE_BUFFER, previousID);


				unsigned int value = uintArray->front();
				std::cout << "Excess after clip" << value << std::endl;

				for (int i = 0; i < 100; i++) {
					std::cout << uintArray->at(i) << std::endl;

				}


			}
			//DEBUGGING/////////////////////////////


			renderInfo.getState()->get<osg::GLExtensions>()->glDispatchCompute((GLuint)((histSize + 63) / 64), 1, 1);

 			renderInfo.getState()->get<osg::GLExtensions>()->glMemoryBarrier(GL_ALL_BARRIER_BITS);
			renderInfo.getState()->get<osg::GLExtensions>()->glUseProgram(0);

			renderInfo.getState()->get<osg::GLExtensions>()->glDeleteBuffers(1, &stepSizeBuffer);





			///////////////DEBUGGING
			{
				osg::ref_ptr<osg::UIntArray> uintArray = new osg::UIntArray(32*_numGrayVals);
				osg::GLBufferObject* glBufferObject = _ssbbHist->getBufferData()->getBufferObject()->getOrCreateGLBufferObject(renderInfo.getState()->getContextID());
				//std::cout << glBufferObject << std::endl;

				GLint previousID = 1;
				glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &previousID);

				if ((GLuint)previousID != glBufferObject->getGLObjectID())
					glBufferObject->_extensions->glBindBuffer(GL_SHADER_STORAGE_BUFFER, glBufferObject->getGLObjectID());

				GLubyte* data = (GLubyte*)glBufferObject->_extensions->glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY_ARB);
				//std::cout << data << std::endl;
				if (data)
				{
					size_t size = osg::minimum<int>(_ssbbHist->getSize(), uintArray->getTotalDataSize());
					memcpy((void*)&(uintArray->front()), data + _ssbbHist->getOffset(), size);
					glBufferObject->_extensions->glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
				}

				if ((GLuint)previousID != glBufferObject->getGLObjectID())
					glBufferObject->_extensions->glBindBuffer(GL_SHADER_STORAGE_BUFFER, previousID);


				unsigned int value = uintArray->front();
				std::cout << "Excess after clip" << value << std::endl;

				for (int i = 0; i < 100; i++) {
						std::cout << uintArray->at(i) << std::endl;

				}


			}
			//DEBUGGING/////////////////////////////
			delete[] stepSize;
		}


		////////////////////////////////////////////////////////////////////////////
	// Map the histograms 
	// - calculate the CDF for each of the histograms and store it in hist
		
		

		osg::ref_ptr<osg::UIntArray> _atomicCounterArray = new osg::UIntArray();
		_atomicCounterArray->push_back(0);
		_atomicCounterArray->push_back(0);
		_atomicCounterArray->push_back(0);
		_acbbminMax->readData(*renderInfo.getState(), *_atomicCounterArray);
		renderInfo.getState()->get<osg::GLExtensions>()->glMemoryBarrier(GL_ALL_BARRIER_BITS);
		

		


		uint32_t numPixelsSB;
		if (numPixels == -1) {
			osg::Vec3 sizeSB = osg::Vec3(volDims.x() / numSB.x(), volDims.y() / numSB.y(), volDims.z() / numSB.z());
			numPixelsSB = sizeSB.x() * sizeSB.y() * sizeSB.z();
		}
		else {
			numPixelsSB = numPixels;
		}

		uint32_t* hist = new uint32_t[histSize];
		memset(hist, 0, histSize * sizeof(uint32_t));
		GLEX()->glBindBuffer(GL_SHADER_STORAGE_BUFFER, histID);
		hist = (uint32_t*)(GLEX()->glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_WRITE_ARB));

		

		std::vector<std::thread> threads;
		for (unsigned int currHistIndex = 0; currHistIndex < _buffersize; currHistIndex++) {
			uint32_t* currHist = &hist[currHistIndex * _numGrayVals];
			threads.push_back(std::thread(mapHistogram, (uint32_t)_atomicCounterArray->at(1), (uint32_t)_atomicCounterArray->at(2), numPixelsSB, (uint32_t)_numGrayVals, currHist));
		}
		for (auto& currThread : threads) {
			currThread.join();
		}
		
		GLEX()->glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
		GLEX()->glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);




		///////////////DEBUGGING
		{
			osg::ref_ptr<osg::UIntArray> uintArray = new osg::UIntArray(_numGrayVals * 32);
			osg::GLBufferObject* glBufferObject = _ssbbHist->getBufferData()->getBufferObject()->getOrCreateGLBufferObject(renderInfo.getState()->getContextID());
			//std::cout << glBufferObject << std::endl;

			GLint previousID = 1;
			glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &previousID);

			if ((GLuint)previousID != glBufferObject->getGLObjectID())
				glBufferObject->_extensions->glBindBuffer(GL_SHADER_STORAGE_BUFFER, glBufferObject->getGLObjectID());

			GLubyte* data = (GLubyte*)glBufferObject->_extensions->glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY_ARB);
			//std::cout << data << std::endl;
			if (data)
			{
				size_t size = osg::minimum<int>(_ssbbHist->getSize(), uintArray->getTotalDataSize());
				memcpy((void*)&(uintArray->front()), data + _ssbbHist->getOffset(), size);
				glBufferObject->_extensions->glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
			}

			if ((GLuint)previousID != glBufferObject->getGLObjectID())
				glBufferObject->_extensions->glBindBuffer(GL_SHADER_STORAGE_BUFFER, previousID);


			unsigned int value = uintArray->front();
			/*std::cout << "Hist after clip" << value << std::endl;

			for (int i = 0; i < 100; i++) {
 					std::cout << uintArray->at(i) << std::endl;
				
			}*/

			
		}
		//DEBUGGING/////////////////////////////


 		stop[0] = 1;

	}


	
}