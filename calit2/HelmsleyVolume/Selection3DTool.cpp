#pragma once

#include "Selection3DTool.h"
#include "HelmsleyVolume.h"
#include <cvrConfig/ConfigManager.h>



using namespace cvr;


osg::Vec3 vecCoordTransform(osg::Vec3 vec, osg::Vec3 dims) {
	osg::Vec3 transformedVec = vec;
	transformedVec.z() = dims.z() - transformedVec.z();
	transformedVec.y() = dims.y() - transformedVec.y();

	return transformedVec;
}

osg::Matrix getObjectToWorldMatrixSelecion3D(osg::ref_ptr<osg::MatrixTransform> transform)
{
	osg::Matrix mat = osg::Matrix::identity();
	if (transform)
	{
		transform->computeLocalToWorldMatrix(mat, NULL);
	}
	return mat;
}

void Selection3DToolUpdate::operator()(osg::Node* node, osg::NodeVisitor* nv)
{
	if (_mainSO && _active) { 

		osg::Vec3 cubeWorldPos = _cubeMT->getMatrix().getTrans() * _mainSO->getObjectToWorldMatrix();
  		osg::Vec3 selectionWorldPos = _ruler->getMatrix().getTrans() * _selectionSO->getObjectToWorldMatrix();
 
		osg::Matrix cubeMat = osg::Matrix::identity();
		_cubeMT->computeWorldToLocalMatrix(cubeMat, NULL);
		osg::Vec3 cubeCoord = selectionWorldPos * _mainSO->getWorldToObjectMatrix() * cubeMat;
 		cubeCoord += osg::Vec3(.5, .5, .5);
		_currCubeCoordNorm = cubeCoord;

		cubeCoord = osg::Vec3(std::abs(_dims.x() * cubeCoord.x()), std::abs(_dims.z() * cubeCoord.y()), std::abs(_dims.y() * cubeCoord.z()));

		_selectionCenter->setElement(0, cubeCoord);

		_currCubeCoord = cubeCoord;
		
	}
	traverse(node, nv);
}

void Selection3DToolUpdate::updateVolume(osg::ref_ptr<osg::MatrixTransform> cubeMT, osg::Vec3 dims) {
	_cubeMT = cubeMT;
	_dims = dims;
}

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
	_selectionMatrixTrans = new osg::MatrixTransform();

	g->addDrawable(box);
	_selectionMatrixTrans->addChild(g);
	this->addChild(_selectionMatrixTrans);
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
	_selectionMatrixTrans->setNodeMask(0xFFFFFFFF);
 }

void Selection3DTool::deactivate()
{
	_selectionMatrixTrans->setNodeMask(0);
 }

void Selection3DTool::update()
{
	if (_locked)
		return;

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

	_scaledDims = osg::Vec3(std::abs(forward.x()), std::abs(forward.z()), std::abs(forward.y()));
	//_scaledDims = osg::Vec3(osg::Vec3(_dims.x(), _dims.y()/2, _dims.z()));
 
 
	//if(_dims == osg::Vec3(0,0,0))
		m.preMultScale(_scaledDims);
	//else
	//	m.preMultScale(osg::Vec3(std::abs(_dims.x()), std::abs(_dims.z()), std::abs(_dims.y())));



	//std::cout << "scale: " << HelmsleyVolume::instance()->printVec3OSG(forward) << std::endl;
	//m.preMultScale(osg::Vec3(length, length, length));
	//m.preMultScale(osg::Vec3(forward.z(), forward.z(), forward.z()));

	//m.postMultTranslate(_start);

	


	_selectionMatrixTrans->setMatrix(m);

	
	//std::cout << "center: " << HelmsleyVolume::instance()->printVec3OSG(_selectionCenterVector) << std::endl;
	_scaledDims = osg::Vec3(std::abs(_scaledDims.x() * _scale.x()), std::abs(_scaledDims.y() * _scale.y()), std::abs(_scaledDims.z() * _scale.z()));
	_normDims = osg::Vec3(_scaledDims.x() / _dims.x(), _scaledDims.y() / _dims.y(), _scaledDims.z() / _dims.z());
	//_normDims = osg::Vec3(_dims.x(), _dims.y(),  _dims.z());

	_selectionDims->setElement(0, _scaledDims);
	//scaleDims = osg::Vec3(std::abs(scaleDims.x()*_scale.x()), std::abs(scaleDims.y()*_scale.y()), std::abs(scaleDims.z() * _scale.z()));

	
	//std::cout << "scale: " << HelmsleyVolume::instance()->printVec3OSG(scaleDims) << std::endl;
}

float Selection3DTool::getLength()
{
	return (_end - _start).length();
}

void Selection3DTool::setRemove(bool remove) {
	if (remove) {
		deactivate();
	}
}

void Selection3DTool::setDisable(bool disable) {
	if (disable) {
		_selectionDims->setElement(0, osg::Vec3(0,0,0));
	}
	else {
		_selectionDims->setElement(0, _scaledDims);
	}
}

void Selection3DTool::setNewVolume(VolumeGroup* g) {
	osg::Vec3 dims = osg::Vec3( g->_volDims.x(), g->_volDims.z(), g->_volDims.y());
	_cubeMT = g->_transform;
	setVoldims(dims,  g->getScale());
	linkUniforms(g->_computeUniforms["SelectionsDims"], g->_computeUniforms["SelectionsCenters"]);
	updateCallbackVolume();
 }

void Selection3DTool::updateCallbackVolume() {
	_updateCallback->updateVolume(_cubeMT, _dims);
}

std::pair<osg::Vec3, osg::Vec3> Selection3DTool::getSelectionCenterAndDims() {
	//osg::Vec3 normalizedCenter = _updateCallback->_currCubeCoordNorm;
	/*float temp = normalizedCenter.y();*/
	/*normalizedCenter.y() = normalizedCenter.z();
	normalizedCenter.z() = temp;*/

  	osg::Vec3 normalizedDims = _normDims;
	float temp = normalizedDims.y();
	normalizedDims.y() = normalizedDims.z();
	normalizedDims.z() = temp;
	 
		return std::pair<osg::Vec3, osg::Vec3>(normalizedDims, _updateCallback->_currCubeCoordNorm);
}