#include "Selection3DTool.h"
#include "HelmsleyVolume.h"
#include <cvrConfig/ConfigManager.h>

#include <sstream>
#include <iomanip>

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
	if (_mainSO) { 

		osg::Vec3 cubeWorldPos = _cubeMT->getMatrix().getTrans() * _mainSO->getObjectToWorldMatrix();
  		osg::Vec3 selectionWorldPos = _ruler->getMatrix().getTrans() * _selectionSO->getObjectToWorldMatrix();
 
		osg::Matrix cubeMat = osg::Matrix::identity();
		_cubeMT->computeWorldToLocalMatrix(cubeMat, NULL);
		osg::Vec3 cubeCoord = selectionWorldPos * _mainSO->getWorldToObjectMatrix() * cubeMat;
 		cubeCoord += osg::Vec3(.5, .5, .5);
		cubeCoord = osg::Vec3(std::abs(_dims.x() * cubeCoord.x()), std::abs(_dims.z() * cubeCoord.y()), std::abs(_dims.y() * cubeCoord.z()));


		

		 
 /*

		osg::Vec3 scaledDims = static_cast<Selection3DTool*>(_selectionSO)->_scaledDims;
		osg::Vec3 scale = static_cast<Selection3DTool*>(_selectionSO)->_scale;

		scaledDims = osg::Vec3(std::abs(scaledDims.x() * scale.x()), std::abs(scaledDims.y() * scale.y()), std::abs(scaledDims.z() * scale.z()));


		osg::Vec3 lowerBounds = cubeCoord - (scaledDims / 2);
		osg::Vec3 upperBounds = cubeCoord + (scaledDims / 2);

		
		*/
		 


		_selectionCenter->setElement(0, cubeCoord);
		//std::cout << "scaledDims: " << HelmsleyVolume::instance()->printVec3OSG(scaledDims) << std::endl;
		////std::cout << "ruler: " << HelmsleyVolume::instance()->printVec3OSG(_ruler->getMatrix().getTrans()) << std::endl;
		//////std::cout << "OLD: " << HelmsleyVolume::instance()->printVec3OSG(cubeWorldPosOLD) << std::endl;
		////std::cout << "selectionWorldPos: " << HelmsleyVolume::instance()->printVec3OSG(selectionWorldPos) << std::endl;
		////std::cout << "cubeCoord: " << HelmsleyVolume::instance()->printVec3OSG(cubeCoord) << std::endl;
 	//	//std::cout << "selectionWorldPos: " << HelmsleyVolume::instance()->printVec3OSG(selectionWorldPos) << std::endl;
  //		std::cout << "center pos: " << HelmsleyVolume::instance()->printVec3OSG(cubeCoord) << std::endl;
 	//	std::cout << "lowerBounds: " << HelmsleyVolume::instance()->printVec3OSG(lowerBounds) << std::endl;
		//std::cout << "upperBound: " << HelmsleyVolume::instance()->printVec3OSG(upperBounds) << std::endl;
	}
	traverse(node, nv);
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
	//_scaledDims = osg::Vec3(480, 480, 179);
 
 
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