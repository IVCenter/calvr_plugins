#ifndef Selection3D_H
#define Selection3D_H


#include <osg/Group>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/PositionAttitudeTransform>
#include <osg/MatrixTransform>
#include <osg/CullFace>

#include <osgText/Text>

#include <cvrKernel/SceneObject.h>

#include <iostream>


class Selection3DToolUpdate : public osg::NodeCallback
{
public:
	Selection3DToolUpdate(cvr::SceneObject* object, cvr::SceneObject* mainSO, osg::ref_ptr<osg::MatrixTransform> cubeMT, osg::MatrixTransform* ruler, osg::Vec3 dims) : _selectionSO(object), _mainSO(mainSO), _cubeMT(cubeMT), _ruler(ruler), _dims(dims) {}

	virtual void operator()(osg::Node* node, osg::NodeVisitor* nv);
	
	cvr::SceneObject* _selectionSO;
	cvr::SceneObject* _mainSO;
	osg::MatrixTransform* _ruler;
	osg::Vec3 _dims;
	osg::Uniform* _selectionCenter;
	osg::ref_ptr<osg::MatrixTransform> _cubeMT;
	
};

class Selection3DTool : public cvr::SceneObject
{
public:
	Selection3DTool(osg::ref_ptr<osg::MatrixTransform> cubeMT, SceneObject* mainSo, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds = false)
		: SceneObject(name, navigation, movable, clip, contextMenu, showBounds), _cubeMT(cubeMT), _mainSO(mainSo)
	{
		init();
 	}

	void setStart(osg::Vec3 v);
	void setEnd(osg::Vec3 v);
 
	float getLength();
	void activate();
	void deactivate();
	void initCallback() {
		_updateCallback = new Selection3DToolUpdate(this, _mainSO, _cubeMT, _selectionMatrixTrans, _dims);
		

		this->getRoot()->addUpdateCallback(_updateCallback);
	}
	void setVoldims(osg::Vec3 dims, osg::Vec3 scale)	
	{
		_dims = dims;
		_scale = scale;
	}

	void linkUniforms(osg::Uniform* selectionDims, osg::Uniform* selectionCenter)
	{
		_selectionDims = selectionDims;

		_updateCallback->_selectionCenter = selectionCenter;
 	}

	osg::Vec3 getCenter() { return _selectionCenterVector; }
	void setLock(bool locked) {
		_locked = locked;
	}
	void setRemove(bool remove);
	void setDisable(bool disable);

	osg::Vec3 _scaledDims;
	osg::Vec3 _scale = osg::Vec3(0, 0, 0);

protected:
	void init();
	void update();

 	osg::ref_ptr<osg::MatrixTransform> _selectionMatrixTrans;

	bool _locked = false;
	bool _remove = false;
	osg::Vec3 _start;
	osg::Vec3 _end;
	osg::Vec3 _dims = osg::Vec3(0,0,0);
	
	osg::Vec3 _selectionCenterVector;
	

	osg::Uniform* _ustart;
	osg::Uniform* _uend;
	osg::StateSet* _stateset;
	Selection3DToolUpdate* _updateCallback;

	osg::ref_ptr<osg::MatrixTransform> _cubeMT;
	cvr::SceneObject* _mainSO;
	osg::Uniform* _selectionDims;
	bool isBackFace = false;
};

#endif