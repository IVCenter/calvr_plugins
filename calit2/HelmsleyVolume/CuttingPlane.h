#ifndef CUTTING_PLANE_H
#define CUTTING_PLANE_H

#include <cvrKernel/InteractionEvent.h>
#include <cvrKernel/SceneObject.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/PopupMenu.h>
#include <cvrMenu/MenuCheckbox.h>

#include <osg/Group>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/PositionAttitudeTransform>
#include <osg/MatrixTransform>
#include "VolumeGroup.h"
#include <time.h>




class CenterlineUpdate : public osg::NodeCallback
{
public:
	CenterlineUpdate(cvr::SceneObject* object, osg::Camera* cam = nullptr) : _camera(cam), _object(object) {}

	virtual void operator()(osg::Node* node, osg::NodeVisitor* nv)
	{
		if (_camera != nullptr) {	// camera loop
			if (_coords) {	
				if (_index >= 0 && _index < _coords->size() - 1 && col == 0 && (!_pause || _index == 0)) {	

					osg::Vec3d pos = _coords->at(_index) * _mTransform->getMatrix();
					osg::Vec3d next = _coords->at((_index + 1)) * _mTransform->getMatrix();
					osg::Matrix mat;
					osg::Vec3 dir = osg::Vec3(next - pos);
					dir.normalize();
					osg::Vec3 up = osg::Vec3(0, 0, 1);
					mat.makeLookAt(pos - (dir * 40), next, up);
					_camera->setViewMatrix(mat);
					_index++;
				}
				else if (_index == _coords->size() - 1) {
					_index = 0;
					col = 1;
				}
				if (_coords2 && _index >= 0 && _coords2->size() - 1 && col == 1 && !_pause) {
					osg::Vec3d pos = _coords2->at(_index) * _mTransform->getMatrix();
					osg::Vec3d next = _coords2->at((_index + 1)) * _mTransform->getMatrix();
					osg::Matrix mat;
					osg::Vec3 dir = osg::Vec3(next - pos);
					dir.normalize();
					osg::Vec3 up = osg::Vec3(0, 0, 1);
					mat.makeLookAt(pos - (dir * 40), next, up);
					_camera->setViewMatrix(mat);
					_index++;
				}
			}
		}
		else { // cp loop
			if (_coords) { 
				if (_index >= 0 && _index < _coords->size() - 1 && col == 0 && (!_pause || _index == 0)) {
					osg::Vec3d pos = _coords->at(_index + 1) * _mTransform->getMatrix();
					osg::Vec3d next = _coords->at((_index)) * _mTransform->getMatrix();

					osg::Vec3d posSwitch = osg::Vec3d(pos.x(), pos.z(), pos.y());
					osg::Vec3d nextSwitch = osg::Vec3d(next.x(), next.z(), next.y());

					osg::Matrix mat = _object->getTransform();
					osg::Matrix scale;
					scale.makeScale(osg::Vec3(0.12, 0.12, 0.12));

					mat.makeLookAt(posSwitch, nextSwitch, osg::Vec3d(0, 0, 1));
					mat = mat * scale;
					mat.setTrans(pos);

					_object->setTransform(mat);

					_index++;
					dirtyPlane();
				}
				else if (_index == _coords->size() - 1) {
					_index = 0;
					col = 1;
				}
				if (_coords2 && _index >= 0 && _coords2->size() - 1 && col == 1 && !_pause) {
					osg::Vec3d pos = _coords2->at(_index + 1) * _mTransform->getMatrix();
					osg::Vec3d next = _coords2->at((_index)) * _mTransform->getMatrix();

					osg::Vec3d posSwitch = osg::Vec3d(pos.x(), pos.z(), pos.y());
					osg::Vec3d nextSwitch = osg::Vec3d(next.x(), next.z(), next.y());

					osg::Matrix mat = _object->getTransform();
					osg::Matrix scale;
					scale.makeScale(osg::Vec3(0.12, 0.12, 0.12));

					mat.makeLookAt(posSwitch, nextSwitch, osg::Vec3d(0, 0, 1));
					mat = mat * scale;
					mat.setTrans(pos);

					_object->setTransform(mat);

					_index++;
					dirtyPlane();
				}
			}
		}

	}

	void setCoords(osg::Vec3dArray* coords, osg::MatrixTransform* transform) {
		_coords = coords;
		_mTransform = transform;
		_index = 0;
		col = 0;
	}
	void setCoords(osg::Vec3dArray* coords) {
		_coords2 = coords;
	}

	void startFromCol() {
		_index = 0;
		col = 0;
	}

	void startFromIll() {
		_index = 0;
		col = 1;
	}


	void dirtyPlane();

	void pause() {
		_pause = true;
	}
	void play() {
		_pause = false;
	}



private:

	cvr::SceneObject* _object;
	osg::Vec3dArray* _coords = nullptr;
	osg::Vec3dArray* _coords2 = nullptr;

	clock_t this_time = clock();
	clock_t last_time = this_time;
	double time_counter = 0;
	double testSeconds = .01;
	osg::MatrixTransform* _mTransform;
	osg::Camera* _camera;
	int _index = -1;
	int col;
	bool _pause = true;
};

class CuttingPlane : public cvr::SceneObject
{
public:
	CuttingPlane(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds = false)
		: SceneObject(name, navigation, movable, clip, contextMenu, showBounds)
	{
		init();
		_updateCallback = new CenterlineUpdate(this);
		this->getRoot()->addUpdateCallback(_updateCallback);
	}

	virtual void updateCallback(int handID, const osg::Matrix& mat);
	virtual void menuCallback(cvr::MenuItem* menuItem);
	virtual bool processEvent(cvr::InteractionEvent* ie);

	VolumeGroup* getVolume() { return _volume; }
	void setVolume(VolumeGroup* v) { _volume = v; }

	SceneObject* getSceneObject() { return _so; }
	void setSceneObject(SceneObject* s) { _so = s; }

	void setCoords(osg::Vec3dArray* coords, osg::MatrixTransform* trans) {
		_updateCallback->setCoords(coords, trans);
	}
	void setCoords(osg::Vec3dArray* coords) {
		_updateCallback->setCoords(coords);
	}

	void pause() {
		_updateCallback->pause();
	}

	void play() {
		_updateCallback->play();
	}
	CenterlineUpdate* getUC() { return _updateCallback; }

	void changePlane() {
		osg::Matrix obj2wrl = getObjectToWorldMatrix();

		osg::Matrix wrl2obj = _so->getWorldToObjectMatrix();
		osg::Matrix wrl2obj2 = _volume->getWorldToObjectMatrix();

		osg::Matrix posmat = obj2wrl * wrl2obj * wrl2obj2;

		osg::Quat q = osg::Quat();
		osg::Quat q2 = osg::Quat();
		osg::Vec3 v = osg::Vec3();
		osg::Vec3 v2 = osg::Vec3();
		osg::Matrix m = osg::Matrix();

		obj2wrl.decompose(v, q, v2, q2);
		m.makeRotate(q);

		wrl2obj.decompose(v, q, v2, q2);
		m.postMultRotate(q);

		wrl2obj2.decompose(v, q, v2, q2);
		m.postMultScale(osg::Vec3(1.0 / v2.x(), 1.0 / v2.y(), 1.0 / v2.z()));
		m.postMultRotate(q);

		osg::Vec4d normal = osg::Vec4(0, 1, 0, 0) * m;
		osg::Vec3 norm = osg::Vec3(normal.x(), normal.y(), normal.z());

		osg::Vec4f position = osg::Vec4(0, 0, 0, 1) * posmat;
		osg::Vec3f pos = osg::Vec3(position.x(), position.y(), position.z());


		//std::cerr << "Position: " << pos.x() << ", " << pos.y() << ", " << pos.z() << std::endl;
		//std::cerr << "Normal: " << norm.x() << ", " << norm.y() << ", " << norm.z() << std::endl << std::endl;


		_volume->_PlanePoint->set(pos);
		_volume->_PlaneNormal->set(norm);
		_volume->dirtyVolumeShader();
		_volume->dirtyComputeShader();

	}


protected:
	void init();


private:
	SceneObject* _so = nullptr;
	VolumeGroup* _volume = nullptr;
	
	cvr::MenuCheckbox* _lock = nullptr;
	//UpdateCuttingPlane* _updateCallback = nullptr; 
	CenterlineUpdate* _updateCallback = nullptr; 
	
	
	//osg::ref_ptr<osg::MatrixTransform> 
};


#endif