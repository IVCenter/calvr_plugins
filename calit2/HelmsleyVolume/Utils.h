#ifndef HELMSLEY_UTILS_H
#define HELMSLEY_UTILS_H

#include <cvrKernel/InteractionEvent.h>
#include <cvrKernel/SceneObject.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/PopupMenu.h>
#include <cvrInput/TrackingManager.h>

#include <osg/Group>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/PositionAttitudeTransform>
#include <osg/MatrixTransform>

#include "VolumeGroup.h"

class PointAtHeadLerp : public osg::NodeCallback
{
public:
	PointAtHeadLerp(float amount, cvr::SceneObject* object = nullptr) : _lerp(amount), _object(object) {}

	virtual void operator()(osg::Node* node, osg::NodeVisitor* nv)
	{
		if (_object)
		{
			osg::Vec3 pos, scale;
			osg::Quat rot, scalerot;
			_object->getTransform().decompose(pos, rot, scale, scalerot);
			osg::Quat current = rot;

			//TODO: add hand/head mapping
			osg::Vec3 head =
				cvr::TrackingManager::instance()->getHeadMat(0).getTrans();

			osg::Vec3 viewerDir = head - pos;
			viewerDir.z() = 0.0;

			osg::Quat target;
			target.makeRotate(osg::Vec3(0, -1, 0), viewerDir);

			osg::Quat actual;
			actual.slerp(_lerp, current, target);

			osg::Matrix m;
			m.makeScale(scale);
			m.postMultRotate(actual);
			m.postMultTranslate(pos);
			_object->setTransform(m);
		}
		else 
		{
			osg::MatrixTransform* mt = dynamic_cast<osg::MatrixTransform*>(node);
			if (mt)
			{
				osg::Vec3 pos, scale;
				osg::Quat rot, scalerot;
				mt->getMatrix().decompose(pos, rot, scale, scalerot);
				osg::Quat current = rot;

				//TODO: add hand/head mapping
				osg::Vec3 head =
					cvr::TrackingManager::instance()->getHeadMat(0).getTrans();

				osg::Vec3 viewerDir = head - pos;
				viewerDir.z() = 0.0;

				osg::Quat target;
				target.makeRotate(osg::Vec3(0, -1, 0), viewerDir);

				osg::Quat actual;
				actual.slerp(_lerp, current, target);

				osg::Matrix m;
				m.makeScale(scale);
				m.postMultRotate(actual);
				m.postMultTranslate(pos);
				mt->setMatrix(m);
			}
		}
		
		traverse(node, nv);
	}

private:
	cvr::SceneObject* _object;
	float _lerp;
};

class FollowSceneObjectLerp : public osg::NodeCallback
{
public:
	FollowSceneObjectLerp(cvr::SceneObject* target, float amount, cvr::SceneObject* follower = nullptr) : _so(target), _lerp(amount), _follower(follower) {}

	virtual void operator()(osg::Node* node, osg::NodeVisitor* nv)
	{
		if (_follower)
		{
			osg::Vec3 pos, scale;
			osg::Quat rot, scalerot;
			_follower->getTransform().decompose(pos, rot, scale, scalerot);

			osg::Vec3 current = pos;

			osg::Vec3 target = _so->getPosition(); // target - current

			osg::Vec3 total = target * _lerp + current * (1.0f - _lerp);

			osg::Matrix m;
			m.makeScale(scale);
			m.postMultRotate(rot);
			m.postMultTranslate(total);


			_follower->setTransform(m);
		}
		else
		{
			osg::MatrixTransform* mt = dynamic_cast<osg::MatrixTransform*>(node);
			if (mt)
			{
				osg::Vec3 pos, scale;
				osg::Quat rot, scalerot;
				mt->getMatrix().decompose(pos, rot, scale, scalerot);

				osg::Vec3 current = pos;

				osg::Vec3 target = _so->getPosition(); // target - current

				osg::Vec3 total = target * _lerp + current * (1.0f - _lerp);

				osg::Matrix m;
				m.makeScale(scale);
				m.postMultRotate(rot);
				m.postMultTranslate(total);


				mt->setMatrix(m);
			}
		}
		traverse(node, nv);
	}

private:
	cvr::SceneObject* _so;
	cvr::SceneObject* _follower;
	float _lerp;
};

#endif