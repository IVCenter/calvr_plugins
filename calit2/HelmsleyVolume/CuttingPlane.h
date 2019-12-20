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

class CuttingPlane : public cvr::SceneObject
{
public:
	CuttingPlane(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds = false)
		: SceneObject(name, navigation, movable, clip, contextMenu, showBounds)
	{
		init();
	}

	virtual void updateCallback(int handID, const osg::Matrix& mat);
	virtual void menuCallback(cvr::MenuItem* menuItem);
	virtual bool processEvent(cvr::InteractionEvent* ie);

	VolumeGroup* getVolume() { return _volume; }
	void setVolume(VolumeGroup* v) { _volume = v; }

	SceneObject* getSceneObject() { return _so; }
	void setSceneObject(SceneObject* s) { _so = s; }

protected:
	void init();

private:
	SceneObject* _so = nullptr;
	VolumeGroup* _volume = nullptr;
	
	cvr::MenuCheckbox* _lock = nullptr;
	//osg::ref_ptr<osg::MatrixTransform> 
};

#endif