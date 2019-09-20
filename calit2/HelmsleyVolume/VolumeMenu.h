#ifndef HELMSLEY_VOLUME_MENU_H
#define HELMSLEY_VOLUME_MENU_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/SceneObject.h>

#include <cvrMenu/MenuItem.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuRangeValueCompact.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuList.h>

#include "VolumeGroup.h"

class VolumeMenu : public cvr::MenuCallback {
public:
	VolumeMenu(cvr::SceneObject* scene, VolumeGroup* volume) : _scene(scene), _volume(volume) {}

	void init();

	virtual void menuCallback(cvr::MenuItem * item);

	enum ColorFunction {
		DEFAULT = 0,
		RAINBOW = 1
	};
	ColorFunction transferFunction;
protected:
	cvr::SceneObject* _scene;
	VolumeGroup* _volume;

	cvr::MenuRangeValueCompact* scale;
	cvr::MenuRangeValueCompact* sampleDistance;

	cvr::MenuRangeValueCompact* contrastBottom;
	cvr::MenuRangeValueCompact* contrastTop;

	cvr::MenuRangeValueCompact* opacityMult;
	cvr::MenuRangeValueCompact* opacityCenter;
	cvr::MenuRangeValueCompact* opacityWidth;

	cvr::MenuCheckbox* adaptiveQuality;
	cvr::MenuCheckbox* highlightColon;
	cvr::MenuCheckbox* organsOnly;

	cvr::MenuList* colorFunction;
};

#endif
