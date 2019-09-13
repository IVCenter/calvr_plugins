#ifndef HELMSLEY_VOLUME_H
#define HELMSLEY_VOLUME_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/FileHandler.h>
#include <cvrKernel/SceneObject.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/ComController.h>

#include <cvrConfig/ConfigManager.h>

#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuRadial.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuRangeValueCompact.h>
#include <cvrMenu/PopupMenu.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuList.h>

#include <osg/PositionAttitudeTransform>

#include "VolumeDrawable.h"
#include "VolumeGroup.h"
#include "MeasurementTool.h"

#include <string>

class HelmsleyVolume : public cvr::MenuCallback, public cvr::CVRPlugin
{
    public:
        HelmsleyVolume();
        virtual ~HelmsleyVolume();

        bool init();
        void preFrame();
		void postFrame();
		bool processEvent(cvr::InteractionEvent* e);
		void menuCallback(cvr::MenuItem* menuItem);
		void createList(cvr::SubMenu* , std::string configbase);


    protected:
		struct MeasurementInfo {
			osg::Vec3 start;
			osg::Vec3 end;
		};

		cvr::SubMenu * _vMenu;
		cvr::MenuButton * _vButton;
		std::vector<VolumeGroup*> _volumes;
		std::vector<cvr::SceneObject*> _sceneObjects;
		osg::MatrixTransform* cuttingPlane;
		MeasurementTool* measurementTool;


		std::map<cvr::MenuItem*, std::string> _buttonMap;
		std::map<cvr::MenuItem*, osg::Vec3> _buttonSizeMap;
		std::map<cvr::MenuItem*, VolumeGroup*> _stepSizeMap;
		std::map<cvr::MenuItem*, cvr::SceneObject*> _scaleMap;
		std::map<cvr::MenuItem*, std::pair<std::string, VolumeGroup*> > _computeShaderMap;
		std::map<cvr::MenuItem*, std::pair<std::string, VolumeGroup*> > _volumeDefineMap;
		std::map<cvr::MenuItem*, std::pair<std::string, VolumeGroup*> > _computeDefineMap;

		cvr::MenuRadial * _radial;
		cvr::PopupMenu * _selectionMenu;

		osg::Matrix _selectionMatrix;

		float _cuttingPlaneDistance;

		int _interactButton = 0;
		int _radialXVal = 0;
		int _radialYVal = 1;

		bool _radialShown = false;
		float _radialX = 0;
		float _radialY = 0;
};

#endif
