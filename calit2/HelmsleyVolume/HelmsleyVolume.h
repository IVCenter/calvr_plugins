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

#include "VolumeGroup.h"
#include "VolumeMenu.h"
#include "MeasurementTool.h"
#include "ScreenshotTool.h"
#include "CenterLineTool.h"
#include "CuttingPlane.h"
#include "FileSelector.h"

#include <string>

class HelmsleyVolume : public cvr::MenuCallback, public cvr::CVRPlugin
{
    public:
		enum ToolState {
			NONE,
			CUTTING_PLANE,
			MEASUREMENT_TOOL,
			POINTER,
			CENTER_LINE
		};

        HelmsleyVolume();
        virtual ~HelmsleyVolume();

		static HelmsleyVolume* instance(){ return _instance; }

        bool init();
        void preFrame();
		void postFrame();

		bool processEvent(cvr::InteractionEvent* e);
		void menuCallback(cvr::MenuItem* menuItem);
		void createList(cvr::SubMenu* , std::string configbase);

		void toggleScreenshotTool(bool on);
		void toggleCenterlineTool(bool on);
		ScreenshotTool* getScreenshotTool() { return screenshotTool; }
		CenterlineTool* getCenterlineTool() { return centerLineTool; }

		CuttingPlane* createCuttingPlane();
		void removeCuttingPlane();

		void toggleCenterLine(bool on);
		
		void toggleHistogram(bool on);
		void toggleClaheTools(bool on);
		void toggleMaskAndPresets(bool on);
		void toggleTFUI(bool on);


		void loadVolume(std::string path, std::string maskpath = "", bool onlyVolume = false);
		void loadVolumeOnly(bool isPreset, std::string path, std::string maskpath = "");
		void loadSecondVolume(std::string path, std::string maskpath = "");
		void removeVolume(int index, bool onlyVolume);
		void removeVolumeOnly(int index);
		void removeSecondVolume();

		void setTool(ToolState tool) { _tool = tool; }
		void activateMeasurementTool(int volume);
		void deactivateMeasurementTool(int volume);
		void activateClippingPath();
		std::vector<CuttingPlane*> getCuttingPlanes() { return _cuttingPlanes; }
		std::vector<cvr::SceneObject*> getSceneObjects() { return _sceneObjects; }
		std::vector<osg::ref_ptr<VolumeGroup>> getVolumes() { return _volumes; }
		FileSelector* getFileSelector() { return fileSelector; }


		static std::string loadShaderFile(std::string filename);
		static void resetOrientation();

		void setVolumeIndex(int i, bool second) { 
			_volumeIndex = i;

			if (second) {
				if (i == 0) {
					_sceneObjects[0]->updateBBActiveColor(osg::Vec4(UI_ACTIVE_COLOR));
					_sceneObjects[0]->updateBBPassiveColor(osg::Vec4(UI_INACTIVE_COLOR));
					_sceneObjects[1]->updateBBActiveColor(osg::Vec4(UI_WHITE_COLOR));
					_sceneObjects[1]->updateBBPassiveColor(osg::Vec4(UI_INACTIVE_WHITE_COLOR));
				}
				else {
					_sceneObjects[1]->updateBBActiveColor(osg::Vec4(UI_ACTIVE_COLOR));
					_sceneObjects[1]->updateBBPassiveColor(osg::Vec4(UI_INACTIVE_COLOR));
					_sceneObjects[0]->updateBBActiveColor(osg::Vec4(UI_WHITE_COLOR));
					_sceneObjects[0]->updateBBPassiveColor(osg::Vec4(UI_INACTIVE_WHITE_COLOR));
				}
			}
		}
		int getVolumeIndex() { return _volumeIndex; }
    protected:
		struct MeasurementInfo {
			osg::Vec3 start;
			osg::Vec3 end;
		};

		cvr::SubMenu * _vMenu;
		cvr::MenuRadial * _vButton;
		cvr::MenuButton * _cpButton;
		cvr::MenuCheckbox * _mtButton;
		cvr::MenuCheckbox * _toolButton;
		cvr::MenuCheckbox * _stCheckbox;
		cvr::MenuButton * _resetHMD;
		osg::MatrixTransform* cuttingPlane;
		ScreenshotTool* screenshotTool;
		CenterlineTool* centerLineTool;
		FileSelector* fileSelector;

		std::vector<MeasurementTool*> _measurementTools;
		int _lastMeasurementTool;

		
		std::vector<CuttingPlane*> _cuttingPlanes;
		std::vector<osg::ref_ptr<VolumeGroup> > _volumes;
		std::vector<cvr::SceneObject*> _sceneObjects;
		std::vector<VolumeMenu*> _contextMenus;
		std::vector<NewVolumeMenu*> _worldMenus;
		std::vector<cvr::MenuButton*> _removeButtons;
		std::vector<cvr::MenuButton*> _removeClippingPlaneButtons;

		std::map<cvr::MenuItem*, std::string> _buttonMap;
		cvr::MenuRadial * _radial;
		cvr::PopupMenu * _selectionMenu;
		ToolMenu* _toolMenu;
		cvr::UIPopup* _splashscreen;

		cvr::SceneObject* _room;
		cvr::SubMenu* _roomLocation;
		cvr::MenuButton* _hideRoom;
		unsigned int _nm;

		osg::Matrix _selectionMatrix;

		float _cuttingPlaneDistance;

		int _frameNum = 0;
		int _interactButton = 0;
		int _radialButton = 3;
		int _radialXVal = 0;
		int _radialYVal = 1;

		bool _radialShown = false;
		float _radialX = 0;
		float _radialY = 0;

		ToolState _tool = NONE;

		static HelmsleyVolume* _instance;
		int _volumeIndex = 0;
};

#endif
