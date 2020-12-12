#pragma once
#ifndef FILE_SELECTOR_H
#define FILE_SELECTOR_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/SceneObject.h>

#include <cvrMenu/MenuItem.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuRangeValueCompact.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/PopupMenu.h>
#include "UIExtensions.h"
#include <cvrMenu/NewUI/UIPopup.h>

#include <cvrConfig/ConfigManager.h>

class FileSelector : public cvr::MenuCallback, public UICallback, public UICallbackCaller {
public:
	FileSelector() 
	{
		init();
	}
	~FileSelector();
	//TODO Deconstructor
	void init();

	virtual void menuCallback(cvr::MenuItem * item);

	enum MenuState {
		NEW,
		CHOOSING,
		CHANGE,
		CHANGING
	};

	enum SelectChoice{
		LEFT,
		RIGHT,
	};

	enum OrganEnum {
		COLON = 4,
		ILLEUM = 16
	};

	enum CategoryEnum {
		PATIENT,
		SERIES,
		STUDYDATE
	};

	struct PatientInfo {
		std::string name;
		std::string series;
		std::string fileCount;
		std::string studyDate;
	};



	static std::vector<std::string> getPresets();
	static osg::ref_ptr<osg::Vec3dArray> loadCenterLine(std::string path, OrganEnum organ);
	static bool checkIfMask(std::string fName);
	virtual void uiCallback(UICallbackCaller* ui);
	std::string getCurrPath() { return _currentPath; }

	void sortByMask(std::map<std::string, std::string>* currMap);
	void loadVolume(std::string seriesPath, bool change, bool onlyVolume);
	void loadVolumeOnly(bool isPreset, std::string seriesPath);
	void loadSecondVolume(std::string seriesPath);
	
protected:
	void updateSelections(SelectChoice choice);
	void updateFileSelection(); //OLD
	//void newUpdateFileSelection();
	std::pair<int, bool> checkIfPatient(std::string fn, int indexFromDicom);
	bool checkSelectionCallbacks(UICallbackCaller* item);
	void loadPatient(std::string pName);
	void getPatientInfo(std::string seriesPath);
	int loadSeriesList(std::string pFN, int indexFromDicom);
	void showDicomThumbnail();
	std::string getMiddleImage(int seriesIndex);

	cvr::SceneObject* _so;

	cvr::PopupMenu* addVolumeMenu;
	cvr::MenuButton* addVol;
	cvr::PopupMenu* volumeFileSelector;
	
	cvr::UIQuadElement* _contentBknd;
	cvr::UIQuadElement* _selectBknd;
	cvr::UIQuadElement* _selectImage;
	cvr::UITexture* _selectTexture;
	CallbackButton* _loadVolumeButton;
	CallbackButton* _loadSecondButton;
	cvr::UIPopup* _fsPopup;

	cvr::UIList* _topList;
	cvr::UIList* _botList;
	cvr::UIList* _categoryList;
	cvr::UIList* _infoList;
	TriangleButton* _rightArrow;
	TriangleButton* _leftArrow;
	TriangleButton* _upArrow;
	int _selectIndex = 0;
	std::vector<Selection*> _selections;
	
	//std::map<cvr::MenuItem*, std::string> loadVolumeMenu;
	std::vector<cvr::MenuItem*> pathSelections;
	std::map<std::string, std::string> _patientDirectories;	//Patient Folder Name, Directory
	std::map<std::string, std::string> _seriesList;
	std::map<std::string, std::string>* _currMap;
	bool _isOnLoad;
	MenuState _state;
	std::string _currentPath;
	PatientInfo _patientInfo;
	bool _menusLoaded;
	int _flag;
};



#endif
