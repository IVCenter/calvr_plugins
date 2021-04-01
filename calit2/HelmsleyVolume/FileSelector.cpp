#pragma once
#include "FileSelector.h"
#include "HelmsleyVolume.h"
#include "Utils.h"
#include "UIExtensions.h"
 

#ifdef WIN32
#include "dirent.h"
#else
#include <dirent.h>
#endif
#include <stdio.h>

#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h>

#undef max
#include <algorithm>

#define SLOTCOUNT 6


FileSelector::~FileSelector() {
	delete addVolumeMenu;

	delete _so;

	for (auto p : _selections) {
		delete p;
	}
	_selections.clear();
}
void FileSelector::init()
{
	_state = NEW;
	pathSelections = std::vector<cvr::MenuItem*>();

	addVolumeMenu = new cvr::PopupMenu("Volumes", "", false, true);
	osg::Vec3 menupos = cvr::ConfigManager::getVec3("Plugin.HelmsleyVolume.Orientation.FileMenu.Position", osg::Vec3(-600, 500, 1100));
	addVolumeMenu->setPosition(menupos);
 

	addVol = new cvr::MenuButton("New Volume", true, "checkbox=TRUE.rgb");
	addVolumeMenu->addMenuItem(addVol);
	addVol->setCallback(this);

	volumeFileSelector = new cvr::PopupMenu("Choose File", "", false, true);
	volumeFileSelector->setPosition(menupos + osg::Vec3(0,0,-100));
 	_currentPath = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.BaseFolder", "C:/", false);


	////////////New FileSelect Implement.///////////
	_so = new cvr::SceneObject("FileSelect", false, false, false, false, false);
	cvr::PluginHelper::registerSceneObject(_so, "HelmsleyVolume");
	osg::Quat rot;
	rot.makeRotate(3.14/2, 0, 0, 1);
	_so->setRotation(rot);
	_so->setPosition(osg::Vec3(-1900, -1600, 2200));
	
	_fsPopup = new cvr::UIPopup();

	_fsPopup->getRootElement()->setAbsoluteSize(osg::Vec3(2400, 1, 1500));
	cvr::UIQuadElement* fsBknd = new cvr::UIQuadElement(UI_BACKGROUND_COLOR);
	fsBknd->setBorderSize(.01);
	fsBknd->setTransparent(true);
	fsBknd->setRounding(0, .05);
	_rightArrow = new TriangleButton(osg::Vec4(0.0, 0.0, 0.0, 1.0));
	_rightArrow->setCallback(this);
	fsBknd->addChild(_rightArrow);
	_rightArrow->setPercentSize(osg::Vec3(0.07, 1.0, .2));
	_rightArrow->setPercentPos(osg::Vec3(0.95, -1.0, -.5));
	
	_leftArrow = new TriangleButton(osg::Vec4(0.0, 0.0, 0.0, 1.0));
	_leftArrow->setCallback(this);
	fsBknd->addChild(_leftArrow);
	_leftArrow->setPercentSize(osg::Vec3(0.07, 1.0, .2));
	_leftArrow->setPercentPos(osg::Vec3(0.05, -1.0, -.5));
	_leftArrow->setRotate(1);

	_upArrow = new TriangleButton(osg::Vec4(0.0, 0.0, 0.0, 1.0));
	_upArrow->setCallback(this);
	fsBknd->addChild(_upArrow);
	_upArrow->setPercentSize(osg::Vec3(0.07, 1.0, .1));
	_upArrow->setPercentPos(osg::Vec3(0.05, -1.0, -.1));
	_upArrow->setRotate(.5f);

	cvr::UIText* legendText = new cvr::UIText("Has Mask:", 45.f, osgText::TextBase::LEFT_CENTER);
	cvr::UIQuadElement* legendMaskBox = new cvr::UIQuadElement(UI_BLACK_COLOR);
	legendText->addChild(legendMaskBox);
	legendText->setColor(UI_BLACK_COLOR);
	legendMaskBox->setPercentPos(osg::Vec3(1.0, 0.0, 0.0));
	legendMaskBox->setPercentSize(osg::Vec3(.25, 1.0, 0.8));
	//fsBknd->addChild(legendText);
	
	legendText->setPercentPos(osg::Vec3(0.05, -1.0, -.94));
	legendText->setPercentSize(osg::Vec3(0.1, 1.0, .04));

	_fsPopup->addChild(fsBknd);
	_fsPopup->setActive(true, true);
	
	
	cvr::UIQuadElement* titleBknd = new cvr::UIQuadElement();
	_fsPopup->addChild(titleBknd);
	titleBknd->setPercentPos(osg::Vec3(.4, -1, -.03));
	titleBknd->setPercentSize(osg::Vec3(.2, 1, .1));
	titleBknd->setBorderSize(.01);
	cvr::UIText* titleText = new cvr::UIText("Load Volume", 70.f, osgText::TextBase::CENTER_CENTER);
	titleText->setColor(osg::Vec4(0.0, 0.0, 0.0, 1.0));
	titleBknd->addChild(titleText);

	 _contentBknd = new cvr::UIQuadElement();
	_contentBknd->setPercentSize(osg::Vec3(.8, 1, .8));
	_contentBknd->setPercentPos(osg::Vec3(.1, -1, -.15));
	_contentBknd->setBorderSize(.01);
	_contentBknd->addChild(legendText);
	_fsPopup->addChild(_contentBknd);



	_so->addChild(_fsPopup->getRoot());
	_so->attachToScene();
	_fsPopup->getRootElement()->updateElement(osg::Vec3(0, 0, 0), osg::Vec3(0, 0, 0));
	_so->dirtyBounds();

	checkIfPatient(_currentPath, 0);	//Fills Patient directories
	_currMap = &_patientDirectories;
	
	


	_topList = new cvr::UIList(cvr::UIList::LEFT_TO_RIGHT, cvr::UIList::CONTINUE);
	_botList = new cvr::UIList(cvr::UIList::LEFT_TO_RIGHT, cvr::UIList::CONTINUE);
	_contentBknd->addChild(_topList);
	_topList->setPercentSize(osg::Vec3(0.8, 1, .4));
	_topList->setPercentPos(osg::Vec3(0.1, 1, -.05));
	_contentBknd->addChild(_botList);
	_botList->setPercentSize(osg::Vec3(0.8, 1, .4));
	_botList->setPercentPos(osg::Vec3(0.1, 1, -.50));

	int selectCount = 0;
	for (auto iter = _patientDirectories.begin(); iter != _patientDirectories.end(); ++iter)
	{
		std::string patient = iter->first;
		

		Selection* selec;
		_selections.push_back(selec);
		_selections[selectCount] = new Selection(iter->first, checkIfMask(iter->second));
		_selections[selectCount]->setId("selection");
		_selections[selectCount]->setButtonCallback(this);
		
		selectCount < SLOTCOUNT/2 ? _topList->addChild(_selections[selectCount]) : _botList->addChild(_selections[selectCount]);
		selectCount++;


		if (selectCount == SLOTCOUNT) {
			_selectIndex = selectCount;	
			break;
		}

	}
	while (selectCount < SLOTCOUNT) {
		Selection* selec;
		_selections.push_back(selec);
		_selections[selectCount] = new Selection("");
		_selections[selectCount]->setId("selection");
		_selections[selectCount]->setButtonCallback(this);

		selectCount < SLOTCOUNT / 2 ? _topList->addChild(_selections[selectCount]) : _botList->addChild(_selections[selectCount]);
		selectCount++;
	}



	//////////////////////////////Select View
	_selectBknd = new cvr::UIQuadElement();
	_selectBknd->setPercentSize(osg::Vec3(.8, 1, .8));
	_selectBknd->setPercentPos(osg::Vec3(.1, -1, -.15));
	_selectBknd->setBorderSize(.01);




	_selectImage = new cvr::UIQuadElement();
	_selectImage->setPercentSize(osg::Vec3(.3125, 1, .5));
	_selectImage->setPercentPos(osg::Vec3(.1, -1, -.25));
	_selectImage->setBorderSize(.01);

	_selectTexture = new cvr::UITexture();


	cvr::UIQuadElement* loadButton = new cvr::UIQuadElement();
	loadButton->setPercentSize(osg::Vec3(.3, 1, .125));
	loadButton->setPercentPos(osg::Vec3(.150, -1, -.8));
	loadButton->setBorderSize(.01);

	_loadVolumeButton = new CallbackButton();
	_loadVolumeButton->setCallback(this);
	loadButton->addChild(_loadVolumeButton);
	
	cvr::UIQuadElement* loadSecondBknd = new cvr::UIQuadElement();
	loadSecondBknd->setPercentSize(osg::Vec3(.3, 1, .125));
	loadSecondBknd->setPercentPos(osg::Vec3(.525, -1, -.8));
	loadSecondBknd->setBorderSize(.01);

	_loadSecondButton = new CallbackButton();
	_loadSecondButton->setCallback(this);
	loadSecondBknd->addChild(_loadSecondButton);

	cvr::UIText* loadText = new cvr::UIText("Load Dataset", 50.f, osgText::TextBase::CENTER_CENTER);
	loadText->setColor(osg::Vec4(0.0, 0.0, 0.0, 1.0));
	cvr::UIText* loadSecondText = new cvr::UIText("Load as Second", 50.f, osgText::TextBase::CENTER_CENTER);
	loadSecondText->setColor(osg::Vec4(0.0, 0.0, 0.0, 1.0));


	_categoryList = new cvr::UIList(cvr::UIList::TOP_TO_BOTTOM, cvr::UIList::CONTINUE);
	cvr::UIText* patientName = new cvr::UIText("Patient: ", 40.f, osgText::TextBase::LEFT_CENTER);
	patientName->setColor(osg::Vec4(0.0, 0.0, 0.0, 1.0));
	cvr::UIText* seriesName = new cvr::UIText("Series: ", 40.f, osgText::TextBase::LEFT_CENTER);
	seriesName->setColor(osg::Vec4(0.0, 0.0, 0.0, 1.0));
	cvr::UIText* studyDate = new cvr::UIText("Study Date: ", 40.f, osgText::TextBase::LEFT_CENTER);
	studyDate->setColor(osg::Vec4(0.0, 0.0, 0.0, 1.0));

	_categoryList->addChild(patientName);
	_categoryList->addChild(seriesName);
	_categoryList->addChild(studyDate);

	_categoryList->setPercentPos(osg::Vec3(0.46, -1, -.25));
	_categoryList->setPercentSize(osg::Vec3(1.0, 1, .2));

	_infoList = new cvr::UIList(cvr::UIList::TOP_TO_BOTTOM, cvr::UIList::CONTINUE);
	
	cvr::UIText* patientInfo = new cvr::UIText("", 40.f, osgText::TextBase::LEFT_CENTER);
	patientInfo->setColor(osg::Vec4(0.0, 0.0, 0.0, 1.0));
	cvr::UIText* seriesInfo = new cvr::UIText("", 40.f, osgText::TextBase::LEFT_CENTER);
	seriesInfo->setColor(osg::Vec4(0.0, 0.0, 0.0, 1.0));
	cvr::UIText* studyDateInfo = new cvr::UIText("", 40.f, osgText::TextBase::LEFT_CENTER);
	studyDateInfo->setColor(osg::Vec4(0.0, 0.0, 0.0, 1.0));
	_infoList->addChild(patientInfo);
	_infoList->addChild(seriesInfo);
	_infoList->addChild(studyDateInfo);
	_infoList->setPercentPos(osg::Vec3(0.6, -1, -.25));
	_infoList->setPercentSize(osg::Vec3(1.0, 1, .2));

	
	

	_selectBknd->addChild(_selectImage);
	_selectBknd->addChild(_categoryList);
	_selectBknd->addChild(_infoList);
	_selectBknd->addChild(loadButton);
	_selectBknd->addChild(loadSecondBknd);

	
	_selectImage->addChild(_selectTexture);
	_selectTexture->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));
	loadButton->addChild(loadText);
	loadSecondBknd->addChild(loadSecondText);



	_menusLoaded = false;
	_isOnLoad = false;
}

void FileSelector::menuCallback(cvr::MenuItem* item)
{
	if (_state == NEW || _state == CHANGE)
	{
		if (item == addVol)
		{
			if (_state == NEW)
			{
				_state = CHOOSING;
			}
			else
			{
				_state = CHANGING;
			}
			volumeFileSelector->setVisible(true);
			updateFileSelection();
		}
	}
	else if (_state == CHOOSING || _state == CHANGING)
	{
		if (std::find(pathSelections.begin(), pathSelections.end(), item) != pathSelections.end())
		{
			cvr::MenuButton* button = (cvr::MenuButton*)item;
			if (strcmp(button->getText().c_str(), "..") == 0)
			{
				//Go back a folder
				_currentPath = _currentPath.substr(0, _currentPath.find_last_of("\\/"));
				updateFileSelection();
			}
			else if (button->getText().find(".dcm") != std::string::npos && 
				button->getText().find(".dcm") == button->getText().find_last_of("."))
			{
				////Load volume
				bool change = _state == CHANGING ? true : false;
				loadVolume(_currentPath, change, false);
			}
			else
			{
				_currentPath = _currentPath + "/" + button->getText();
				updateFileSelection();
			}
		}
	}
}

void FileSelector::threadTest() {
	int result = system("C:/Users/g3aguirre/Documents/CAL/Testing/ExternalAppTest/x64/Release/ExternalAppTest.exe");
  }

void FileSelector::uiCallback(UICallbackCaller* ui){
	

	//_futures.push_back(std::async(std::launch::async, threadTest));
	
	
	if (ui == _rightArrow) {
		if(_currMap->size() > _selectIndex)
			updateSelections(RIGHT);
	}
	else if (ui == _leftArrow) {
		if (_selectIndex > SLOTCOUNT)
			updateSelections(LEFT);
	}
	else if (ui == _upArrow) {
		_selectIndex = 0;
		if (_isOnLoad) {
			_fsPopup->addChild(_contentBknd);
			_contentBknd->setActive(true);
			_fsPopup->getRootElement()->getGroup()->removeChild(_selectBknd->getGroup());
			_selectBknd->_parent = nullptr;
			_selectBknd->setActive(false);

			_isOnLoad = false;
			_currMap = &_seriesList;
		}
		else if (_currMap == &_seriesList) {
			_selectIndex = 0;
			_currMap = &_patientDirectories;
			for (unsigned int i = 0; i < SLOTCOUNT; ++i) {
				_selections[i]->removeImage();
			}
		}
		updateSelections(RIGHT);
	}
	else if (checkSelectionCallbacks(ui)) {
		return;
	}
	else if (ui == _loadVolumeButton) {
		if (_menusLoaded == false) {
			loadVolume(_currentPath, false, false);
			_menusLoaded = true;
		}
		else {
			loadVolumeOnly(false, _currentPath);
		}
	}
	else if (ui == _loadSecondButton && _menusLoaded) {
		//bool change = _state == CHANGING ? true : false;
		loadSecondVolume(_currentPath);
	}



}

bool FileSelector::checkSelectionCallbacks(UICallbackCaller* item) {
	bool found = false;
	for (int i = 0; i < SLOTCOUNT; i++) {
		if (item == _selections[i]->getButton()) {
			found = true;
			if (_currMap == &_patientDirectories) {
				std::string copy = _selections[i]->getName();
				if (copy == "")
					break;

				copy.erase(copy.begin());	//erase m/n
				_patientInfo.name = copy;
				loadPatient(_selections[i]->getName());

 			}
			else if (_currMap == &_seriesList) {
				std::string copy = _selections[i]->getName();
				if (copy == "")
					break;

				_fsPopup->getRootElement()->getGroup()->removeChild(_contentBknd->getGroup());
				_contentBknd->_parent = nullptr;
				_contentBknd->setActive(false);


				_fsPopup->addChild(_selectBknd);
				_selectBknd->setActive(true);
				_selectTexture->setTexture(_selections[i]->getImage()->getTexture());

				_selectTexture->_geode->getDrawable(0)->getOrCreateStateSet()->setDefine("GRAYSCALE", true);
				_selectTexture->setDirty(true);

				

				_currentPath = _seriesList[_selections[i]->getName()];
			
				getPatientInfo(_currentPath);

			



				copy.erase(copy.begin());
				_patientInfo.series = copy;


				cvr::UIText* category = (cvr::UIText*)_infoList->getChild(FileSelector::CategoryEnum::PATIENT);
				category->setText(_patientInfo.name);
				category = (cvr::UIText*)_infoList->getChild(FileSelector::CategoryEnum::SERIES);
				category->setText(_patientInfo.series);
				category = (cvr::UIText*)_infoList->getChild(FileSelector::CategoryEnum::STUDYDATE);
				category->setText(_patientInfo.studyDate);

				_isOnLoad = true;
			}
			
			return found;
		}
	}
	return found;
}

void FileSelector::getPatientInfo(std::string pName) {
	
	DIR* dir = opendir(pName.c_str());
	std::string filePath = "";
	if (dir != nullptr) {
		struct dirent* entry = readdir(dir);
	
		while (entry != NULL) {
			if (entry->d_type == DT_REG && strcmp(strrchr(entry->d_name, '.') + 1, "dcm") == 0) {
				filePath = pName + "/" + entry->d_name;
				break;
			}
			entry = readdir(dir);
		}
		closedir(dir);
	}

	
	DcmFileFormat fileFormat;
	if (fileFormat.loadFile(filePath.c_str()).good()) {
	
	}
	DcmDataset* dataset = fileFormat.getDataset();

	std::string studyDate = "";
	OFString oFSTD;
	if (dataset->findAndGetOFString(DCM_StudyDate, oFSTD).good()) {
	
		studyDate = oFSTD.c_str();
	}

	_patientInfo.studyDate = studyDate;

}

void FileSelector::loadPatient(std::string pName) {
	std::string pFileName = _patientDirectories[pName];
	_seriesList.clear();
	loadSeriesList(pFileName, -1);
	_selectIndex = 0;
	_currMap = &_seriesList;
	updateSelections(RIGHT);



}


void FileSelector::showDicomThumbnail() {
	for (int i = 0; i < SLOTCOUNT; i++) {
		unsigned int seriesIndex = (_selectIndex - SLOTCOUNT) + i;
		if (seriesIndex >= _seriesList.size())
			break;
		std::string path = getMiddleImage(seriesIndex);
		DicomImage* image = new DicomImage(path.c_str());
		assert(image != NULL);
		assert(image->getStatus() == EIS_Normal);

		// Get information
		DcmFileFormat fileFormat;
		assert(fileFormat.loadFile(path.c_str()).good());
		DcmDataset* dataset = fileFormat.getDataset();
		double spacingX = 0.0;
		double spacingY = 0.0;
		double thickness = 0.0;
		OFCondition cnd;
		cnd = dataset->findAndGetFloat64(DCM_PixelSpacing, spacingX, 0);
		cnd = dataset->findAndGetFloat64(DCM_PixelSpacing, spacingY, 1);
		
		

		unsigned int w = image->getWidth();
		unsigned int h = image->getHeight();
		unsigned int d = 1;



		osg::ref_ptr<osg::Image> img = new osg::Image();
		img->allocateImage(w, h, d, GL_RG, GL_UNSIGNED_SHORT);
		uint16_t* data = (uint16_t*)img->data();
		memset(data, 0, w * h * d * sizeof(uint16_t) * 2);

		image->setMinMaxWindow();
		uint16_t* pixelData = (uint16_t*)image->getOutputData(16);

		//memcpy(data + w * h*i, pixelData, w * h * sizeof(uint16_t));
		unsigned int j = 0;
		for (unsigned int x = 0; x < w; x++) {
			for (unsigned int y = 0; y < h; y++) {
				j = 2 * (x + y * w);
				data[j] = pixelData[x + y * w];
				data[j + 1] = 0xFFFF;
			}
		}

		osg::ref_ptr<osg::Texture2D> texture = new osg::Texture2D(img);
		texture->setInternalFormat(GL_RGBA8);
		cvr::UITexture* imgTexture = new cvr::UITexture(texture);
		imgTexture->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));
		imgTexture->_geode->getDrawable(0)->getOrCreateStateSet()->setDefine("GRAYSCALE", true);

		_selections[i]->setImage(imgTexture);

		delete image;
	}
}

std::string FileSelector::getMiddleImage(int seriesIndex) {
	auto it = _seriesList.begin();
	for (unsigned int i = 0; i != seriesIndex; i++) {
		it++;
	}
	std::string seriesPath = it->second;

	DIR* dir = opendir(seriesPath.c_str());
	struct dirent* entry = readdir(dir);
	int count = 0;
	while (entry != NULL) {
		if (entry->d_type == DT_REG && strcmp(strrchr(entry->d_name, '.') + 1, "dcm") == 0) {
			count++;
		}
		entry = readdir(dir);
	}
	closedir(dir);



	dir = opendir(seriesPath.c_str());
	entry = readdir(dir);
	for (int i = 0; i < count / 2; i++) {
		entry = readdir(dir);
	}
	closedir(dir);
	return (seriesPath + "/" + entry->d_name);
}

int FileSelector::loadSeriesList(std::string pFN, int indexFromDicom) {
	DIR* dir = opendir(pFN.c_str());
	struct dirent* entry = readdir(dir);
	while (entry != NULL) {
		if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
			std::string newFn = pFN + "/" + entry->d_name;
			int newIndex = loadSeriesList(newFn, indexFromDicom);
			if (newIndex > -1) {
				char mask;
				std::string key;
				checkIfMask(pFN + "/" + entry->d_name) ? mask = 'm' : mask = 'n';	//m = mask/n= no mask
				key = mask + std::string(entry->d_name);
				_seriesList[key] = pFN + "/" + entry->d_name;
			}
		}
		else if (entry->d_type == DT_REG && strcmp(strrchr(entry->d_name, '.') + 1, "dcm") == 0) {
			closedir(dir);
			return indexFromDicom + 1;
		}
		entry = readdir(dir);
	}
	closedir(dir);
	return indexFromDicom + 0;
}

void FileSelector::updateSelections(SelectChoice choice) {


	
	
	int slotsLeft = SLOTCOUNT;
	int topBotIndex = 0;
	int maskCount = 0;
	if (choice == LEFT)
		_selectIndex -= SLOTCOUNT * 2;



	auto it = _currMap->begin();
	std::advance(it, _selectIndex);

	while (slotsLeft > 0) {

		Selection* selec;
		topBotIndex < SLOTCOUNT / 2 ? selec = (Selection*)_topList->getChild(topBotIndex)
			: selec = (Selection*)_botList->getChild(topBotIndex % (SLOTCOUNT / 2));

		if (_currMap->size() > _selectIndex) {
			selec->setName(it->first);
			bool hasMask = checkIfMask(it->second);
			selec->setMask(hasMask);
			if (hasMask)
				maskCount++;
		}
		else {
			selec->setName("");
			selec->setMask(false);
		}

		topBotIndex++;
		it++;			//path map iterator
		_selectIndex++; //path map "index"
		slotsLeft--;
	}
	
	if (_currMap == &_seriesList) {
		for (unsigned int i = 0; i < SLOTCOUNT; ++i) {
			if(_selections[i] != nullptr)
				_selections[i]->removeImage();
		}
		showDicomThumbnail();
	}
	
	
}

void FileSelector::sortByMask(std::map<std::string, std::string>* currMap) {
	
	for (auto iter = currMap->begin(); iter != currMap->end(); ++iter)
	{
		
		if (checkIfMask(iter->second)) {
			
		}
		

	}
}
void FileSelector::loadVolume(std::string seriesPath, bool change, bool onlyVolume) {
	std::string maskpath = "";

	DIR* dir = opendir((seriesPath + "/mask").c_str());
	if (dir != NULL)
	{
		maskpath = seriesPath + "/mask";
		closedir(dir);
	}

	HelmsleyVolume::instance()->loadVolume(seriesPath, maskpath, onlyVolume);
	_state = CHANGE;//change//changing
}

void FileSelector::loadVolumeOnly(bool isPreset, std::string seriesPath) {
	HelmsleyVolume::instance()->removeVolumeOnly(0);
	

	std::string maskpath = "";

	DIR* dir = opendir((seriesPath + "/mask").c_str());
	if (dir != NULL)
	{
		maskpath = seriesPath + "/mask";
		closedir(dir);
	}

	HelmsleyVolume::instance()->loadVolumeOnly(isPreset, seriesPath, maskpath);
	_state = CHANGE;
}

void FileSelector::loadSecondVolume(std::string seriesPath) {
	HelmsleyVolume::instance()->removeSecondVolume();
	std::string maskpath = "";

	DIR* dir = opendir((seriesPath + "/mask").c_str());
	if (dir != NULL)
	{
		maskpath = seriesPath + "/mask";
		closedir(dir);
	}

	HelmsleyVolume::instance()->loadSecondVolume(seriesPath, maskpath);
	_state = CHANGE;
}

void FileSelector::updateFileSelection()
{
	//Clear current directory options
	for (int i = 0; i < pathSelections.size(); ++i)
	{
		volumeFileSelector->removeMenuItem(pathSelections[i]);
		//delete(pathSelections[i]);
	}
	pathSelections.clear();

	int begin = std::max((int)(_currentPath.size() - 25), 0);
	std::string title = "";
	if (begin > 0)
	{
		title += "...";
	}
	title += _currentPath.substr(begin, _currentPath.size());
	volumeFileSelector->setTitle(title);


	DIR* dir = opendir(_currentPath.c_str());

	int dcmCount = 0;
	cvr::MenuButton* dcmButton = NULL;
	
	struct dirent* entry = readdir(dir);
	while (entry != NULL)
	{
		if ((entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "mask") != 0) || //folders
			(entry->d_type == DT_REG && strcmp(strrchr(entry->d_name, '.') + 1, "dcm") == 0)) //first dcm file
		{
			if (entry->d_type == DT_REG)
			{
				dcmCount++;
				if (dcmCount > 1)
				{
					entry = readdir(dir);
					continue;
				}
			}
			cvr::MenuButton* button = new cvr::MenuButton(entry->d_name);
			if (entry->d_type == DT_REG)
			{
				dcmButton = button;
			}
			button->setCallback(this);
			pathSelections.push_back(button);
			volumeFileSelector->addMenuItem(button);
		}


		entry = readdir(dir);
	}

	if (dcmButton)
	{
		dcmButton->setText(dcmButton->getText() + "    (" + std::to_string(dcmCount) + " files)");
	}

	closedir(dir);
}

std::pair<int, bool> FileSelector::checkIfPatient(std::string fn, int indexFromDicom) {
	std::pair<int, bool> toReturn(indexFromDicom, false);
	DIR* dir = opendir(fn.c_str());
	struct dirent* entry = readdir(dir);
	while (entry != NULL) {
		if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 && strcmp(entry->d_name, "mask") != 0) {
			std::string newFn = fn + "/" + entry->d_name;
			toReturn.first++;
			toReturn = checkIfPatient(newFn, indexFromDicom);
			if (toReturn.second == true) {

 				std::string seriesPath = fn + "/" + entry->d_name;
				std::size_t pos = fn.find_last_of("\\");
				std::size_t pos2 = fn.find_last_of("/");

				if (pos != std::string::npos && pos2 != std::string::npos) {
					pos = std::max(pos, pos2);
				}
				else {
					pos = min(pos, pos2);
				}
				
				

				std::string patientName = fn.substr(pos+1);
				

 				std::string patientDir = fn.substr(0, pos+1);
 				char mask;
				std::string key;
				checkIfMask(patientDir) ? mask = 'm' : mask = 'n';	//m = mask/n= no mask
				key = mask + patientName;

				_patientDirectories[key] = fn;
				toReturn.second = false;
				

			}
			
		}
		else if (entry->d_type == DT_REG && strcmp(strrchr(entry->d_name, '.') + 1, "dcm") == 0) {
			closedir(dir);
			toReturn.second = true;


			//toReturn.second = true;
			return toReturn;
		}
		entry = readdir(dir);
	}
	closedir(dir);
	return toReturn;

}

std::vector<std::string> FileSelector::getPresets() {
	std::string currPath = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.PresetsFolder", "C:/", false);
	std::string presetPath = currPath;
	DIR* dir = opendir(presetPath.c_str());
	if (dir == nullptr) {

	}
	std::vector<std::string> presetPaths;
	struct dirent* entry = readdir(dir);
	while (entry != NULL) {
		if (entry->d_type == DT_REG && strcmp(strrchr(entry->d_name, '.') + 1, "yml") == 0) {
			presetPaths.push_back(presetPath + "\\" + entry->d_name);
		}
		entry = readdir(dir);
	}
	return presetPaths;
}

bool FileSelector::checkIfMask(std::string seriesPath) {
	
	DIR* mainDir = opendir(seriesPath.c_str());
	if (mainDir == nullptr)
		return false;
	struct dirent* entry = readdir(mainDir);
	while (entry != NULL) {
		if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
		
			if (strcmp(entry->d_name, "mask") == 0) {
				closedir(mainDir);
				return true;
			}
			std::string newFn = seriesPath + "/" + entry->d_name;
			if (checkIfMask(newFn)) {
				closedir(mainDir);
				return true;
			}
		}
		entry = readdir(mainDir);
	}
	closedir(mainDir);
	return false;

}

osg::ref_ptr<osg::Vec3dArray> FileSelector::loadCenterLine(std::string path, OrganEnum organ) {
	DIR* dir = opendir(path.c_str());
	osg::ref_ptr<osg::Vec3dArray> coords = new osg::Vec3dArray();
	std::string coordsPath = "";

	if (dir != NULL)
	{
		struct dirent* entry = readdir(dir);
		while (entry != NULL) {
			if (entry->d_type == DT_REG && strcmp(strrchr(entry->d_name, '.') + 1, "yaml") == 0) {
				coordsPath = path + "\\" + entry->d_name;
				break;
			}
			entry = readdir(dir);
		}
		
		if (coordsPath != "") {

			YAML::Node yamlFile = YAML::LoadFile(coordsPath);
			YAML::Node yamlCoords;
			if (organ == OrganEnum::COLON)
				yamlCoords = yamlFile[1][std::to_string(organ)];
			if (organ == OrganEnum::ILLEUM)
				yamlCoords = yamlFile[2][std::to_string(organ)];

			osg::Vec3d coord;
			for (unsigned i = 0; i < yamlCoords.size(); i++) {
				coord.x() = (yamlCoords[i]["x"].as<double>());
				coord.y() = -(yamlCoords[i]["y"].as<double>());
				coord.z() = ((yamlCoords[i]["z"].as<double>()));
				coords->push_back(coord);
			}
			
		}
	}
	closedir(dir);
	return coords;
}

