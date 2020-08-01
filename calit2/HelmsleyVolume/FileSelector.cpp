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
void FileSelector::init()
{
	_state = NEW;
	pathSelections = std::vector<cvr::MenuItem*>();

	addVolumeMenu = new cvr::PopupMenu("Volumes", "", false, true);
	osg::Vec3 menupos = cvr::ConfigManager::getVec3("Plugin.HelmsleyVolume.Orientation.FileMenu.Position", osg::Vec3(-600, 500, 1100));
	addVolumeMenu->setPosition(menupos);
	addVolumeMenu->setVisible(true);
	//addVolumeMenu->setMovable(false);

	addVol = new cvr::MenuButton("New Volume", true, "checkbox=TRUE.rgb");
	addVolumeMenu->addMenuItem(addVol);
	addVol->setCallback(this);

	volumeFileSelector = new cvr::PopupMenu("Choose File", "", false, true);
	volumeFileSelector->setPosition(menupos + osg::Vec3(0,0,-100));
	//volumeFileSelector->getRootSubMenu()->setTextScale(0.5);
	_currentPath = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.BaseFolder", "C:/", false);
	std::cout << "Volume search path: " << _currentPath << std::endl;





	////////////New FileSelect Implement.///////////
	_so = new cvr::SceneObject("FileSelect", false, true, false, false, false);
	cvr::PluginHelper::registerSceneObject(_so, "HelmsleyVolume");
	osg::Quat rot;
	rot.makeRotate(3.14/2, 0, 0, 1);
	_so->setRotation(rot);
	_so->setPosition(osg::Vec3(-1900, -1600, 2200));
	
	cvr::UIPopup* fsPopup = new cvr::UIPopup();

	fsPopup->getRootElement()->setAbsoluteSize(osg::Vec3(2400, 1, 1500));
	cvr::UIQuadElement* fsBknd = new cvr::UIQuadElement(UI_BACKGROUND_COLOR);
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

	fsPopup->addChild(fsBknd);
	fsPopup->setActive(true, true);
	
	
	cvr::UIQuadElement* titleBknd = new cvr::UIQuadElement();
	fsPopup->addChild(titleBknd);
	titleBknd->setPercentPos(osg::Vec3(.4, -1, -.03));
	titleBknd->setPercentSize(osg::Vec3(.2, 1, .1));
	titleBknd->setBorderSize(.01);
	cvr::UIText* titleText = new cvr::UIText("Load Volume", 70.f, osgText::TextBase::CENTER_CENTER);
	titleText->setColor(osg::Vec4(0.0, 0.0, 0.0, 1.0));
	titleBknd->addChild(titleText);

	cvr::UIQuadElement* contentBknd = new cvr::UIQuadElement();
	contentBknd->setPercentSize(osg::Vec3(.8, 1, .8));
	contentBknd->setPercentPos(osg::Vec3(.1, -1, -.15));
	contentBknd->setBorderSize(.01);
	fsPopup->addChild(contentBknd);

	_so->addChild(fsPopup->getRoot());
	_so->attachToScene();
	fsPopup->getRootElement()->updateElement(osg::Vec3(0, 0, 0), osg::Vec3(0, 0, 0));
	_so->dirtyBounds();

	checkIfPatient(_currentPath, -1);

	


	_topList = new cvr::UIList(cvr::UIList::LEFT_TO_RIGHT, cvr::UIList::CONTINUE);
	_botList = new cvr::UIList(cvr::UIList::LEFT_TO_RIGHT, cvr::UIList::CONTINUE);
	contentBknd->addChild(_topList);
	_topList->setPercentSize(osg::Vec3(0.8, 1, .4));
	_topList->setPercentPos(osg::Vec3(0.1, 1, -.05));
	contentBknd->addChild(_botList);
	_botList->setPercentSize(osg::Vec3(0.8, 1, .4));
	_botList->setPercentPos(osg::Vec3(0.1, 1, -.50));

	int selectCount = 0;
	for (auto iter = _patientDirectories.begin(); iter != _patientDirectories.end(); ++iter)
	{
		std::string patient = iter->first;
		std::cout << patient << std::endl;

		Selection* selec;
		_selections.push_back(selec);
		_selections[selectCount] = new Selection(iter->first);
		_selections[selectCount]->setId("selection");
		_selections[selectCount]->setButtonCallback(this);
		
		selectCount < SLOTCOUNT/2 ? _topList->addChild(_selections[selectCount]) : _botList->addChild(_selections[selectCount]);
		selectCount++;


		if (selectCount == SLOTCOUNT) {
			_selectIndex = selectCount;	
			break;
		}

	}
	_currMap = &_patientDirectories;
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

void FileSelector::uiCallback(UICallbackCaller* ui){
	if (ui == _rightArrow) {
		if(_currMap->size() > _selectIndex)
			updateSelections(RIGHT);
	}
	if (ui == _leftArrow) {
		std::cout << "left arrow" << std::endl;
		if (_selectIndex > SLOTCOUNT)
			updateSelections(LEFT);
	}
	if (checkSelectionCallbacks(ui)) {
		return;
	}
	//std::cout << _patientDirectories[ui->getId()];
	//newUpdateFileSelection();
}

bool FileSelector::checkSelectionCallbacks(UICallbackCaller* item) {
	bool found = false;
	for (int i = 0; i < SLOTCOUNT; i++) {
		if (item == _selections[i]->getButton()) {
			found = true;
			std::cout << _selections[i]->getName() << std::endl;
			if(_currMap == &_patientDirectories)
				loadPatient(_selections[i]->getName());
		}
	}
	return found;
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
		std::cout << path << std::endl;
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



		osg::Image* img = new osg::Image();
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
		imgTexture->_geode->getDrawable(0)->getOrCreateStateSet()->setDefine("GRAYSCALE", true);

		_selections[i]->setImage(imgTexture);
	}
}

std::string FileSelector::getMiddleImage(int seriesIndex) {
	/////////Test with irbo1
	auto it = _seriesList.begin();
	for (unsigned int i = 0; i != seriesIndex; i++) {
		it++;
	
	}


	std::string seriesPath = it->second;
	std::cout << seriesPath << std::endl;

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

	std::cout << "middle number: " << count << std::endl;

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
			//std::cout << "fn: " << fn << std::endl;
			//std::cout << "newfn: " << entry->d_name << std::endl;
			std::string newFn = pFN + "/" + entry->d_name;
			int newIndex = loadSeriesList(newFn, indexFromDicom);
			if (newIndex > -1) {
				_seriesList[std::string(entry->d_name)] = pFN + "/" + entry->d_name;
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

	if (choice == LEFT)
		_selectIndex -= SLOTCOUNT * 2;



	auto it = _currMap->begin();
	std::advance(it, _selectIndex);

	while (slotsLeft > 0) {

		Selection* selec;
		topBotIndex < SLOTCOUNT / 2 ? selec = (Selection*)_topList->getChild(topBotIndex)
			: selec = (Selection*)_botList->getChild(topBotIndex % (SLOTCOUNT / 2));
		_currMap->size() > _selectIndex ? selec->setName(it->first)
			: selec->setName("");
		topBotIndex++;
		it++;			//path map iterator
		_selectIndex++; //path map "index"
		slotsLeft--;
	}
	
	if (_currMap == &_seriesList) {
		for (unsigned int i = 0; i < SLOTCOUNT; ++i) {
			_selections[i]->removeImage();
		}
		showDicomThumbnail();
	}
	
}

void FileSelector::loadVolume(std::string seriesPath, bool change, bool onlyVolume) {
	//Load volume
	if (change)
	{
		HelmsleyVolume::instance()->removeVolume(0, onlyVolume);
	}

	std::string maskpath = "";

	DIR* dir = opendir((seriesPath + "/mask").c_str());
	if (dir != NULL)
	{
		maskpath = seriesPath + "/mask";
		closedir(dir);
	}

	HelmsleyVolume::instance()->loadVolume(seriesPath, maskpath, onlyVolume);
	_state = CHANGE;
	addVol->setText("Change Volume");
	volumeFileSelector->setVisible(false);
}

void FileSelector::loadVolumeOnly(std::string seriesPath) {
	HelmsleyVolume::instance()->removeVolumeOnly(0);
	std::string maskpath = "";

	DIR* dir = opendir((seriesPath + "/mask").c_str());
	if (dir != NULL)
	{
		maskpath = seriesPath + "/mask";
		closedir(dir);
	}

	HelmsleyVolume::instance()->loadVolumeOnly(seriesPath, maskpath);
	_state = CHANGE;
	addVol->setText("Change Volume");
	volumeFileSelector->setVisible(false);
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

int FileSelector::checkIfPatient(std::string fn, int indexFromDicom) {
	//for every file/direct check recursively if there is a series to dicom connection
	//if true mark file/direct as patient and add to menu
	

	DIR* dir = opendir(fn.c_str());
	struct dirent* entry = readdir(dir);
	while (entry != NULL) {
		if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
			std::cout << "fn: " << fn << std::endl;
			std::cout << "newfn: " << entry->d_name << std::endl;
			std::string newFn = fn + "/" + entry->d_name;
			int newIndex = checkIfPatient(newFn, indexFromDicom);
			if (newIndex > -1) {
				if (newIndex == 1) {
					_patientDirectories[std::string(entry->d_name)] = fn+"/"+entry->d_name;
				}
				else {
					closedir(dir);
					return newIndex + 1;
				}
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

std::vector<std::string> FileSelector::getPresets() {
	std::string currPath = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.PresetsFolder", "C:/", false);
	std::string presetPath = currPath;
	DIR* dir = opendir(presetPath.c_str());
	if (dir == nullptr) {
		std::cout << "Directory not found in: " << presetPath << std::endl;
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

