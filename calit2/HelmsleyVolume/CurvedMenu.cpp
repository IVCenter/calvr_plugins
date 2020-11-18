#include "CurvedMenu.h"
CurvedMenu::CurvedMenu(UICallback* callback, int numItems) : UIElement() {
	_numItems = numItems;

	//osg::Vec3 size = osg::Vec3(1, 1, 1);
	
	//_parent->setAbsoluteSize(size);
	_list = new cvr::UIList(cvr::UIList::LEFT_TO_RIGHT, cvr::UIList::CUT);

	osg::ref_ptr<osg::Vec3Array> debugColors = new osg::Vec3Array();
	debugColors->push_back(osg::Vec3(1, 0, 0));
	debugColors->push_back(osg::Vec3(1, 1, 0));
	debugColors->push_back(osg::Vec3(1, 0, 1));
	debugColors->push_back(osg::Vec3(0, 1, 1));

	for (int i = 0; i < numItems; i++) {

		//if (i % 2 == 0) {
		osg::Vec4 debugColor = osg::Vec4(debugColors->at(i % 4), 1);

		CurvedQuad* segment = new CurvedQuad(i, numItems, UI_BACKGROUND_COLOR);
		segment->setCallback(callback);
		//segment->setPercentSize(osg::Vec3(numItems, 1.0, 1.0));
		//
		//segment->setPercentSize(osg::Vec3(1.0/(float)numItems, 1.0, 1.0));
		//float spaceBetween = 1 / (float)numItems;// +1 / numItems / numItems;
		//segment->setPercentPos(osg::Vec3((1/(float)numItems * i) - .5, 0.0, 0.0));
		//_parent->addChild(segment);
		_list->addChild(segment);
		//}
	}

	this->addChild(_list);
}

void CurvedMenu::setImage(int index, std::string iconPath) {
	cvr::UITexture* uitext = new cvr::UITexture(iconPath);
	uitext->setTransparent(true);
	uitext->setPercentPos(osg::Vec3(0.1, 0.0, .4));
	uitext->setPercentSize(osg::Vec3(0.7, 1.0, .8));

	_list->getChild(index)->addChild(uitext);
}