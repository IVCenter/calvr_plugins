#include "PluginTest.h"

#include <iostream>
#include <menu/MenuSystem.h>
#include <menu/SubMenu.h>
#include <kernel/PluginHelper.h>
#include <kernel/InteractionManager.h>

CVRPLUGIN(PluginTest)

using namespace cvr;

PluginTest::PluginTest()
{
    std::cerr << "PluginTest created." << std::endl;
}

PluginTest::~PluginTest()
{
    std::cerr << "PluginTest destroyed." << std::endl;
    delete testButton1;
    delete testButton2;
    delete testButton3;
    delete testButton4;
    delete testButton5;
    delete menu1;
    delete menu2;
    delete menu3;
    delete pmenu1;
    delete checkbox1;
    delete rangeValue;
    delete pcheckbox1;
    delete pbutton1;
}

bool PluginTest::init()
{
    std::cerr << "PluginTest init()." << std::endl;

    menu1 = new SubMenu("My Menu1","Menu Bar1");
    menu1->setCallback(this);

    menu2 = new SubMenu("My Menu2","Menu Bar2");
    menu2->setCallback(this);

    menu3 = new SubMenu("My Menu3","Menu Bar3");
    menu3->setCallback(this);

    testButton1 = new MenuButton("Test Button 1");
    testButton1->setCallback(this);

    testButton2 = new MenuButton("Test Button 2");
    testButton2->setCallback(this);

    testButton3 = new MenuButton("Test Button 3");
    testButton3->setCallback(this);

    testButton4 = new MenuButton("Test Button 4");
    testButton4->setCallback(this);

    testButton5 = new MenuButton("Test Button 5");
    testButton5->setCallback(this);

    checkbox1 = new MenuCheckbox("Checkbox 1",true);
    checkbox1->setCallback(this);

    rangeValue = new MenuRangeValue("RangeValue 1",0.01,10.0,5.0);
    rangeValue->setCallback(this);

    textButtonSet1 = new MenuTextButtonSet(true, 400, 40, 3);
    textButtonSet1->setCallback(this);
    textButtonSet1->addButton("This");
    textButtonSet1->addButton("Is A");
    textButtonSet1->addButton("Test");
    textButtonSet1->addButton("Of");
    textButtonSet1->addButton("Multiple");
    textButtonSet1->addButton("Rows");

    menu1->addItem(testButton1);
    menu1->addItem(rangeValue);
    menu1->addItem(textButtonSet1);
    menu2->addItem(testButton3);
    menu2->addItem(menu3);
    menu2->addItem(checkbox1);
    menu3->addItem(testButton4);
    menu3->addItem(testButton5);

    MenuSystem::instance()->addMenuItem(menu1);
    MenuSystem::instance()->addMenuItem(menu2);
    MenuSystem::instance()->addMenuItem(testButton2);

    //MenuSystem::instance()->addMenuItem(testButton1);
    //MenuSystem::instance()->addMenuItem(testButton2);

    popup1 = new PopupMenu("Test Popup");
    //popup1->setVisible(true);

    pcheckbox1 = new MenuCheckbox("Popup Check", true);
    pmenu1 = new SubMenu("Popup menu line","Popup Menu head");
    pbutton1 = new MenuButton("Popup button");
    pmenu1->addItem(pbutton1);

    popup1->addMenuItem(pcheckbox1);
    popup1->addMenuItem(pmenu1);

    tdp1 = new TabbedDialogPanel(400,40,3,"Tabbed Dialog Panel");
    tdp1->setVisible(true);

    tdp1->addTextTab("Tab1","I am Tab 1");
    tdp1->addTextTab("Tab2","I am Tab 2");
    tdp1->addTextTab("Tab3","This is a test of the maxWidth attribute of osgText::Text. Hopefully if will wrap on whitespace.");

    //std::cerr << "NodeMask: " << PluginHelper::getObjectsRoot()->getNodeMask() << std::endl;

    return true;
}

void PluginTest::menuCallback(MenuItem * item)
{
    std::cerr << "Got menu item callback." << std::endl;
    if(item == testButton1)
    {
	std::cerr << "Test Button 1" << std::endl;
    }
    else if(item == testButton2)
    {
	std::cerr << "Test Button 2" << std::endl;
    }
    else if(item == testButton3)
    {
	std::cerr << "Test Button 3" << std::endl;
	for(int i = 0; i < menu3->getNumChildren(); i++)
	{
	    MenuItem * mi = menu3->getChild(i);
	    menu3->removeItem(menu3->getChild(i));
	    delete mi;
	}
	menu2->removeItem(menu3);
	delete menu3;
    }
    else if(item == testButton4)
    {
	std::cerr << "Test Button 4" << std::endl;
    }
    else if(item == testButton5)
    {
	std::cerr << "Test Button 5" << std::endl;
    }
    else if(item == checkbox1)
    {
	std::cerr << "Checkbox 1 value: " << checkbox1->getValue() << std::endl;
    }
    else if(item == rangeValue)
    {
	std::cerr << "RangeValue 1 value: " << rangeValue->getValue() << std::endl;
    }
}

void PluginTest::preFrame()
{
    //std::cerr << "PluginTest preFrame()." << std::endl;
}

void PluginTest::postFrame()
{
    //std::cerr << "PluginTest postFrame()." << std::endl;
}

bool PluginTest::keyEvent(bool keyDown, int key, int mod)
{
    std::cerr << "PluginTest keyEvent: keyDown: " << keyDown << " key: " << key << " char: " << (char)key << " mod: " << mod << std::endl;
    return false;
}

bool PluginTest::buttonEvent(int type, int button, int hand, const osg::Matrix& mat)
{
    std::cerr << "Button event type: ";
    switch(type)
    {
	case BUTTON_DOWN:
	    std::cerr << "BUTTON_DOWN ";
	    break;
	case BUTTON_UP:
	    std::cerr << "BUTTON_UP ";
	    break;
	case BUTTON_DRAG:
	    std::cerr << "BUTTON_DRAG ";
	    break;
	case BUTTON_DOUBLE_CLICK:
	    std::cerr << "BUTTON_DOUBLE_CLICK ";
	    break;
	default:
	    std::cerr << "UNKNOWN ";
	    break;
    }

    std::cerr << "hand: " << hand << " button: " << button << std::endl;
    return false;
}

bool PluginTest::mouseButtonEvent(int type, int button, int x, int y, const osg::Matrix& mat)
{
    std::cerr << "Mouse Button event type: ";
    switch(type)
    {
	case MOUSE_BUTTON_DOWN:
	    std::cerr << "MOUSE_BUTTON_DOWN ";
	    break;
	case MOUSE_BUTTON_UP:
	    std::cerr << "MOUSE_BUTTON_UP ";
	    break;
	case MOUSE_DRAG:
	    std::cerr << "MOUSE_DRAG ";
	    break;
	case MOUSE_DOUBLE_CLICK:
	    std::cerr << "MOUSE_DOUBLE_CLICK ";
	    break;
	default:
	    std::cerr << "UNKNOWN ";
	    break;
    }

    std::cerr << "button: " << button << std::endl;
    return false;
}
