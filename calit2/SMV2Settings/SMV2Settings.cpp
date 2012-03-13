#include "SMV2Settings.h"

#include <iostream>
#include <input/TrackingManager.h>
#include <config/ConfigManager.h>
#include <kernel/InteractionManager.h>
#include <kernel/PluginHelper.h>
#include <kernel/ScreenConfig.h>
#include <kernel/ScreenMVZones.h>
#include <menu/MenuSystem.h>

#include <osg/ShapeDrawable>

CVRPLUGIN(SMV2Settings)

#ifdef WIN32
#define M_PI 3.141592653589793238462643
#endif

using namespace cvr;

SMV2Settings::SMV2Settings()
{
    std::cerr << "SMV2Settings created." << std::endl;
}

SMV2Settings::~SMV2Settings()
{
    std::cerr << "SMV2Settings destroyed." << std::endl;
    delete contributionMenu;
    delete linearFunc;
    delete gaussianFunc;
    delete orientation3d;
    delete contributionVar;
    delete autoAdjust;
    delete zoneRowQuantity;
    delete zoneColumnQuantity;
    delete autoAdjustTarget;
    delete autoAdjustOffset;
    delete zoneColoring;
    delete zoneMenu;
    delete multipleUsers;
    delete mvsMenu;
}

bool SMV2Settings::init()
{
    std::cerr << "SMV2Settings init()." << std::endl;

    /*** Ensure that ScreenMVZones is in use. ***/
    ScreenConfig * sConfig = ScreenConfig::instance();
    ScreenMVZones * smv2 = NULL;
    for (int i=0; i < sConfig->getNumScreens(); i++)
    {
        smv2 = dynamic_cast<ScreenMVZones *> (sConfig->getScreen(i));
        if (smv2 != NULL)
            break;
    }
    if (smv2 == NULL)
    {
        std::cerr<<"Cannot initialize SMV2Settings without running a ScreenMVZones screen.\n";
        return false;
    }

    /*** Menu Setup ***/
    mvsMenu = new SubMenu("SMV2Settings", "SMV2Settings");
    mvsMenu->setCallback(this);

    multipleUsers = new MenuCheckbox("Multiple Users",
            ScreenMVZones::getMultipleUsers());
    multipleUsers->setCallback(this);

    mvsMenu->addItem(multipleUsers);

    contributionMenu = new SubMenu("Contribution Control", "Contribution Control");
    contributionMenu->setCallback(this);

    orientation3d = new MenuCheckbox("3D Orientation Contribution Balancing",
            ScreenMVZones::getOrientation3d());
    orientation3d->setCallback(this);

    linearFunc = new MenuCheckbox("Linear Contribution Balancing", false);
    linearFunc->setCallback(this);

    cosineFunc = new MenuCheckbox("Cosine Contribution Balancing", true);
    cosineFunc->setCallback(this);

    gaussianFunc = new MenuCheckbox("Gaussian Contribution Balancing", false);
    gaussianFunc->setCallback(this);

    autoContrVar = new MenuCheckbox("Auto Adjust Contribution Variable",
            ScreenMVZones::getAutoContributionVar());
    autoContrVar->setCallback(this);

    contributionVar = new MenuRangeValue("Contribution Variable", 1, 180,
            ScreenMVZones::getContributionVar()*180/M_PI, 1);
    contributionVar->setCallback(this);

    contributionMenu->addItem(orientation3d);
    contributionMenu->addItem(linearFunc);
    contributionMenu->addItem(cosineFunc);
    contributionMenu->addItem(gaussianFunc);
    contributionMenu->addItem(autoContrVar);
    if (!autoContrVar->getValue())
        contributionMenu->addItem(contributionVar);
    mvsMenu->addItem(contributionMenu);

    zoneMenu = new SubMenu("Zone Control", "Zone Control");
    zoneMenu->setCallback(this);

    autoAdjust = new MenuCheckbox("AutoAdjust Zones for FPS",
                    ScreenMVZones::getAutoAdjust());
    autoAdjust->setCallback(this);

    zoneRowQuantity = new MenuRangeValue("Zone Row Quantity", 1,
                    ScreenMVZones::getMaxZoneRows(),
                    ScreenMVZones::getZoneRows(), 1);
    zoneRowQuantity->setCallback(this);

    zoneColumnQuantity = new MenuRangeValue("Zone Column Quantity", 1,
                    ScreenMVZones::getMaxZoneColumns(),
                    ScreenMVZones::getZoneColumns(), 1);
    zoneColumnQuantity->setCallback(this);

    autoAdjustTarget = new MenuRangeValue("AutoAdjust FPS Target", 1, 70,
                    ScreenMVZones::getAutoAdjustTarget(), 1);
    autoAdjustTarget->setCallback(this);

    autoAdjustOffset = new MenuRangeValue("AutoAdjust FPS Offset", 0, 10,
                    ScreenMVZones::getAutoAdjustOffset(), 1);
    autoAdjustOffset->setCallback(this);

    zoneColoring = new MenuCheckbox("Zone Coloring",
                    ScreenMVZones::getZoneColoring());
    zoneColoring->setCallback(this);

    zoneMenu->addItem(zoneColoring);
    zoneMenu->addItem(autoAdjust);
    zoneMenu->addItem(autoAdjustTarget);
    zoneMenu->addItem(autoAdjustOffset);
    mvsMenu->addItem(zoneMenu);

    MenuSystem::instance()->addMenuItem(mvsMenu);
    /*** End Menu Setup ***/

    return true;
}

void SMV2Settings::menuCallback(MenuItem * item)
{
    static float linearVar = 180;
    static float cosineVar = 180;
    static float gaussianVar = 180;
    static float * contrVar = &cosineVar;

    if (item == multipleUsers)
    {
        ScreenMVZones::setMultipleUsers(multipleUsers->getValue());
    }
    else if (item == orientation3d)
    {
        ScreenMVZones::setOrientation3d(orientation3d->getValue());
    }
    else if (item == linearFunc)
    {
        ScreenMVZones::setSetContributionFunc(0);
        contrVar = &linearVar;
        contributionVar->setValue(*contrVar);

        if (!autoContrVar->getValue())
            ScreenMVZones::setContributionVar(*contrVar*M_PI/180);

        linearFunc->setValue(true);
        cosineFunc->setValue(false);
        gaussianFunc->setValue(false);
    }
    else if (item == cosineFunc)
    {
        ScreenMVZones::setSetContributionFunc(1);
        contrVar = &cosineVar;
        contributionVar->setValue(*contrVar);

        if (!autoContrVar->getValue())
            ScreenMVZones::setContributionVar(*contrVar*M_PI/180);

        linearFunc->setValue(false);
        cosineFunc->setValue(true);
        gaussianFunc->setValue(false);
    }
    else if (item == gaussianFunc)
    {
        ScreenMVZones::setSetContributionFunc(2);
        contrVar = &gaussianVar;
        contributionVar->setValue(*contrVar);

        if (!autoContrVar->getValue())
            ScreenMVZones::setContributionVar(*contrVar*M_PI/180);

        linearFunc->setValue(false);
        cosineFunc->setValue(false);
        gaussianFunc->setValue(true);
    }
    else if (item == autoContrVar)
    {
        if (autoContrVar->getValue())
        {
            contributionMenu->removeItem(contributionVar);
            menuCallback(contributionVar);
        }
        else
        {
            contributionMenu->addItem(contributionVar);
            
        }
        ScreenMVZones::setAutoContributionVar(autoContrVar->getValue());
    }
    else if (item == contributionVar)
    {
        *contrVar = contributionVar->getValue();
        ScreenMVZones::setContributionVar(*contrVar*M_PI/180);
    }
    else if (item == autoAdjust)
    {
        bool adjust = autoAdjust->getValue();
        ScreenMVZones::setAutoAdjust(adjust);

        if (adjust)
        {
            zoneMenu->addItem(autoAdjustTarget);
            zoneMenu->addItem(autoAdjustOffset);
            zoneMenu->removeItem(zoneRowQuantity);
            zoneMenu->removeItem(zoneColumnQuantity);
        }
        else
        {
            zoneMenu->removeItem(autoAdjustTarget);
            zoneMenu->removeItem(autoAdjustOffset);
            zoneMenu->addItem(zoneRowQuantity);
            zoneMenu->addItem(zoneColumnQuantity);
        }
    }
    else if (item == zoneRowQuantity)
    {
        ScreenMVZones::setZoneRows((int)(zoneRowQuantity->getValue()));
    }
    else if (item == zoneColumnQuantity)
    {
        ScreenMVZones::setZoneColumns((int)(zoneColumnQuantity->getValue()));
    }
    else if (item == autoAdjustTarget)
    {
        ScreenMVZones::setAutoAdjustTarget(autoAdjustTarget->getValue());
    }
    else if (item == autoAdjustOffset)
    {
        ScreenMVZones::setAutoAdjustOffset(autoAdjustOffset->getValue());
    }
    else if (item == zoneColoring)
    {
        ScreenMVZones::setZoneColoring(zoneColoring->getValue());
    }
}
