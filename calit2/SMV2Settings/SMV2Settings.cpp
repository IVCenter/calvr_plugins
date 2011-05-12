#include "SMV2Settings.h"

#include <iostream>
#include <input/TrackingManager.h>
#include <config/ConfigManager.h>
#include <kernel/InteractionManager.h>
#include <kernel/PluginHelper.h>
#include <kernel/ScreenConfig.h>
#include <kernel/ScreenMultiViewer2.h>
#include <menu/MenuSystem.h>

#include <osg/ShapeDrawable>

CVRPLUGIN(SMV2Settings)

using namespace cvr;

SMV2Settings::SMV2Settings()
{
    std::cerr << "SMV2Settings created." << std::endl;
}

SMV2Settings::~SMV2Settings()
{
    std::cerr << "SMV2Settings destroyed." << std::endl;
    delete mvsMenu;
    delete multipleUsers;
    delete contributionMenu;
    delete linearFunc;
    delete gaussianFunc;
    delete orientation3d;
    delete contributionVar;
    delete zoneMenu;
    delete autoAdjust;
    delete zoneRowQuantity;
    delete zoneColumnQuantity;
    delete autoAdjustTarget;
    delete autoAdjustOffset;
}

bool SMV2Settings::init()
{
    std::cerr << "SMV2Settings init()." << std::endl;

    /*** Menu Setup ***/
    mvsMenu = new SubMenu("SMV2Settings", "SMV2Settings");
    mvsMenu->setCallback(this);

    multipleUsers = new MenuCheckbox("Multiple Users",
            ScreenMultiViewer2::getMultipleUsers());
    multipleUsers->setCallback(this);

    mvsMenu->addItem(multipleUsers);

    contributionMenu = new SubMenu("Contribution Control", "Contribution Control");
    contributionMenu->setCallback(this);

    orientation3d = new MenuCheckbox("3D Orientation Contribution Balancing",
            ScreenMultiViewer2::getOrientation3d());
    orientation3d->setCallback(this);

    linearFunc = new MenuCheckbox("Linear Contribution Balancing", false);
    linearFunc->setCallback(this);

    cosineFunc = new MenuCheckbox("Cosine Contribution Balancing", true);
    cosineFunc->setCallback(this);

    gaussianFunc = new MenuCheckbox("Gaussian Contribution Balancing", false);
    gaussianFunc->setCallback(this);

    contributionVar = new MenuRangeValue("Contribution Variable", 1, 180,
            ScreenMultiViewer2::getContributionVar(), 1);
    contributionVar->setCallback(this);

    contributionMenu->addItem(orientation3d);
    contributionMenu->addItem(linearFunc);
    contributionMenu->addItem(cosineFunc);
    contributionMenu->addItem(gaussianFunc);
    contributionMenu->addItem(contributionVar);
    mvsMenu->addItem(contributionMenu);

    zoneMenu = new SubMenu("Zone Control", "Zone Control");
    zoneMenu->setCallback(this);

    autoAdjust = new MenuCheckbox("AutoAdjust Zones for FPS",
                    ScreenMultiViewer2::getAutoAdjust());
    autoAdjust->setCallback(this);

    zoneRowQuantity = new MenuRangeValue("Zone Row Quantity", 1,
                    ScreenMultiViewer2::getMaxZoneRows(),
                    ScreenMultiViewer2::getZoneRows(), 1);
    zoneRowQuantity->setCallback(this);

    zoneColumnQuantity = new MenuRangeValue("Zone Column Quantity", 1,
                    ScreenMultiViewer2::getMaxZoneColumns(),
                    ScreenMultiViewer2::getZoneColumns(), 1);
    zoneColumnQuantity->setCallback(this);

    autoAdjustTarget = new MenuRangeValue("AutoAdjust FPS Target", 1, 70,
                    ScreenMultiViewer2::getAutoAdjustTarget(), 1);
    autoAdjustTarget->setCallback(this);

    autoAdjustOffset = new MenuRangeValue("AutoAdjust FPS Offset", 0, 10,
                    ScreenMultiViewer2::getAutoAdjustOffset(), 1);
    autoAdjustOffset->setCallback(this);

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
    static float linearVar = 90;
    static float cosineVar = 90;
    static float gaussianVar = 90;
    static float * contrVar = &cosineVar;

    if (item == multipleUsers)
    {
        ScreenMultiViewer2::setMultipleUsers(multipleUsers->getValue());
    }
    else if (item == orientation3d)
    {
        ScreenMultiViewer2::setOrientation3d(orientation3d->getValue());
    }
    else if (item == linearFunc)
    {
        ScreenMultiViewer2::setSetContributionFunc(0);
        contrVar = &linearVar;
        contributionVar->setValue(*contrVar);
        ScreenMultiViewer2::setContributionVar(*contrVar);
        linearFunc->setValue(true);
        cosineFunc->setValue(false);
        gaussianFunc->setValue(false);
    }
    else if (item == cosineFunc)
    {
        ScreenMultiViewer2::setSetContributionFunc(1);
        contrVar = &cosineVar;
        contributionVar->setValue(*contrVar);
        ScreenMultiViewer2::setContributionVar(*contrVar);
        linearFunc->setValue(false);
        cosineFunc->setValue(true);
        gaussianFunc->setValue(false);
    }
    else if (item == gaussianFunc)
    {
        ScreenMultiViewer2::setSetContributionFunc(2);
        contrVar = &gaussianVar;
        contributionVar->setValue(*contrVar);
        ScreenMultiViewer2::setContributionVar(*contrVar);
        linearFunc->setValue(false);
        cosineFunc->setValue(false);
        gaussianFunc->setValue(true);
    }
    else if (item == contributionVar)
    {
        *contrVar = contributionVar->getValue();
        ScreenMultiViewer2::setContributionVar(*contrVar);
    }
    else if (item == autoAdjust)
    {
        bool adjust = autoAdjust->getValue();
        ScreenMultiViewer2::setAutoAdjust(adjust);

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
        ScreenMultiViewer2::setZoneRows((int)(zoneRowQuantity->getValue()));
    }
    else if (item == zoneColumnQuantity)
    {
        ScreenMultiViewer2::setZoneColumns((int)(zoneColumnQuantity->getValue()));
    }
    else if (item == autoAdjustTarget)
    {
        ScreenMultiViewer2::setAutoAdjustTarget(autoAdjustTarget->getValue());
    }
    else if (item == autoAdjustOffset)
    {
        ScreenMultiViewer2::setAutoAdjustOffset(autoAdjustOffset->getValue());
    }
}
