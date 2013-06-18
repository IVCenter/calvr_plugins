#include "LayoutManager.h"

#include <cvrKernel/SceneManager.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrKernel/PluginHelper.h>
#include <PluginMessageType.h>

#include <iostream>

using namespace osg;
using namespace std;
using namespace cvr;

CVRPLUGIN(LayoutManager)

/*virtual*/ int LayoutManager::getPriority()
{
    return 90;
}

bool LayoutManager::init()
{
    std::cerr << "LayoutManager init" << std::endl;

    mActiveLayout = NULL;

    mLMMenu = new SubMenu("LayoutManager", "LayoutManager");
    mLMMenu->setCallback(this);

    std::cerr << "LayoutManager init done.\n";

    return true;
}


LayoutManager::~LayoutManager()
{
}

void LayoutManager::menuCallback(MenuItem* menuItem)
{
    if (!mActiveLayout)
    {
        for (ButtonLayoutMap::iterator i = mMenuLayoutsMap.begin();
            mMenuLayoutsMap.end() != i;
            ++i)
        {
            if(i->first == menuItem)
            {
                mActiveLayout = i->second;

                if (!mActiveLayout->Start())
                {
                    std::cerr << "Failed to start layout:\t" << mActiveLayout->Name() << std::endl;
                    mActiveLayout = NULL;
                }
	    }
        }
    }
}

void LayoutManager::preFrame()
{
    if (mActiveLayout)
    {
        if (mActiveLayout->Update())
        {
            mActiveLayout->Cleanup();
            mActiveLayout = NULL;
        }
    }
}

void LayoutManager::message(int type, char * &data, bool collaborative)
{
    if (LM_ADD_LAYOUT == type)
    {
        LayoutManagerAddLayoutData* lmald = (LayoutManagerAddLayoutData*)data;
        Layout* layout = lmald->layout;

	MenuButton* button = new MenuButton( layout->Name() );
	button->setCallback(this);

	mMenuLayoutsMap[ button ] = layout;

	mLMMenu->addItem( button );

        if (1 == mLMMenu->getNumChildren())
            MenuSystem::instance()->addMenuItem(mLMMenu);
    }
}

