#ifndef _LAYOUT_MANAGER_H_
#define _LAYOUT_MANAGER_H_

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/MenuButton.h>

#include <PluginMessageType.h>

#include <map>
#include <string>

class LayoutManager : public cvr::MenuCallback, public cvr::CVRPlugin
{
public:        

    virtual ~LayoutManager();

    bool init();

    void menuCallback(cvr::MenuItem * item);
    void preFrame();

    virtual int getPriority();

    virtual void message(int type, char * &data, bool collaborative=false);

protected:
    cvr::SubMenu * mLMMenu;

    typedef std::map<cvr::MenuButton*, Layout*> ButtonLayoutMap;
    ButtonLayoutMap mMenuLayoutsMap;

    Layout* mActiveLayout;
};

#endif /*_LAYOUT_MANAGER_H_*/

