#ifndef _STRUCTVIEW_PLUGIN_H_
#define _STRUCTVIEW_PLUGIN_H_

#include <vector>

#include <kernel/CVRPlugin.h>

#include <menu/SubMenu.h>
#include <menu/MenuCheckbox.h>

using namespace cvr;

/** Plugin to load StructView data
  @author Andre Barbosa (abarbosa@ucsd.edu) based on work by Jurgen Schulze (jschulze@ucsd.edu)
*/
class StructView : public CVRPlugin, public MenuCallback
{
  protected:
    osg::ref_ptr<osg::Switch> mainNode;
    std::vector<MenuCheckbox*> menuList;
    SubMenu* structViewMenu;
    MenuCheckbox* enable;
  public:
    static CVRPlugin * plugin;

    StructView();
    virtual ~StructView();
    bool init();
    void menuCallback(MenuItem*);
};

#endif

// EOF 
