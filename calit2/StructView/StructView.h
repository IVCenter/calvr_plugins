#ifndef _STRUCTVIEW_PLUGIN_H_
#define _STRUCTVIEW_PLUGIN_H_

#include <vector>

#include <cvrKernel/CVRPlugin.h>

#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuCheckbox.h>

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

    bool _collabOp;
  public:
    static CVRPlugin * plugin;

    StructView();
    virtual ~StructView();
    bool init();
    void menuCallback(MenuItem*);

    virtual void message(int type, char *&data, bool collaborative=false);
};

#endif

// EOF 
