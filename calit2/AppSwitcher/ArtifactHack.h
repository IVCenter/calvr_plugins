#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/ComController.h>
#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/InteractionEvent.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginHelper.h>

#include <osgEarth/Map>


#include <cvrMenu/MenuText.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuRangeValue.h>
#include <string>

class DistanceMenu : public cvr::SubMenu
{
  protected:
    double distance;
    std::string _partial_name;

  public:
    DistanceMenu(std::string name, double distance);
    virtual ~DistanceMenu();
    
    void setDistance(double distance);
};

class ArtifactHack : public cvr::CVRPlugin
{
    cvr::SubMenu*  menu_root;
    DistanceMenu*  menu_artifact;
    cvr::MenuText* menu_artifactText;
    cvr::MenuText* menu_flyerText;
    cvr::MenuRangeValue* menu_flyerrange;

    void makeMenu();

    void update_rangemenu();

  public:
    virtual ~ArtifactHack();

    virtual bool init();
    virtual void preFrame();

    virtual bool processEvent(cvr::InteractionEvent* event);

};

