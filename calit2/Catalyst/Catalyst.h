#ifndef PLUGIN_CATALYST_H
#define PLUGIN_CATALYST_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrKernel/SceneObject.h>

#include <osg/Vec3>
#include <osgText/Font>

#include <vector>
#include <string>
#include <fstream>

#include <json/json.h>

struct PanoMetadata
{
    std::string title;
    std::string description;
    std::string leftImage;
    std::string rightImage;
};

/*
struct PanoSetMetadata
{
    std::string title;
    std::string description;
    std::vector<PanoMetadata> panos;

    // visual control data    
    osg::Vec3 bounds[4];
    osg::ref_ptr<osgText::Text> descriptionText;


    PanoSetMetadata(std::string n, std::string v, std::string sn, int ne)
    {
        title = n;
        description = v;
        setname = sn;
        panos.resize(ne);
    }
};
*/
class Catalyst : public cvr::CVRPlugin, public cvr::MenuCallback 
{
    public:
        Catalyst();
        virtual ~Catalyst();
        int getPriority() { return 50; } // want to consume all the events (a way to disable mouse etc itneraction)
        bool init();
        //void preFrame();
        //bool processEvent(cvr::InteractionEvent * event);
        void menuCallback(cvr::MenuItem * item);
        void message(int type, char *&data, bool collaborative=false);



    protected:

        // menu elements
        cvr::SubMenu * _catalystMenu, * _catalystloadMenu;
        cvr::MenuCheckbox * _descriptionToggle;
        cvr::MenuButton * _removeButton;

        std::map<cvr::MenuItem* , PanoMetadata> _catalystFileMap;
        cvr::SceneObject* _descriptionMenu;
        osg::PositionAttitudeTransform* _description;

        std::string _cacheDirectory;
        std::string _remoteMount;

        // check local cache
        bool checkCache(std::string fileName);

        // create on the fly to replace old onei TODO make it intersectable SceneObject (user can move it)
        void createDescriptionPanel(osg::PositionAttitudeTransform* parent, PanoMetadata& data, osg::Vec3 pos, osg::Vec4 textColor, osg::Vec4 frameColor, float fontSize);
     
        // helper functions 
        osg::Geode* createIcon(osg::Vec3 position, osg::Texture2D* texture, bool enableBlending = false);
        void parseMetaData(std::string filename, std::map<cvr::MenuItem* , PanoMetadata> &,cvr::SubMenu *);
        osg::Geode* setSelectionBound(int selectIndex );
        void remove();
        //void adjustFade(osg::PositionAttitudeTransform* parent, float fade);
        //void fadeTransition(float timeBetweenFades);
        //void enableDemoMode(bool enable);
        //void enableStereo(bool enable);
        
        // general font
        osg::ref_ptr<osgText::Font> _font;
        float _fontSize;
        osg::Vec4 _textColor;
        osg::Vec4 _frameColor;

        // textures to use
        osg::ref_ptr<osg::Texture2D> _logoHeaderTexture;
        osg::ref_ptr<osg::Texture2D> _logoFooterTexture;
        osg::ref_ptr<osg::Texture2D> _backTexture;

        // created due to being used in multiple locations        
        osg::ref_ptr<osg::Geode> _logoGeode;

        // decription position
        osg::Vec3 _descriptionPos;

        // default directory
        std::string _directory;
};

#endif
