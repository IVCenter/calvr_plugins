#ifndef _OSGCHROMOSOME_
#define _OSGCHROMOSOME_

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/SceneObject.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>

//#include <string>
//#include <vector>

class OsgChromosome : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:        
        OsgChromosome();
        ~OsgChromosome();
        void menuCallback(cvr::MenuItem * item);
        void AddCylinderBetweenPoints(osg::Vec3	& StartPoint, osg::Vec3 & EndPoint, float radius, osg::Vec4	CylinderColor, cvr::SceneObject * pAddToThisGroup);
	    bool init();

    protected:
        cvr::SubMenu * mymenu;
        cvr::MenuButton * clear;
        cvr::MenuButton * chr1; 
        cvr::MenuButton * chr2;
        cvr::MenuButton * chr3;
        cvr::MenuButton * chr4;
        cvr::MenuButton * chr5;
        cvr::MenuButton * chr6;
        cvr::MenuButton * chr7;
        cvr::MenuButton * chr8;
        cvr::MenuButton * chr9;
        cvr::MenuButton * chr10;
        cvr::MenuButton * chr11;
        cvr::MenuButton * chr12;
        cvr::MenuButton * chr13;
        cvr::MenuButton * chr14;
        cvr::MenuButton * chr15;
        cvr::MenuButton * chr16;
        cvr::MenuButton * chr17;
        cvr::MenuButton * chr18;
        cvr::MenuButton * chr19;
        cvr::MenuButton * chr20;
        cvr::MenuButton * chr21;
        cvr::MenuButton * chr22;
        cvr::MenuButton * chr23;

        void renderChromosome(char * file);

        std::string basepath;

};

#endif
