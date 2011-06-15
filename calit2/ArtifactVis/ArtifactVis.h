#ifndef _ARTIFACTVIS_
#define _ARTIFACTVIS_

#include <kernel/CVRPlugin.h>
#include <menu/SubMenu.h>
#include <menu/MenuCheckbox.h>
#include <menu/MenuButton.h>
#include <menu/TabbedDialogPanel.h>
#include <menu/DialogPanel.h>

#include <osg/Material>
#include <osg/MatrixTransform>

#include <string>
#include <vector>
#include <map>

/** Descriptor for a single artifact
Order of variables:
EDM	DCCODE	LOCUS	BASKET	SQUARE	DATES	AREA	SITE	NORTHING	EASTING	ELEVATION	STRATUM	OPENED	CLOSED	SQUARES IN LOCUS	1ST BASKET	DESCRIPTION
*/
class Artifact
{
  public:
    int edm;            ///< electronic distance measure (unique per artifact)
    std::string dc;     ///< descriptor code
    int locus;          ///< artifact site within excavated area
    int basket;         ///< unique artifact ID (multiple sherds of same vase share same basket number)
    std::string square; ///< grid square (two letters for first axis, two digits for second axis)
    std::string date;   ///< date artifact was discovered
    char area;          ///< excavated area with site
    std::string site;   ///< excavation site
    double pos[3];      ///< 3D location of artifact (northing, easting, elevation) in meters
    //osg::Node* geode;  ///< The actual object... when it is made.
    osg::Drawable * drawable;
    osg::Vec3 modelPos;
    bool visible;
    bool selected;
    
    Artifact()
    {
      pos[0] = 0.0;
      pos[1] = 0.0;
      pos[2] = 0.0;
    }
};


class ArtifactVis : public cvr::MenuCallback, public cvr::CVRPlugin
{
    public:        
        ArtifactVis();
        virtual ~ArtifactVis();

	bool init();
        bool buttonEvent(int type, int button, int hand, const osg::Matrix & mat);
        bool mouseButtonEvent(int type, int button, int x, int y, const osg::Matrix & mat);
        void menuCallback(cvr::MenuItem * item);
        void preFrame();

        void setDCVisibleStatus(std::string dc, bool status);
        void updateVisibleStatus();

    protected:
        struct Locus
        {
            int id;
            std::vector<osg::Vec3d> coords;
        };

        cvr::MenuCheckbox *_showSiteCB;
        cvr::MenuCheckbox *_showSpheresCB;
        cvr::SubMenu *_avMenu;
        cvr::MenuCheckbox * _selectArtifactCB;
        cvr::MenuCheckbox * _selectCB;
        cvr::SubMenu * _dcFilterMenu;
        cvr::MenuCheckbox * _dcFilterAuto;
        cvr::MenuButton * _dcFilterShowAll;
        cvr::MenuButton * _dcFilterShowNone;
        std::vector<cvr::SubMenu *> _dcFilterSubMenus;
        std::vector<cvr::MenuCheckbox *> _dcFilterItems;
        osg::MatrixTransform * _root;

        cvr::TabbedDialogPanel * _artifactPanel;
        cvr::DialogPanel * _selectionStatsPanel;

        std::string _picFolder;

        std::vector<Artifact*> _artifacts;    ///< container for artifacts
        std::vector<std::string> _descriptor_list;
        std::vector<osg::Vec4> _descriptor_list_colors;

        std::vector<std::string> _dcList;
        std::map<std::string,bool> _dcVisibleMap;

        std::vector<Locus*> _locusList;

        osg::Vec3 _selectStart;
        osg::Vec3 _selectCurrent;

        osg::Material * _defaultMaterial;

        //osg::LOD * _my_own_root;
        osg::ref_ptr<osg::MatrixTransform> _sphereRoot;
        osg::ref_ptr<osg::MatrixTransform> _siteRoot;
        osg::ref_ptr<osg::MatrixTransform> _selectBox;
        osg::ref_ptr<osg::MatrixTransform> _selectMark;
        //float _LODmaxRange;
        float _sphereRadius;
        int _activeArtifact;
        float _filterTime;
        bool _selectActive;

        void setActiveArtifact(int art);
        void readArtifactsFile(std::string);
        void listArtifacts();
        void displayArtifacts(osg::Group * root_node);
        osg::Drawable* createObject(int index, float tessellation, osg::Vec3f & pos);
        void readSiteFile();
        void readLocusFile();
        void setupDCFilter();
        void updateSelect();
};

#endif
