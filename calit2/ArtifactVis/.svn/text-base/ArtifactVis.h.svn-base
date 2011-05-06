#ifndef _ARTIFACTVIS_
#define _ARTIFACTVIS_

#include <kernel/CVRPlugin.h>
#include <menu/SubMenu.h>
#include <menu/MenuCheckbox.h>

#include <osg/Material>
#include <osg/MatrixTransform>

#include <string>
#include <vector>

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
    osg::Node* geode;  ///< The actual object... when it is made.
    
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
        void menuCallback(cvr::MenuItem * item);

    protected:
        cvr::MenuCheckbox *showCheckbox;
        cvr::SubMenu *avMenu;
        osg::MatrixTransform *root;

        std::string configPath;

        std::vector<Artifact*> _artifacts;    ///< container for artifacts
        std::vector<std::string> _descriptor_list;
        std::vector<osg::Vec4> _descriptor_list_colors;

        osg::LOD * _my_own_root;
        osg::Group * _my_sphere_root;
        float _LODmaxRange;

        void readArtifactsFile(std::string);
        void listArtifacts();
        void displayArtifacts(osg::Group * root_node);
        osg::Node* createObject(int index, int edm, float tessellation, osg::Vec3f & pos);

};

#endif
