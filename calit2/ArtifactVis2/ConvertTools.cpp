#include "ConvertTools.h"

using namespace cvr;
using namespace std;
using namespace osg;

ConvertTools::ConvertTools(std::string name)
{
  _active = false;

}
ConvertTools::~ConvertTools()
{

}
void ConvertTools::saveModelConfig(std::string name,std::string path,std::string filename,std::string q_filetype,std::string q_type,std::string q_group,osg::Vec3 pos,osg::Quat rot, float scaleFloat, bool newConfig)
{
   // string name = saveModel->name;
   // string path = getPathFromFilePath(saveModel->fullpath);
   // string filename = saveModel->filename;
    size_t found=name.find(".");
    string filetype;
    string file;
            if (found!=string::npos)
	    {
                 int start = int(found);
                 filetype = name;
                 filetype.erase(0,(start+1)); 
                 if(filetype == "kml")
                 {
                  filetype = q_filetype;
                  file = path;
                  file.append(name);
                 }
                 else
                 {
                 filename = name;
                 name.erase((start+1),4);
                 name.append("kml");
                 file = path;
                 file.append(name);
                 }                

	     }
if(newConfig)
{
/*
  if(manualEnterName != "")
  {
     name = manualEnterName;
  }
*/
     string newFile;
     bool nameExists = true;
     name.erase((name.length()-4),4);
  //   cerr << "Name : " << name << "Path: " << path << endl;
     string tempName = "";
     int inc = 0;
     while(nameExists)
     {          
                 tempName = name;
                 std:stringstream ss;
                 ss << inc;
                 tempName.append("_");
                 tempName.append(ss.str());
                 tempName.append(".kml");
                 newFile = path;
                 newFile.append(tempName);
                 inc++;
       
           if(!modelExists(newFile.c_str())) nameExists = false;
     }
     name = tempName;
     file = newFile;
//cerr << "newFile: " << file << "\n";

}


//Create Placemarks
//string q_type = saveModel->modelType;
//string q_group = saveModel->group;
//Vec3 pos;
//Quat rot;
//float scaleFloat;
/*
if(q_type == "Model")
{
pos = saveModel->so->getPosition();
rot = saveModel->so->getRotation();
scaleFloat = saveModel->so->getScale();
}
else
{
pos = saveModel->pcObject->getPosition();
rot = saveModel->pcObject->getRotation();
scaleFloat = saveModel->pcObject->getScale();
}
*/
cerr << "NewFile: " << file << endl;

saveTo3Dkml(name, filename, file, filetype, pos, rot, scaleFloat, q_type, q_group);
}


void ConvertTools::saveTo3Dkml(string name,string filename, string file, string filetype, Vec3 pos, Quat rot, float scaleFloat,string q_type, string q_group) 
{

    mxml_node_t *xml;    /* <?xml ... ?> */
    mxml_node_t *kml;   /* <kml> */
    mxml_node_t *document;   /* <Document> */
    mxml_node_t *nameKML;   /* <name> */
    mxml_node_t *filetypeKML;   /* <name> */
    mxml_node_t *open;   /* <name> */
    mxml_node_t *type;   /* <type> */
    mxml_node_t *timestamp;   /* <timestamp> */

    mxml_node_t *placemark;   /* <Placemark> */
    mxml_node_t *description;   /* <description> */

    mxml_node_t *lookat;   /* <LookAt> */
    mxml_node_t *longitude;   /* <data> */
    mxml_node_t *latitude;   /* <data> */
    mxml_node_t *altitude;   /* <data> */
    mxml_node_t *range;   /* <data> */
    mxml_node_t *tilt;   /* <data> */
    mxml_node_t *heading;   /* <data> */
    mxml_node_t *w;   /* <data> */
    mxml_node_t *styleurl;   /* <data> */
    mxml_node_t *altitudeMode;   /* <data> */
    mxml_node_t *group;   /* <data> */
    mxml_node_t *model;   /* <data> */
    mxml_node_t *orientation;   /* <data> */
    mxml_node_t *scale;   /* <data> */
    mxml_node_t *x;   /* <data> */
    mxml_node_t *y;   /* <data> */
    mxml_node_t *z;   /* <data> */
    mxml_node_t *link;   /* <data> */
    mxml_node_t *href;   /* <data> */
    mxml_node_t *resourceMap;   /* <data> */

//Create KML Container

//KML Name
    string q_name = name;
   // string g_timestamp = getTimeStamp();
    string g_timestamp = "00";

   const char* kml_name = q_name.c_str();
   const char* kml_timestamp = g_timestamp.c_str();

xml = mxmlNewXML("1.0");
        kml = mxmlNewElement(xml, "kml");
            document = mxmlNewElement(kml, "Document");
                nameKML = mxmlNewElement(document, "name");
                  mxmlNewText(nameKML, 0, kml_name);
                open = mxmlNewElement(document, "open");
                  mxmlNewText(open, 0, "1");
                timestamp = mxmlNewElement(document, "timestamp");
                  mxmlNewText(timestamp, 0, kml_timestamp);
//.................................................................
//Get Placemarks





   //Get Comments Description
   string q_description = "";

stringstream buffer;
   buffer << pos.x();
   string q_longitude = buffer.str();
   buffer.str("");
   buffer << pos.y();
   string q_latitude = buffer.str();
   buffer.str("");
   buffer << pos.z();
   string q_altitude = buffer.str();
   buffer.str("");
   buffer << rot.x();
   string q_x = buffer.str();
   buffer.str("");
   buffer << rot.y();
   string q_y = buffer.str();
   buffer.str("");
   buffer << rot.z();
   string q_z = buffer.str();
   buffer.str("");
   buffer << rot.w();
   string q_w = buffer.str();
   buffer.str("");
   buffer << scaleFloat;
   string scaleTemp = buffer.str();
   buffer.str("");
   string q_scaleX = scaleTemp;
   string q_scaleY = scaleTemp;
   string q_scaleZ = scaleTemp;

   string q_href = filename;

                placemark = mxmlNewElement(document, "Placemark");
                    nameKML = mxmlNewElement(placemark, "name");
                      mxmlNewText(nameKML, 0, q_name.c_str());
                    type = mxmlNewElement(placemark, "type");
                      mxmlNewText(type, 0, q_type.c_str());
                    filetypeKML = mxmlNewElement(placemark, "filetype");
                      mxmlNewText(filetypeKML, 0, filetype.c_str());
                    group = mxmlNewElement(placemark, "group");
                      mxmlNewText(group, 0, q_group.c_str());
                    styleurl = mxmlNewElement(placemark, "styleUrl");
                      mxmlNewText(styleurl, 0, "#msn_GR");

                    description = mxmlNewElement(placemark, "description");
                      mxmlNewText(description, 0, q_description.c_str());
                    model = mxmlNewElement(placemark, "Model");
                        altitudeMode = mxmlNewElement(model, "altitudeMode");
                          mxmlNewText(altitudeMode, 0, "absolute");
                    
                    lookat = mxmlNewElement(model, "Location");
                        longitude = mxmlNewElement(lookat, "longitude");
                          mxmlNewText(longitude, 0, q_longitude.c_str());
                        latitude = mxmlNewElement(lookat, "latitude");
                          mxmlNewText(latitude, 0, q_latitude.c_str());
                        altitude = mxmlNewElement(lookat, "altitude");
                          mxmlNewText(altitude, 0, q_altitude.c_str());
                    orientation = mxmlNewElement(model, "Orientation");
                        range = mxmlNewElement(orientation, "heading");
                          mxmlNewText(range, 0, q_x.c_str());
                        tilt = mxmlNewElement(orientation, "tilt");
                          mxmlNewText(tilt, 0, q_y.c_str());
                        heading = mxmlNewElement(orientation, "roll");
                          mxmlNewText(heading, 0, q_z.c_str());
                        w = mxmlNewElement(orientation, "w");
                          mxmlNewText(w, 0, q_w.c_str());
                    scale = mxmlNewElement(model, "Orientation");
                        x = mxmlNewElement(scale, "x");
                          mxmlNewText(x, 0, q_scaleX.c_str());
                        y = mxmlNewElement(scale, "y");
                          mxmlNewText(y, 0, q_scaleY.c_str());
                        z = mxmlNewElement(scale, "z");
                          mxmlNewText(z, 0, q_scaleZ.c_str());
                    link = mxmlNewElement(model, "Link");
                        href = mxmlNewElement(link, "href");
                          mxmlNewText(href, 0, q_href.c_str());
//.......................................................
//Save File
  const char *ptr;
    ptr = "";
  ptr = mxmlSaveAllocString(xml, MXML_NO_CALLBACK);
    //cout << ptr;
    FILE *fp;
    
    filename = file;
    kml_name = filename.c_str();
    fp = fopen(kml_name, "w");

    fprintf(fp, ptr);

    fclose(fp);
 
cerr << "Saved File\n";
}
std::string ConvertTools::getPathFromFilePath(string filepath)
{

                //Get Full path
 size_t found=filepath.find_last_of("/");
    string path;
            if (found!=string::npos)
	    {
                 int start = int(found);
                 path = filepath;
                 path.erase(start,(path.length()-start));
                 path.append("/"); 
                  // cerr << "path: " << path << endl;
            }
return path;
}
bool ConvertTools::modelExists(const char* filename)
{
    ifstream ifile(filename);
    return !ifile.fail();
}
