#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdio.h>
#include <time.h>
//#include <tchar.h>
#include <cstdlib>
#include <libpq-fe.h>
//Inorder to compile with GCC you must also include the library: libpq.lib
#include <string>
#include <sstream>
#include <vector>
#include <mxml.h>
//During compilation you need to include: -lmxml


using namespace std;

PGconn *connectDB(string param);

PGresult *queryPG(string query, string serverName);

vector< vector<string> > queryPostGIS(string query, string list, string type, string srid);

void newtablePG(string table);

void dropTable(string table);

vector<string> splitSTR(string tmp, char lim);

vector<string> fileIntoArray(const char* file);

vector< vector<string> > parseConfigFile();

void importCSV(string sfile, string table, string type, string delim);

void importPointData(string table, vector< vector<string> > data );

void importPolygonData(string table, vector< vector<string> > data );

void importTableData(string table, vector< vector<string> > data, string fields );

int newStdTable(string tableS, string fields);

int newPointTable(string tableS);

int newPolygonTable(string tableL);

void runDemo();

void importDemos();

void commandLineParse(int argc, char *argv[]);

void parametersHelp();

vector< vector<string> > getArrayofQueryFunction(string table, string where);

void outputPostGISarray(vector< vector<string> > rowdata, string list);

void kmlFromQuery(string table, string where, string name);

void makePointKML(vector< vector<string> > rowdata, string list, string where, string q_name);

void makePolygonKML(vector< vector<string> > rowdata, string list, string where, string q_name);

string outputDescription(vector< vector<string> > rowdata, string list, int row);

string tableType(string table);

void printQueryResults(string table, string where);

void listTables(string table);

void getDistinctQuery(string table);

void syncLogon(string table);

void syncTable(string table);

string connectParam(string serverName);

vector< vector<string> > parseWhere(string where);

bool preventOverwrite(string name);

string getNxtQuery(string type);

string generateQName(string where, string type);

mxml_node_t * getTree(string filename);

vector<string> getNodeArray(string filename, const char* node_name);

void appendStringXML(string filename, const char* node_name, string str);

void resetStringXML(string filename, const char* node_name, string str);

void renameQfile(string file, string name);

void deleteQfile(string name);

vector<string> makePolyhedron(string temp, string locus);

double findBottom(string locus);

string getLocusColor(string locus,string table);

string getLocusName(string locus, string table);

void convertKML(string filename);

string outputDescriptionBase(string gid, string the_geom, string list);

string getTimeStamp();

int     main(int argc, char *argv[]) {
//In main we will run all the common queries needed for Archfield and common outputs to GE or CalVR.

commandLineParse(argc, argv);
//..
//cin.get();

}
void convertKML(string filename)
{

//const char* node_name = "placemark";
//vector<string>array;

    mxml_node_t * tree;
    //string tableName;
    tree = getTree(filename);
    int len = filename.length() - 4;
    string kmlfile = filename.substr(0,len);
    //Make Time Stamp
       string timestamp = getTimeStamp();


       mxml_node_t *Document;

       mxml_node_t *query;
       mxml_node_t *time;
       mxml_node_t *nameKML;

       nameKML = mxmlFindElement(tree, tree, "name", NULL, NULL, MXML_DESCEND);
       if (nameKML != NULL)
       {
                cout << "Name Exists!\n";
                //mxmlSetText(nameKML, 0,kmlfile.c_str());

                mxmlSetText(nameKML->child, 0,kmlfile.c_str());

       }


       Document = mxmlFindElement(tree, tree, "Document", NULL, NULL, MXML_DESCEND);


            query = mxmlNewElement(Document, "query");
                mxmlNewText(query, 0,"Not from Query");

           time = mxmlNewElement(Document, "timestamp");
                          mxmlNewText(time, 0,timestamp.c_str());


mxml_node_t * node;
string coord;
string pname;
string placemarkname;
    //for (node = mxmlFindElement(tree, tree, "Point", NULL, NULL, MXML_DESCEND); node != NULL; node = mxmlFindElement(node, tree, "Point", NULL, NULL, MXML_DESCEND))
    mxml_node_t * check;
        check = mxmlFindElement(tree, tree, "description", NULL, NULL, MXML_DESCEND);

        if (check != NULL)
        {
            //cout << "Exists!\n";
            //mxml_node_t * upname = mxmlWalkPrev(check,tree,0);
            placemarkname = check->parent->value.element.name;

            //cout << placemarkname << "\n";
        }
string description;
string the_geom;
string list;
int r = 0;

    for (node = mxmlFindElement(tree, tree, placemarkname.c_str(), NULL, NULL, MXML_DESCEND); node != NULL; node = mxmlFindElement(node, tree, placemarkname.c_str(), NULL, NULL, MXML_DESCEND))

    {

        mxml_node_t * desc_node;
        desc_node = mxmlFindElement(node, tree, "Point", NULL, NULL, MXML_DESCEND);
        cout << "Pass: " << r << "\n";
        string inc;
        std::stringstream ss;
        //ss.precision(19);
        ss << r;
        inc = ss.str();
        ss.flush();

        if (desc_node != NULL)
        {


            mxml_node_t * desc_child;

            desc_child = mxmlFindElement(desc_node, tree, "coordinates", NULL, NULL, MXML_DESCEND);
            coord = desc_child->child->value.text.string;
            //cout << coord << "\n";

            the_geom = coord;
            cout << "Point:" << the_geom << "\n";
            desc_node = mxmlFindElement(node, tree, "name", NULL, NULL, MXML_DESCEND);
            if (desc_node != NULL)
            {
                pname = desc_node->child->value.text.string;
                //cout << pname << "\n";
            }

            list = "gid,basket,chk,locus,dccode,square,date,area,publication,heightinstrument,poleheight,bone,flint,pottery,rc,soilsample,crate,location,the_geom";

            description = outputDescriptionBase(inc, the_geom, list);
            /*
            desc_node = mxmlFindElement(node, tree, "description", NULL, NULL, MXML_DESCEND);
           if (desc_node != NULL)
           {
                //pname = desc_node->child->value.text.string;
                pname = desc_node->value.text.string;
                cout << pname << "\n";
           }
           */
        }
        desc_node = mxmlFindElement(node, tree, "Polygon", NULL, NULL, MXML_DESCEND);

        if (desc_node != NULL)
        {
        //cout << "found\n";

        mxml_node_t * desc_child;

        desc_child = mxmlFindElement(desc_node, tree, "coordinates", NULL, NULL, MXML_DESCEND);
        mxml_node_t * desc_child_child;
        vector< vector<string> > data;
        //vector<string>piece;
        string the_geom = "";
        char delim = ',';
        int m=0;
        for(desc_child_child = desc_child->child; desc_child_child != NULL; desc_child_child = desc_child_child->next)
        {


            string coord = desc_child_child->value.text.string;
            if (coord != "" && coord != "\n" && coord != "\r" && coord != " ")
            {
                the_geom.append(coord);
                the_geom.append(" ");
                //char delim = ',';
                //cout << coord << "\n";
                //piece.push_back(coord);

                data.push_back(splitSTR(coord,delim));
           // coord = desc_child->child->value.text.string;
                //cout << piece[0] << " " << piece[1] << "\n";
                //cout << piece[m] << "\n";
                //cout << m << " " << coord << "\n";
            }
            m++;
        }

        //cout << the_geom << "\n";
        //cout << "MAde it here\n";
        desc_node = mxmlFindElement(node, tree, "name", NULL, NULL, MXML_DESCEND);



       if (desc_node != NULL)
       {
            //pname = desc_node->child->value.text.string;
            //cout << pname << "\n";
       }
       else
       {
           //Make Name tag
           pname = inc;
           mxml_node_t *name;
           name = mxmlNewElement(node, "name");
                          mxmlNewText(name, 0,pname.c_str());
           //BANG

       }

        list = "gid,basket,chk,dccode,locusdesc,locus,square,date,area,structure,publication,heightinstrument,poleheight,the_geom";
        description = outputDescriptionBase(inc, the_geom, list);

        desc_node = mxmlFindElement(node, tree, "description", NULL, NULL, MXML_DESCEND);
        if (desc_node != NULL)
        {
            //cout << "Exists!\n";
            mxmlRemove(desc_node);
            desc_node = mxmlNewElement(node, "description");
            mxmlNewText(desc_node, 0,description.c_str());

        }
        else
        {
            desc_node = mxmlNewElement(node, "description");
            mxmlNewText(desc_node, 0,description.c_str());

        }


        //cout << description << "\n";

        int count = data.size();
        //cout << count << "\n";

        string top = "1452.000";
        string ph_top = "";
        int i;
        double tempD;
        //string tempB;
        string bottom;
        double lowelev = 10000;
        for (i=0; i<count; i++)
	{
        ph_top.append(data[i][0]);
        ph_top.append(" ");
        ph_top.append(data[i][1]);
        ph_top.append(" ");

        if (data[i].size() == 3)
        {
            ph_top.append(data[i][2]);
            tempD = atof(data[i][2].c_str());
            //tempB = data[i][2];
        }
        else
        {
            //Top Always PS
            ph_top.append(top);
            tempD = atof(top.c_str());
            //tempB = top;
        }



        if (lowelev > tempD)
        {
            lowelev = tempD;
            //bottom = tempB;
        }
        if (i != (count -1))
        {
        ph_top.append("\n");
        }

	}
        string ph_bottom = "";
        double newelev = lowelev - 0.1;
        std::stringstream ss;
        ss.precision(19);
        ss << newelev;
        bottom = ss.str();
        ss.flush();


	   for (i=0; i<count; i++)
	{
        ph_bottom.append(data[i][0]);
        ph_bottom.append(" ");
        ph_bottom.append(data[i][1]);
        ph_bottom.append(" ");

            //Top Always PS
        ph_bottom.append(bottom);



        if (i != (count -1))
        {
        ph_bottom.append("\n");
        }

	}

	//cout << ph_bottom << "\n";

	mxml_node_t *polyhedron;
	mxml_node_t *coordTop;
	mxml_node_t *coordBottom;
	mxml_node_t *color;

	polyhedron = mxmlNewElement(node, "Polyhedron");
                        coordTop = mxmlNewElement(polyhedron, "coordTop");
                          mxmlNewText(coordTop, 0, ph_top.c_str());
                        coordBottom = mxmlNewElement(polyhedron, "coordBottom");
                          mxmlNewText(coordBottom, 0, ph_bottom.c_str());
                        color = mxmlNewElement(polyhedron, "color");
                          mxmlNewText(color, 0,"0 255 0");
    /*
        desc_node = mxmlFindElement(node, tree, "description", NULL, NULL, MXML_DESCEND);
       if (desc_node != NULL)
       {
            //pname = desc_node->child->value.text.string;
            pname = desc_node->value.text.string;
            cout << pname << "\n";
       }
       */
        }
        r++;
        }
    /*
    mxml_node_t * table = mxmlFindElement(tree, tree, node_name, NULL, NULL, MXML_DESCEND);
    for(mxml_node_t * child = table-> child; child != NULL; child = child->next)
    {
       tableName = child->value.text.string;
       array.push_back(tableName);
       //cout << "test: " << tableName << "\n";
    }
*/
//Add file to kmlfiles.xml

//Rename Placemark

char *ptr = mxmlSaveAllocString(tree, MXML_NO_CALLBACK);

    FILE *fp;
    string filestr = "queries/";
    filestr.append(kmlfile);
    filestr.append(".kml");
    const char* file = filestr.c_str();
    fp = fopen(file, "w");

    fprintf(fp, ptr);

    fclose(fp);
    cout << "File Written! Completed!\n";

}
string getLocusName(string locus, string table)
{
string      query;
string      list;
string      type;
string      srid;
string      name;
vector< vector<string> > rowdata;

table = "kis10sf";

query = "SELECT basket,dccode,locus,date FROM ";
query.append(table);
query.append(" WHERE locus = '");
query.append(locus);
query.append("' AND (dccode ='PS-PERIMETER_START' OR dccode ='PB-PERIMETER_BOTTOM')");
list = "basket,dccode,locus,date";
type = "Point";
srid = "";
rowdata = queryPostGIS(query, list, type, srid);

int rows = rowdata.size();
//cout << rows << "\n";
int m;
//double tempD;
//double lowelev = 10000;
string status = "";
string date = "";
if (rows != 0)
{
    for(m=0; m<rows; m++)
    {

       //Placemark
        string temp;

        //D_CODE
        temp = rowdata[m][1];

        if (temp == "PB-PERIMETER_BOTTOM")
        {
            status = "closed";
            date = rowdata[m][3];
        }

        if (status != "closed")
        {
            date = rowdata[m][3];
        }
    }

    if (status == "closed")
    {
        name = "L.";
        name.append(locus);
        name.append(" (Closed ");
        name.append(date);
        name.append(")");
    }
    else
    {
        name = "L.";
        name.append(locus);
        name.append(" (Opened ");
        name.append(date);
        name.append(")");
    }
}
else
{
    name = "L.";
    name.append(locus);
}
return name;
}
string getLocusColor(string locus,string table)
{
string      query;
//string      list;
//string      type;
//string      srid;
PGresult        *res;
vector< vector<string> > rowdata;



query = "SELECT locus,locusdesc FROM ";
query.append(table);
query.append(" WHERE locus = '");
query.append(locus);
query.append("'");
//cout << query << "\n";

res = queryPG(query,"");

string locusdesc = PQgetvalue(res, 0, 1);

//cout << locusdesc << "\n";

//Find RGB for locusdesc
query = "SELECT rgb FROM locusdesc";
//query.append(table);
query.append(" WHERE locusdesc = '");
query.append(locusdesc);
query.append("'");
//cout << query << "\n";

res = queryPG(query,"");
int rows = PQntuples(res);
string rgb;
if (rows != 0)
{
    rgb = PQgetvalue(res, 0, 0);
}
else
{
    rgb = "";
}

if (rgb == "")
{
    rgb = "0 127 255";
}

//cout << rgb << "\n";
return rgb;
}
double findBottom(string locus)
{
   double lowest;
string      query;
string      list;
string      type;
string      srid;
vector< vector<string> > rowdata;

string table = "kis10sf";

query = "SELECT basket,dccode,locus FROM ";
query.append(table);
query.append(" WHERE locus = '");
query.append(locus);
query.append("' AND (dccode ='GR-GROUND_SHOT' OR dccode ='PB-PERIMETER_BOTTOM')");
list = "basket,dccode,locus,the_geom";
type = "Point";
srid = "4326";
rowdata = queryPostGIS(query, list, type, srid);

int rows = rowdata.size();
//cout << rows << "\n";
int m;
double tempD;
double lowelev = 10000;
for(m=0; m<rows; m++)
{

   //Placemark
    string temp;

    //Name
    temp = rowdata[m][3];
    vector<string>array;
    char delim = ' ';
    array = splitSTR(temp,delim);
    tempD = atof(array[2].c_str());
        if (lowelev > tempD)
        {
            lowelev = tempD;
        }

    //cout << lowelev << "\n";
}
    if (lowelev != 10000)
    {
        lowest = lowelev;
    }
    else
    {
        lowest = 0;
    }
   return lowest;

}
string findTop(string locus)
{

string      query;
string      list;
string      type;
string      srid;
vector< vector<string> > rowdata;

string table = "kis10sf";

query = "SELECT basket,dccode,locus FROM ";
query.append(table);
query.append(" WHERE locus = '");
query.append(locus);
query.append("' AND (dccode ='PS-PERIMETER_START')");
list = "basket,dccode,locus,the_geom";
type = "Point";
srid = "4326";
rowdata = queryPostGIS(query, list, type, srid);

int rows = rowdata.size();
//cout << rows << "\n";
//int m;

string tempD;
if (rows > 0)
{


//double lowelev = 10000;

   //Placemark
    string temp;

    temp = rowdata[0][3];
    vector<string>array;
    char delim = ' ';
    array = splitSTR(temp,delim);
    tempD = array[2];



}
else
{
    //cout << "No PS found, will try GR\n";

    query = "SELECT basket,dccode,locus FROM ";
    query.append(table);
    query.append(" WHERE locus = '");
    query.append(locus);
    query.append("' AND (dccode ='GR-GROUND_SHOT')");
    list = "basket,dccode,locus,the_geom";
    type = "Point";
    srid = "4326";
    rowdata = queryPostGIS(query, list, type, srid);

    rows = rowdata.size();

    if (rows != 0)
    {
        double tempG = 0;
        double lowelev = 0;
        string tempString;
        int m;
        string temp;
        for(m=0; m<rows; m++)
        {

           //Placemark
            //string temp;

            //Name
            temp = rowdata[m][3];
            //vector<string>array;
            char delim = ' ';
            vector<string>array;
            array = splitSTR(temp,delim);
            tempString = array[2];
            //cout << tempString << "\n";
            tempG = atof(array[2].c_str());
                if (lowelev < tempG)
                {
                    lowelev = tempG;
                    tempD = tempString;
                }

            //cout << lowelev << "\n";
        }

    }
    else
    {
        //cout << "No GR found either\n";
        tempD = "0";
    }
    //tempD = "0";
}
   //cout << tempD << "\n";
   return tempD;

}
vector<string> makePolyhedron(string temp, string locus)
{
    vector<string>ph_array;
    //cout << temp << "\n";
    double bottom = findBottom(locus);
    string top = findTop(locus);
    //cout << locus << ": Top Elev = " << top << "\n";

    vector<string>array;
    char delim = ' ';
    array = splitSTR(temp,delim);

    int count = array.size();
    vector< vector<string> > data;
    int i;
    delim = ',';

    for (i=0; i<count; i++)
	{
	data.push_back(splitSTR(array[i],delim));
	}
    //cout << data[0][1] << "\n";
    //string top = "0";
    /*
    if (data[0].size() < 3)
    {
        //cout << "Missing Top Elevations!\n";
        top = findTop(locus);
        //cout << "New Top Elev will be" << top << "\n";
    }
    */
	string ph_top = "";
    double lowelev = 10000;
    double tempD;
    for (i=0; i<count; i++)
	{
        ph_top.append(data[i][0]);
        ph_top.append(" ");
        ph_top.append(data[i][1]);
        ph_top.append(" ");

        if (top == "0")
        {
            ph_top.append(data[i][2]);
            tempD = atof(data[i][2].c_str());
        }
        else
        {
            //Top Always PS
            ph_top.append(top);
            tempD = atof(top.c_str());
        }



        if (lowelev > tempD)
        {
            lowelev = tempD;
        }
        if (i != (count -1))
        {
        ph_top.append("\n");
        }




	}
	//cout.precision(19);
	//cout << lowelev << "\n";
    ph_array.push_back(ph_top);

    if (bottom > lowelev)
    {
       // cout << "Lowest Top Polygon Elevation (" << lowelev << ") is lower than\n lowest closing elevation (" << bottom << ")!\n";

        //cout << "Will default closing elevation to 10cm.\n";
        bottom = 0;
    }

    string ph_bottom = "";
    double depth = 0;
    //double velev = 0;
    double selev;
    if (bottom == 0)
    {
        selev = 0.1;
    }
    else
    {
        selev = bottom;
    }


    string newelev;
    //double bottom;
    for (i=0; i<count; i++)
	{
        ph_bottom.append(data[i][0]);
        ph_bottom.append(" ");
        ph_bottom.append(data[i][1]);
        ph_bottom.append(" ");

        //velev = atof(data[i][2].c_str());
        if (selev == 0.1)
        {
            bottom = lowelev - selev;
        }
        else
        {
            //depth = lowelev - selev;
        //bottom = lowelev - depth;
        }

        //cout.precision(19);
        //cout << lowelev << " " << bottom << "\n";
        std::stringstream ss;
        ss.precision(19);
        ss << bottom;
        newelev = ss.str();
        ss.flush();
        //ss.close();
        //final = ss.str();
        ph_bottom.append(newelev);

        if (i != (count -1))
        {
        ph_bottom.append("\n");
        }
        depth = 0;

	}

    ph_array.push_back(ph_bottom);

    return ph_array;
}
void syncTable(string table)
{
    PGresult        *res;
    string query;
    //string gidLocal1;
    //string gidLocal2;
    //string gidCentral1;
    //string gidCentral2;
    //  int rows;
    query = "SELECT true FROM pg_tables WHERE tablename = '";
    query.append(table);
    query.append("';");

    res = queryPG(query,"Local");
    int tableCheckLocal = PQntuples(res);

    res = queryPG(query,"Central");
    int tableCheckCentral = PQntuples(res);

    if (tableCheckLocal == 0 && tableCheckCentral == 0)
    {
        cout << "Table does not exist on Local or Central Servers!\n";
    }
    else if (tableCheckLocal == 0 && tableCheckCentral == 1)
    {
        cout << "Table does not exist on Local but exists on Central Server!\n";
        cout << "Will add table to Local Server!\n";
    }
    else if (tableCheckLocal == 1 && tableCheckCentral == 0)
    {
        cout << "Table does not exists on Local but not on Central Server!\n";
        cout << "Will add table to Central Server!\n";
    }
    else if (tableCheckLocal == 1 && tableCheckCentral == 1)
    {
        cout << "Table exists on Local and Central Servers!\n";

        query = "SELECT gid FROM ";
        query.append(table);
        query.append(" ORDER BY gid DESC");
        cout << query << "\n";
        res = queryPG(query,"Local");
        int rowsLocal = PQntuples(res);
        cout << "Rows: " << rowsLocal << "\n";
        int gidLocal1 = atoi(PQgetvalue(res, 0, 0));
        int gidLocal2 = atoi(PQgetvalue(res, 1, 0));
        cout << gidLocal1 << "\n";
        cout << gidLocal2 << "\n";

        res = queryPG(query,"Central");
        int rowsCentral = PQntuples(res);
        cout << rowsCentral << "\n";
        cout << "Rows: " << rowsCentral << "\n";
        int gidCentral1 = atoi(PQgetvalue(res, 0, 0));
        int gidCentral2 = atoi(PQgetvalue(res, 1, 0));
        cout << gidCentral1 << "\n";
        cout << gidCentral2 << "\n";

        //rowsLocal++;
        //gidLocal1++;

        if (gidLocal1 == gidCentral1)
        {
            cout << "Last entry for Local table is the same as Central table\n";
            cout << "Will assume Central has most up to date data and will update Local table\n";
        }
        else if (gidLocal1 > gidCentral1 && rowsLocal > rowsCentral)
        {
            cout << "Local table has new entries and will append to Central table\n";
        }
        else
        {
            cout << "Tables have been inapropriately modified!\n";
        }
    }
    else
    {
        cout << "Tables are not setup appropriately!\n";
    }



}
void getDistinctLists(string table)
{
    PGresult        *res;
    PGresult        *res2;
    string          query;


    query = "Select column_name FROM information_schema.columns WHERE table_name='";
    query.append(table);
    query.append("'");
    res = queryPG(query,"");

    int rows = PQntuples(res);

    //cout << "Rows: " << rows << "\n";

if (rows > 0)
{


    mxml_node_t *xml;    /* <?xml ... ?> */
    mxml_node_t *field;   /* <field> */

    int count;
    int i;
    int m;
    string value;
    string column;
    string list;
    i = 0;
    xml = mxmlNewXML("1.0");
    const char* columnChar;
    const char* listChar;
for(m=0; m<rows; m++)
{

    //cout << PQgetvalue(res, m, 0) << "\n";
    column = PQgetvalue(res, m, 0);
    list = "";


    if (column != "the_geom" && column != "chk")
    {
            columnChar = column.c_str();
            field = mxmlNewElement(xml, columnChar);
            query = "SELECT DISTINCT ";
            query.append(column);
            query.append(" FROM ");
            query.append(table);
            res2 = queryPG(query,"");
            count = PQntuples(res2);
            //cout << query << "\n";
    }
    else
    {
        count = 0;
    }
    for (i=0; i<count; i++)
	{
	    value = PQgetvalue(res2, i, 0);
        if (value == "" || value == "NULL" || value == " " || value == "\r" || value == "\n")
        {

        }
        else
        {
        //cout << "-" << value << "\n";
        list.append(value);
        list.append("\n");
        }


	}
	listChar = list.c_str();
	mxmlNewText(field, 0, listChar);

}
    FILE *fp;
    //const char* kml_name = "Query Test";
    fp = fopen("menu.xml", "w");
    mxmlSaveFile(xml, fp, MXML_NO_CALLBACK);
    fclose(fp);

    cout << "File Written! Completed!\n";
}
else
{
    cout << "No Data to Export!\n";
}

}
string tableType(string table)
{
    string type;
    int end = table.size();
    char check = table[end-1];
    //cout << check << "\n";

     switch (check) {

      case 'f':   type = "Point";
            break;

      case 'l':   type = "KML-P";
            break;

      default:    type = "Error";
                  cout << "Table must end with either 'sf' for points or 'l' for polygons!\n";

     }
    //type = check;
    return type;
}
void printQueryResults(string table, string where)
{
    string      list;
    string      type;
    vector< vector<string> > rowdata;

    rowdata = getArrayofQueryFunction(table, where);

    if (rowdata.size() > 0)
    {
        type = tableType(table);
        if (type == "Point")
        {
            list = "gid,basket,chk,locus,dccode,square,date,area,publication,heightinstrument,poleheight,bone,flint,pottery,rc,soilsample,crate,location,the_geom";
            outputPostGISarray(rowdata, list);
        }
        else if (type == "KML-P")
        {
            list = "gid,basket,chk,dccode,locusdesc,locus,square,date,area,structure,publication,heightinstrument,poleheight,the_geom";
            outputPostGISarray(rowdata, list);
        }
    }
    else
    {
        cout << "Query failed to find any results!\n";
    }

}
vector< vector<string> > getArrayofQueryFunction(string table, string where)  //Function -a
{
string      query;
string      list;
string      type;
string      srid;
vector< vector<string> > rowdata;
type = tableType(table);


    if (type == "Point")
    {
        query = "SELECT gid,basket,chk,locus,dccode,square,date,area,publication,heightinstrument,poleheight,bone,flint,pottery,rc,soilsample,crate,location FROM ";
        query.append(table);
        query.append(" WHERE ");
        query.append(where);
        list = "gid,basket,chk,locus,dccode,square,date,area,publication,heightinstrument,poleheight,bone,flint,pottery,rc,soilsample,crate,location,the_geom";
        srid = "4326";
        //cout << query;
        rowdata = queryPostGIS(query, list, type, srid);




    }
    else if (type == "KML-P")
    {

        query = "SELECT gid,basket,chk,dccode,locusdesc,locus,square,date,area,structure,publication,heightinstrument,poleheight FROM ";
        query.append(table);
        query.append(" WHERE ");
        query.append(where);
        list = "gid,basket,chk,dccode,locusdesc,locus,square,date,area,structure,publication,heightinstrument,poleheight,the_geom";
        srid = "4326";
        rowdata = queryPostGIS(query, list, type, srid);




    }
    return rowdata;
}
void kmlFromQuery(string table, string where, string name)
{
    string      type;
    string      list;
    vector< vector<string> > rowdata;
    rowdata = getArrayofQueryFunction(table, where);

    //type = "KML-P";
    //type = "Point";
    if (rowdata.size() > 0)
    {
        type = tableType(table);

        if (type == "Point")
        {
            list = "gid,basket,chk,locus,dccode,square,date,area,publication,heightinstrument,poleheight,bone,flint,pottery,rc,soilsample,crate,location,the_geom";
            //outputPostGISarray(rowdata, list);
            makePointKML(rowdata, list, where, name);

        }
        else if (type == "KML-P")
        {
            list = "gid,basket,chk,dccode,locusdesc,locus,square,date,area,structure,publication,heightinstrument,poleheight,the_geom";
            //outputPostGISarray(rowdata, list);
            makePolygonKML(rowdata, list, where, name);
            //cout << "Polygon Export not implemented yet\n";
        }
    }
    else
    {
        cout << "Query failed to find any results!\n";
    }
}
string getTimeStamp()
{
    string timestamp;
    time_t rawtime;
  struct tm * timeinfo;
  char buffer [80];

  time ( &rawtime );
  timeinfo = localtime ( &rawtime );

  strftime (buffer,80,"%Y%j%H%M%S",timeinfo);
    //cout << buffer << "\n";

    timestamp = string(buffer);


    return timestamp;
}
vector< vector<string> > parseWhere(string where)
{
    vector<string>array;
    int count;
    char delim;
    //char file[256];
    const char* file;
    file = "config.ini";

    //array = fileIntoArray(file);
    delim = ' ';
    array = splitSTR(where,delim);
    count = array.size();
    //cout << count << "\n";
    delim = '=';
    //vector<string>data;
    vector< vector<string> > data;
    //(3, vector<string>(2,0));
    int i;
    //count = 0;
    string test;
    int count2;
    vector<string>check;
    int pos;
    for (i=0; i<count; i++)
	{
	    //cout << array[i] << "\n";
	check = splitSTR(array[i],delim);


	//test = data[i][1];
	count2 = check.size();
	int length;
	if (count2 > 1)
	{
	    pos = check[0].find("(");
	    //cout << pos << "\n";
        if( pos >= 0)
        {
            check[0].erase(pos,1);

        }
        //pos = -1;
        pos = check[1].find(")");
        //cout << pos << "\n";
        if(pos >= 0)
        {
            check[1].erase(pos);

        }
        check[1].erase(0,1);
        length = check[1].size();
        check[1].erase(length-1);
	    data.push_back(check);
	    //cout << check[0] << " " << check[1] << "\n";
	    //pos = -1;
	}

	}



    return data;
}
bool preventOverwrite(string name)
{
    string fullname = "queries/";
    fullname.append(name);
    fullname.append(".kml");

    const char* t1 = fullname.c_str();
    const char* t2 = fullname.c_str();
    if(rename(t1, t2) == -1)
    {
        //cout << "File does not exist or could not be deleted!\n";

        return false;
    }
    else
    {
        //cout << "File exists!\n";
        return true;
    }

}
string getNxtQuery(string type)
{
    string nextquery;

    //string filename;
        vector<string>kmlfiles;
        string list = "";
        string qname;
        int inc = 0;
        int old = 0;
        string final;
        kmlfiles = getNodeArray("kmlfiles.xml", "kmlfiles");
        int count = kmlfiles.size();
        int i;
        for(i=0; i<count; i++)
        {
            if (type == "Point")
            {


                qname = kmlfiles[i];
                if (qname.substr(0,5) == "query")
                {
                    list = qname.substr(5,2);
                    inc = atoi(list.c_str());
                    if (inc > old)
                    {
                        old = inc;
                    }

                //cout << qname << " " << inc << " " << old << "\n";
                }
            }
            else if (type == "Polygon")
            {


                qname = kmlfiles[i];
                if (qname.substr(0,5) == "querp")
                {
                    list = qname.substr(5,2);
                    inc = atoi(list.c_str());
                    if (inc > old)
                    {
                        old = inc;
                    }

                //cout << qname << " " << inc << " " << old << "\n";
                }
            }

        }
        old++;

        if (type == "Point")
        {
        nextquery = "query";
        }
        else if (type == "Polygon")
        {
        nextquery = "querp";
        }


  std::stringstream ss;
  ss << old;
    final = ss.str();
    if (old < 10)
    {
       nextquery.append("0");
    }

    nextquery.append(final);


    //cout << nextquery << "\n";

    return nextquery;
}
string generateQName(string where, string type)
{
    string name;
    string nextquery;
   // vector< vector<string> > data;
    //data = parseWhere(where);
    nextquery = getNxtQuery(type);

    name = nextquery;
    return name;
}
mxml_node_t * getTree(string filename)
{
    FILE * fp;
  mxml_node_t * tree;
  fp = fopen(filename.c_str(),"r");
  if(fp == NULL){
    std::cerr << "Unable to open file: " << filename << std::endl;
    //return;
  }
  fclose(fp);


  //tree = mxmlLoadFile(NULL, fp, MXML_TEXT_CALLBACK);


ifstream in(filename.c_str());
stringstream buffer;
buffer << in.rdbuf();
string contents(buffer.str());

  tree = mxmlLoadString(NULL, contents.c_str(), MXML_NO_CALLBACK);
  //char *ptr = mxmlSaveAllocString(xml, MXML_TEXT_CALLBACK);
  //cout << "LoadingFile\n";
  //fclose(fp);
  if(tree == NULL){
    std::cerr << "Unable to parse XML file: " << filename << std::endl;
    //return;
  }
    return tree;
}
vector<string> getNodeArray(string filename, const char* node_name)
{
    vector<string>array;

    mxml_node_t * tree;
    string tableName;
    tree = getTree(filename);
    mxml_node_t * table = mxmlFindElement(tree, tree, node_name, NULL, NULL, MXML_DESCEND);
    for(mxml_node_t * child = table-> child; child != NULL; child = child->next)
    {
       tableName = child->value.text.string;
       array.push_back(tableName);
       //cout << "test: " << tableName << "\n";
    }
    return array;
}
void appendStringXML(string filename, const char* node_name, string str)
{
    mxml_node_t * tree;
    tree = getTree(filename);

    mxml_node_t *node;
    mxml_node_t * desc_child;


    node = mxmlFindElement(tree, tree, node_name,
                           NULL, NULL,
                           MXML_DESCEND);
    desc_child = node->child;
    const char* value = str.c_str();
    mxmlNewText(node, 0, value);
    //mxmlSetText(desc_child, 0, name.c_str());
//Save File

  const char *ptr;
    ptr = "";
  ptr = mxmlSaveAllocString(tree, MXML_NO_CALLBACK);
    //cout << ptr;
    FILE *fp;

    const char* kml_name = filename.c_str();
    fp = fopen(kml_name, "w");

    fprintf(fp, ptr);

    fclose(fp);
    /*
    FILE * fp;

    fp = fopen(kml_name, "w");
    mxmlSaveFile(tree, fp, MXML_NO_CALLBACK);
    fclose(fp);
    */

}
void resetStringXML(string filename, const char* node_name, string str)
{
   // mxml_node_t * tree;
    //tree = getTree(filename);

    mxml_node_t *node;
    mxml_node_t *xml;

    const char* value = str.c_str();
    //mxml_node_t * desc_child;

    xml = mxmlNewXML("1.0");
    node = mxmlNewElement(xml, node_name);
        mxmlNewText(node, 0, value);
/*
    node = mxmlFindElement(tree, tree, node_name,
                           NULL, NULL,
                           MXML_DESCEND);
    desc_child = node->child;
    const char* value = str.c_str();
    //mxmlNewText(node, 0, value);
    mxmlDelete(node);
    name = mxmlNewElement(xml, node_name);
    mxmlNewText(node, 0, value);
    //mxmlSetText(desc_child, 0, value);
*/
//Save File

  char *ptr = mxmlSaveAllocString(xml, MXML_NO_CALLBACK);

    FILE *fp;

    const char* kml_name = filename.c_str();
    fp = fopen(kml_name, "w");

    fprintf(fp, ptr);

    fclose(fp);
/*
    FILE * fp;
    const char* kml_name = filename.c_str();
    fp = fopen(kml_name, "w");
    mxmlSaveFile(xml, fp, MXML_NO_CALLBACK);
    fclose(fp);
*/
}
void renameQfile(string file, string name)
{
  string    filename = "queries/";
  filename.append(file);
  filename.append(".kml");
  mxml_node_t * tree;
  tree = getTree(filename);
  //cerr << "Reading query: " << filename << endl;



    mxml_node_t *node;
    mxml_node_t * desc_child;


    node = mxmlFindElement(tree, tree, "name",
                           NULL, NULL,
                           MXML_DESCEND);

                           //const char *type;
//type = node->value.element.value;
string q_name;
desc_child = node->child;

if (name == "")
{
   q_name = desc_child->value.text.string;
}
else
{
    //desc_child->value.text.string = name.c_str();
    q_name = name;

    mxmlSetText(desc_child, 0, name.c_str());

}

//desc_text = node->value.text.string;
//double desc_dbl = atof(desc_text);
if (preventOverwrite(q_name))
{
    cout << "File Exists will try to increment!\n";
    if (q_name.substr(0,5) == "query")
    {
        q_name = generateQName("","Point");
    }
    else if (q_name.substr(0,5) == "querp")
    {
        q_name = generateQName("","Polygon");
    }
}


//cout << "Text:" << q_name << "\n";
string wname = q_name;
q_name.append(".kml");
string fullname = "queries/";
fullname.append(q_name);
//cout << "saving file\n";
//Save File
  //char *buffer;
  //mxmlSaveString(tree, buffer, sizeof(buffer),MXML_NO_CALLBACK);
cout << "saving file\n";
  char *ptr = mxmlSaveAllocString(tree, MXML_NO_CALLBACK);

    FILE *fp;

    const char* kml_name = fullname.c_str();
    fp = fopen(kml_name, "w");

    //fprintf(fp, ptr);
    fputs(ptr,fp);
    fclose(fp);
    cout << "saved file\n";
    /*
const char* kml_name = fullname.c_str();
    //FILE *fp;
    //mxml_node_t *tree;
    //mxmlSetWrapMargin(0);
    FILE * fp;
    fp = fopen(kml_name, "w");
    mxmlSaveFile(tree, fp, MXML_NO_CALLBACK);
    fclose(fp);
*/
    //Write new file to kmlfiles
    //string q_name;
    vector<string>kmlfiles;
    filename = "kmlfiles.xml";

    string str = "\n";
    str.append(wname);
    appendStringXML(filename, "kmlfiles", str);
    //kmlfiles = getNodeArray(filename, "kmlfiles");
    //cout << kmlfiles[0] <<"\n";

}
void deleteQfile(string name)
{
    string filename = "queries/";
    filename.append(name);
    filename.append(".kml");
    const char* f = filename.c_str();

    if( remove(f) == -1 )
    {
        cout << "File does not exist or could not be deleted!\n";
    }
    else
    {
        cout << "File deleted\n";
                vector<string>kmlfiles;
        string list = "";
        kmlfiles = getNodeArray("kmlfiles.xml", "kmlfiles");
        int count = kmlfiles.size();
        int i;
        for(i=0; i<count; i++)
        {
        if (kmlfiles[i] != name)
        {
        list.append(kmlfiles[i]);
        list.append("\n");
        }

        }
        int pos = list.rfind("\n");
        list.erase(pos);
        cout << list << "\n";
        resetStringXML("kmlfiles.xml", "kmlfiles", list);
        cout << "Xml entry deleted\n";
    }



}
void saveMXML(mxml_node_t *xml, string filename)
{
  //char *buffer;
  //mxmlSaveString(xml, buffer, sizeof(buffer),MXML_NO_CALLBACK);

  char *ptr = mxmlSaveAllocString(xml, MXML_NO_CALLBACK);

        FILE *fp;
    //mxml_node_t *tree;
    //mxmlSetWrapMargin(0);
    fp = fopen("queries/query.kml", "w");
    cout << "This far\n";
    fprintf(fp, ptr);
    //mxmlSaveFile(xml, fp, MXML_NO_CALLBACK);
    fclose(fp);


    cout << "File Written! Completed!\n";
}
void makePointKML(vector< vector<string> > rowdata, string list, string where, string q_name)
{
    mxml_node_t *xml;    /* <?xml ... ?> */
    mxml_node_t *kml;   /* <kml> */
    mxml_node_t *document;   /* <Document> */
    mxml_node_t *name;   /* <name> */
    mxml_node_t *query;   /* <query> */
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
    mxml_node_t *styleurl;   /* <data> */
    mxml_node_t *point;   /* <data> */
    mxml_node_t *altitudeMode;   /* <data> */
    mxml_node_t *coordinates;   /* <data> */



//Create KML Container

//KML Name
    if (q_name == "")
    {
        //q_name = generateQName(where,"Point");
        q_name = "query";
    }

    string g_timestamp = getTimeStamp();

   const char* kml_name = q_name.c_str();
   const char* kml_query = where.c_str();
   const char* kml_timestamp = g_timestamp.c_str();

xml = mxmlNewXML("1.0");
        kml = mxmlNewElement(xml, "kml");
            document = mxmlNewElement(kml, "Document");
                name = mxmlNewElement(document, "name");
                  mxmlNewText(name, 0, kml_name);
                query = mxmlNewElement(document, "query");
                  mxmlNewText(query, 0, kml_query);
                timestamp = mxmlNewElement(document, "timestamp");
                  mxmlNewText(timestamp, 0, kml_timestamp);
//.......................................................
int rows = rowdata.size();
int m;

for(m=0; m<rows; m++)
{

   //Placemark
    string temp;

    //Name
    temp = rowdata[m][1];
    //cout << temp << "\n";
    //temp.append(" ");
    //temp.append(rowdata[m][4]);
    //cout << temp << "\n";
    const char* q_name = temp.c_str ();
    //cout << q_name << "\n";
    //q_name = temp.c_str ();

    //LookAt default variables for KIS
    const char* q_longitude = "35.4991377749059";
    const char* q_latitude = "30.4672489448774";
    const char* q_altitude = "0";
    const char* q_range = "10";
    const char* q_tilt = "0";
    const char* q_heading = "0";
    /*
    double longit = 35.4991377749059;
    double latit = 30.4672489448774;
    double altit = 1459.5757575755555;
    */
    //cout << setprecision(17) << longit << "\n";
    //Coordinates

    char delim = ',';
    vector<string>field;
    field = splitSTR(list,delim);
    int end = field.size() - 1;
    //cout << end << "\n";
    temp = rowdata[m][end];
    const char* q_coordinates = temp.c_str ();

    //Description

    temp = outputDescription(rowdata, list, m);

    const char* q_description = temp.c_str ();

    //cout << q_description << "\n";
    //const char* q_description ="Basket: 50479\nChk: 2\nDCCODE: GR-GROUND SHOT\nLocus: 7054\nSquare: A1\nDate: 072210\nArea: D\nPublication: KIS10\nbone:\nflint:\npottery:\nrc:\nsoilsample:\nlocation:\ncrate:";


                placemark = mxmlNewElement(document, "Placemark");
                    name = mxmlNewElement(placemark, "name");
                      mxmlNewText(name, 0, q_name);
                    description = mxmlNewElement(placemark, "description");
                      mxmlNewText(description, 0, q_description);
                    lookat = mxmlNewElement(placemark, "LookAt");
                        longitude = mxmlNewElement(lookat, "longitude");
                          mxmlNewText(longitude, 0, q_longitude);
                          //mxmlNewReal(longitude,longit);
                        latitude = mxmlNewElement(lookat, "latitude");
                          mxmlNewText(latitude, 0, q_latitude);
                          //mxmlNewReal(latitude,latit);
                        altitude = mxmlNewElement(lookat, "altitude");
                          mxmlNewText(altitude, 0, q_altitude);
                          //mxmlNewReal(altitude,altit);
                        range = mxmlNewElement(lookat, "range");
                          mxmlNewText(range, 0, q_range);
                        tilt = mxmlNewElement(lookat, "tilt");
                          mxmlNewText(tilt, 0, q_tilt);
                        heading = mxmlNewElement(lookat, "heading");
                          mxmlNewText(heading, 0, q_heading);
                    styleurl = mxmlNewElement(placemark, "styleUrl");
                      mxmlNewText(styleurl, 0, "#msn_GR");
                    point = mxmlNewElement(placemark, "Point");
                        altitudeMode = mxmlNewElement(point, "altitudeMode");
                          mxmlNewText(altitudeMode, 0, "absolute");
                        coordinates = mxmlNewElement(point, "coordinates");
                          mxmlNewText(coordinates, 0, q_coordinates);

}
string filename = "queries/query.kml";
    //saveMXML(xml, filename);

  //char *buffer;
  //mxmlSaveString(xml, buffer, sizeof(buffer),MXML_NO_CALLBACK);

  char *ptr = mxmlSaveAllocString(xml, MXML_NO_CALLBACK);
  //int l = strlen(ptr);

        FILE *fp;
    //mxml_node_t *tree;
    //mxmlSetWrapMargin(0);
    fp = fopen("queries/query.kml", "w");

    fprintf(fp, ptr);
    //mxmlSaveFile(xml, fp, MXML_NO_CALLBACK);
    fclose(fp);


    cout << "File Written! Completed!\n";

//Test retreiving data for Connor



}
void makePolygonKML(vector< vector<string> > rowdata, string list, string where, string q_name)
{
    mxml_node_t *xml;    /* <?xml ... ?> */
    mxml_node_t *kml;   /* <kml> */
    mxml_node_t *document;   /* <Document> */
    mxml_node_t *name;   /* <name> */
    mxml_node_t *query;   /* <query> */
    mxml_node_t *timestamp;   /* <timestamp> */

    mxml_node_t *placemark;   /* <Placemark> */
    mxml_node_t *description;   /* <description> */

    //mxml_node_t *lookat;   /* <LookAt> */
    //mxml_node_t *longitude;   /* <data> */
    //mxml_node_t *latitude;   /* <data> */
    //mxml_node_t *altitude;   /* <data> */
    //mxml_node_t *range;   /* <data> */
    //mxml_node_t *tilt;   /* <data> */
    //mxml_node_t *heading;   /* <data> */
    mxml_node_t *styleurl;   /* <data> */
    //mxml_node_t *point;   /* <data> */
    mxml_node_t *altitudeMode;   /* <data> */
    mxml_node_t *coordinates;   /* <data> */
    mxml_node_t *polygon;
    mxml_node_t *extrude;
    mxml_node_t *tessellate;
    mxml_node_t *outerBoundaryIs;
    mxml_node_t *LinearRing;
    mxml_node_t *polyhedron;
    mxml_node_t *coordTop;
    mxml_node_t *coordBottom;
    mxml_node_t *color;



//Create KML Container

//KML Name
    q_name = "querp";
    if (q_name == "")
    {
        q_name = generateQName(where,"Polygon");
    }
    string g_timestamp = getTimeStamp();

   const char* kml_name = q_name.c_str();
   const char* kml_query = where.c_str();
   const char* kml_timestamp = g_timestamp.c_str();

xml = mxmlNewXML("1.0");
        kml = mxmlNewElement(xml, "kml");
            document = mxmlNewElement(kml, "Document");
                name = mxmlNewElement(document, "name");
                  mxmlNewText(name, 0, kml_name);
                query = mxmlNewElement(document, "query");
                  mxmlNewText(query, 0, kml_query);
                timestamp = mxmlNewElement(document, "timestamp");
                  mxmlNewText(timestamp, 0, kml_timestamp);
//.......................................................
int rows = rowdata.size();
int m;
string table = "kis10l";
for(m=0; m<rows; m++)
{

   //Placemark
    string temp;
    string locus = rowdata[m][5];
    //Name
    //temp = rowdata[m][1];
    string nametemp;
    nametemp = getLocusName(locus, table);
    //cout << temp << "\n";
    //temp.append(" ");
    //temp.append(rowdata[m][4]);
    //cout << temp << "\n";
    const char* q_name = nametemp.c_str ();
    //cout << q_name << "\n";
    //q_name = temp.c_str ();

    //LookAt default variables for KIS
    /*
    const char* q_longitude = "35.4991377749059";
    const char* q_latitude = "30.4672489448774";
    const char* q_altitude = "0";
    const char* q_range = "10";
    const char* q_tilt = "0";
    const char* q_heading = "0";
    */

    //Coordinates

    char delim = ',';
    vector<string>field;
    field = splitSTR(list,delim);
    int end = field.size() - 1;
    //cout << end << "\n";
    temp = rowdata[m][end];
    const char* q_coordinates = temp.c_str ();

    //Polyhedron
    vector<string>ph_array;
    temp = rowdata[m][end];
    //string bottom = "";

    //cout << locus << "\n";
    //double bottom = findBottom(locus);
    ph_array = makePolyhedron(temp,locus);
    const char* ph_top = ph_array[0].c_str();
    const char* ph_bottom = ph_array[1].c_str();
    //const char* ph_top = "Top";
    //const char* ph_bottom = "Bottom";


    //Description
    string test;
    test = outputDescription(rowdata, list, m);
    //cout << test<< "\n";
    const char* q_description = test.c_str ();

    //Get Locus Color

    temp = getLocusColor(locus,table);
    //temp = "Test";
    const char* lcolor = temp.c_str();

    //cout << q_description << "\n";
    //const char* q_description ="Basket: 50479\nChk: 2\nDCCODE: GR-GROUND SHOT\nLocus: 7054\nSquare: A1\nDate: 072210\nArea: D\nPublication: KIS10\nbone:\nflint:\npottery:\nrc:\nsoilsample:\nlocation:\ncrate:";


                placemark = mxmlNewElement(document, "Placemark");
                    name = mxmlNewElement(placemark, "name");
                      mxmlNewText(name, 0, q_name);
                    description = mxmlNewElement(placemark, "description");
                      mxmlNewText(description, 0, q_description);
                    /*
                    lookat = mxmlNewElement(placemark, "LookAt");
                        longitude = mxmlNewElement(lookat, "longitude");
                          mxmlNewText(longitude, 0, q_longitude);
                          //mxmlNewReal(longitude,longit);
                        latitude = mxmlNewElement(lookat, "latitude");
                          mxmlNewText(latitude, 0, q_latitude);
                          //mxmlNewReal(latitude,latit);
                        altitude = mxmlNewElement(lookat, "altitude");
                          mxmlNewText(altitude, 0, q_altitude);
                          //mxmlNewReal(altitude,altit);
                        range = mxmlNewElement(lookat, "range");
                          mxmlNewText(range, 0, q_range);
                        tilt = mxmlNewElement(lookat, "tilt");
                          mxmlNewText(tilt, 0, q_tilt);
                        heading = mxmlNewElement(lookat, "heading");
                          mxmlNewText(heading, 0, q_heading);
                    */
                    styleurl = mxmlNewElement(placemark, "styleUrl");
                      mxmlNewText(styleurl, 0, "#msn_GR");
                    polygon = mxmlNewElement(placemark, "Polygon");
                        extrude = mxmlNewElement(polygon, "extrude");
                          mxmlNewText(extrude, 0, "0");
                        tessellate = mxmlNewElement(polygon, "tessellate");
                          mxmlNewText(tessellate, 0, "1");
                        altitudeMode = mxmlNewElement(polygon, "altitudeMode");
                          mxmlNewText(altitudeMode, 0, "absolute");
                        outerBoundaryIs = mxmlNewElement(polygon, "outerBoundaryIs");
                            LinearRing = mxmlNewElement(outerBoundaryIs, "LinearRing");
                                coordinates = mxmlNewElement(LinearRing, "coordinates");
                                  mxmlNewText(coordinates, 0, q_coordinates);
                    polyhedron = mxmlNewElement(placemark, "Polyhedron");
                        coordTop = mxmlNewElement(polyhedron, "coordTop");
                          mxmlNewText(coordTop, 0, ph_top);
                        coordBottom = mxmlNewElement(polyhedron, "coordBottom");
                          mxmlNewText(coordBottom, 0, ph_bottom);
                        color = mxmlNewElement(polyhedron, "color");
                          mxmlNewText(color, 0,lcolor);


}
//Save File
  //char *buffer;
  //mxmlSaveString(xml, buffer, sizeof(buffer),MXML_NO_CALLBACK);

  char *ptr = mxmlSaveAllocString(xml, MXML_NO_CALLBACK);

    FILE *fp;

    //const char* kml_name = filename.c_str();
    fp = fopen("queries/querp.kml", "w");

    fprintf(fp, ptr);

    fclose(fp);
    cout << "File Written! Completed!\n";

/*
        FILE *fp;
    //mxml_node_t *tree;
    //mxmlSetWrapMargin(0);
    fp = fopen("queries/queryp.kml", "w");
    mxmlSaveFile(xml, fp, MXML_NO_CALLBACK);
    fclose(fp);

    cout << "File Written! Completed!\n";

//Test retreiving data for Connor

*/

}
void outputPostGISarray(vector< vector<string> > rowdata, string list)
{
    int rows = rowdata.size();
    int count;
    int m;
    int i;

    //Converts list sent by user to array for next for loop

    char delim = ',';
    vector<string>field;
    field = splitSTR(list,delim);
    cout << "Rows: " << rows << "\n";
    count = field.size();
    cout << "Fields: " << count << "\n";

for(m=0; m<rows; m++)
{

    cout << "Entry: " << m  << "\n";
    for(i=0; i<count; i++)
    {
        cout << field[i] << ": " << rowdata[m][i] << "\n";
    }
    cout << "\n";
}
    //string example = rowdata[0][1];
    //cout << "Array Count: " << rows << " Example: " << example << "\n";
}
string outputDescription(vector< vector<string> > rowdata, string list, int row)
{
    //int rows = rowdata.size();
    int count;
    //int m;
    int i;
    string description;

    //Converts list sent by user to array for next for loop

    char delim = ',';
    vector<string>field;
    field = splitSTR(list,delim);
    //cout << "Rows: " << rows << "\n";
    count = field.size();
    //cout << "Fields: " << count << "\n";
description = "";


    //cout << "Entry: " << m  << "\n";
    for(i=0; i<count; i++)
    {
        //cout << field[i] << ": " << rowdata[m][i] << "\n";
        description.append(field[i]);
        description.append(": ");
        description.append(rowdata[row][i]);
        description.append("\n");
    }
    //cout << "\n";

    return description;
}
string outputDescriptionBase(string gid, string the_geom, string list)
{
    //int rows = rowdata.size();
    int count;
    //int m;
    int i;
    string description;

    //Converts list sent by user to array for next for loop

    char delim = ',';
    vector<string>field;
    field = splitSTR(list,delim);
    //cout << "Rows: " << rows << "\n";
    count = field.size();
    //cout << "Fields: " << count << "\n";
description = "";


    //cout << "Entry: " << m  << "\n";
    for(i=0; i<count; i++)
    {
        //cout << field[i] << ": " << rowdata[m][i] << "\n";
        if (field[i] == "the_geom")
        {

        description.append(field[i]);
        description.append(": ");
        description.append(the_geom);
        description.append("\n");
        }
        else if (field[i] == "dccode")
        {

        description.append(field[i]);
        description.append(": ");
        description.append("GR-GROUND_SHOT");
        description.append("\n");
        }
        else if (field[i] == "locusdesc")
        {

        description.append(field[i]);
        description.append(": ");
        description.append("FI-FILL");
        description.append("\n");
        }
        else if (field[i] == "gid")
        {

        description.append(field[i]);
        description.append(": ");
        description.append(gid);
        description.append("\n");
        }
        else
        {


        description.append(field[i]);
        description.append(": ");
        description.append("0");
        description.append("\n");
        }
    }
    //cout << "\n";

    return description;
}
void commandLineParse(int argc, char *argv[])
{
        /*
        //Check command line working properly
        cout << "There are " << argc << " arguments:" << endl;
	    // Loop through each argument and print its number and value
	    for (int nArg=0; nArg < argc; nArg++)
	        cout << nArg << " " << argv[nArg] << endl;
        */
    if (argc >= 2)
    {
        string table;
        string where;
        string kmlfile;
        string name;
        string file;
        string type;
        string delim;

        switch (argv[1][1]) {

      case 'h':   cout << "You have selected Help\n";
                  parametersHelp();

            break;

      case 'a':   cout << "You have selected Get Array of Query Function\n";
                  if (argc == 4)
                  {
                      table = argv[2];
                      where = argv[3];
                      cout << "You selected table=" << table << " and where=" << where << "\n";
                      printQueryResults(table, where);
                  }
                  else
                  {
                      cout << "You did not enter data correctly see help\n";
                      parametersHelp();
                  }
            break;

      case 'b':   cout << "You have selected Generate New KML from Query\n";
                 if (argc > 3)
                  {
                      table = argv[2];
                      where = argv[3];
                      if (argc < 5)
                      {
                          name = "";
                      }
                      else
                      {
                          name = argv[4];
                      }
                      cout << "You selected table=" << table << " and where=" << where << "\n";
                      kmlFromQuery(table, where, name);
                  }
                  else
                  {
                      cout << "You did not enter both table and where statements, see help\n";
                      parametersHelp();
                  }

            break;

      case 'c':   cout << "You have selected Insert New Entry and update KML\n";
                  cout << "-This function has not been impletmented yet!\n";

            break;

      case 'd':   cout << "You have selected Update Field's Cell by ID and update KML\n";
                  cout << "-This function has not been impletmented yet!\n";

            break;

      case 'e':   cout << "You have selected Regenerate KML for specific day\n";
                  cout << "-This function has not been impletmented yet!\n";

            break;

      case 'f':   cout << "You have selected Pivot Table Query and Generate KML\n";
                  cout << "-This function has not been impletmented yet!\n";

            break;

      case 'g':   cout << "You have selected Python/Matlab Analysis, return KML\n";
                  cout << "-This function has not been impletmented yet!\n";

            break;

      case 'k':   cout << "You have selected to convert a KML for ArtifactVis\n";
                  if (argc == 3)
                  {
                      kmlfile = argv[2];
                      convertKML(kmlfile);
                  }
                  else
                  {
                      cout << "You must enter the kmlfile you wish to convert\n";
                  }


            break;

      case 'l':   cout << "You have selected to list Tables/Columns\n";
                  if (argc == 3)
                  {
                      table = argv[2];
                  }
                  else
                  {
                      table = "";
                  }
                  listTables(table);

            break;

      case 'm':   cout << "You have selected to generate an xml menu\n";
                  if (argc == 3)
                  {
                  table = argv[2];
                  getDistinctLists(table);
                  }
                  else
                  {
                      cout << "Please Input Table(s)!\n";
                  }

            break;


      case 'n':   cout << "You have selected to delete a kml file and its entry\n";
                  if (argc == 3)
                  {
                  name = argv[2];
                  deleteQfile(name);
                  }
                  else
                  {
                      cout << "Please Input Kml name!\n";
                  }

            break;

      case 'r':   cout << "You have selected to rename a queried file\n";
                  if (argc == 3)
                  {
                  kmlfile = argv[2];
                  renameQfile(kmlfile,"");
                  }
                  else if (argc == 4)
                  {
                  kmlfile = argv[2];
                  name = argv[3];
                  renameQfile(kmlfile,name);
                  }
                  else
                  {
                      cout << "You must enter the name of the file you want to rename!\n";
                  }

            break;

      case 's':   cout << "You have selected to sync ArchField tables\n";
                  cout << "-This function will only test connection and tables it will not make any changes!\n";
                  if (argc == 3)
                  {
                  table = argv[2];
                  syncLogon(table);
                  }
                  else
                  {
                      cout << "Please Input Table to Sync!\n";
                  }


            break;


      case 'u':   cout << "You have selected to upload tables\n";
                  //cout << "-This function currently only runs importDemo()!\n";
                  if (argc == 6)
                  {
                  file = argv[2];
                  table = argv[3];
                  type = argv[4];
                  delim = argv[5];

                  importCSV(file, table, type, delim);

                  }
                  else
                  {
                      cout << "Please Input: \"File\" \"Name\" \"Type\" \"Delimeter(e.g. ',' or '|')\"\n";
                      importDemos();
                  }




            break;

      default: cout << "You must select a  proper parameter with '-' \n";
                parametersHelp();

            break;

       }
    }
    else
    {
    cout << "You need to give some command\n";
    parametersHelp();

    }
}
void listTables(string table)
{
    string query;
    PGresult        *res;


if (table != "")
{


    if (table == "xml")
    {
        cout << "You have selected to output list of tables into xml!\n";

        query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'";
        res = queryPG(query,"");

        int rows = PQntuples(res);
        if (rows > 0)
        {

            int i;

            mxml_node_t *xml;    /* <?xml ... ?> */
                mxml_node_t *tablenames;   /* <field> */

            xml = mxmlNewXML("1.0");
                tablenames = mxmlNewElement(xml, "tables");
                    string value;
                    string list;
                    const char* listChar;

            for (i=0; i<rows; i++)
            {
                value = PQgetvalue(res, i, 0);
                if (value != "spatial_ref_sys" && value != "geometry_columns" && value != "geography_columns" )
                {
                    //cout << value << "\n";
                    list.append(value);
                    list.append("\n");
                }
            }
            listChar = list.c_str();
            mxmlNewText(tablenames, 0, listChar);

            FILE *fp;
            //const char* kml_name = "Query Test";
            fp = fopen("tables.xml", "w");
            mxmlSaveFile(xml, fp, MXML_NO_CALLBACK);
            fclose(fp);

            cout << "File Written! Completed!\n";
        }
        else
        {
            cout << "No Tables in Database!\n";
        }

    }
    else
    {
        query = "Select column_name FROM information_schema.columns WHERE table_name='";
        query.append(table);
        query.append("'");
        res = queryPG(query,"");
        int i;
        int rows = PQntuples(res);

        if (rows > 0)
        {
            cout << "Listing Columns for Table: " << table << "\n";
            for (i=0; i<rows; i++)
            {
                cout << PQgetvalue(res, i, 0) << "\n";
            }
        }
        else
        {
            cout << "Table does not exist or no columns available!\n";
        }
    }


}
else
{

    query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'";
    res = queryPG(query,"");

    int rows = PQntuples(res);
    int i;
    for (i=0; i<rows; i++)
    {
        cout << PQgetvalue(res, i, 0) << "\n";
    }
}

}
void parametersHelp()
{

    cout << "usage: ArchInterface [options]\n";
    cout << "\n";
    cout << "options:\n";
    cout << "-h                                               Help\n";
    cout << "-a \"[table name]\" \"[where statement]\"            Get Array of Query Function\n";
    cout << "   (Example Where statement \"basket = '50005' AND dccode = 'SP'\")\n";
    cout << "\n";
    cout << "-b \"[table names*|]\" \"[where statement]\"         Generate New KML from Query\n";
    cout << "-c \"[table name]\" \"[data*|]\"                     Insert New Entry and update KML\n";
    cout << "-d \"[table name]\" \"[id]\" \"[field]\" \"[data]\"      Update Field's Cell by ID and update KML\n";
    cout << "-e \"[table name]\" \"[date]\"                       Regenerate KML for specific day\n";
    cout << "-f \"[table names*|]\" \"[where statement]\" \"[field for rows]\" \"[field for columns]\" Pivot Table Query and Generate KML\n";
    cout << "-g \"[table names*|]\" \"[where statement]\" \"[analysis type]\"  Python/Matlab Analysis, return KML\n";
    cout << "-l \"[table name]\"                                  List available tables, if table name entered lists fields, use xml for table if want tables.xml\n";
    cout << "-m \"[table names*]\"                                Generate XML Menu\n";
    cout << "-s \"[table names*]\"                                Sync Archfield tables between Remote and Server\n";
    cout << "-u \"[file]\"                                        Upload CSV table\n";
}
void syncLogon(string table)
{
    string param;
    string serverName;
    PGconn          *conn;
    const char*    c;
    bool check;
    bool check2;

    //.........................................................
    //Check Local Server
    serverName = "Local";
    param = connectParam(serverName);
    c = param.c_str();
    conn = PQconnectdb(c);

    if (PQstatus(conn) == CONNECTION_BAD)
    {
        cout << "Can't Connect to Local Server\n";
        check = false;
    }
    else
    {
        cout << "Connected to Local Server\n";
        check = true;
    }
    //..........................................................
    //
    serverName = "Central";
    param = connectParam(serverName);
    c = param.c_str();
    conn = PQconnectdb(c);

    if (PQstatus(conn) == CONNECTION_BAD)
    {
        cout << "Can't Connect to Central Server\n";
        check2 = false;
    }
    else
    {
        cout << "Connected to Central Server\n";
        check2 = true;
    }

    if (check && check2)
    {
        vector< vector<string> > output;
        output = parseConfigFile();
        string local = output[2][1];
        string central = output[7][1];
        if (local != central)
        {
        cout << "Will sync Local table: " << table << " with Central Server!\n";
        syncTable(table);
        }
        else
        {
            cout << "Local server is the same as Central server, nothing to sync\n";
            syncTable(table);
        }
    }
    else
    {
        cout << "Fix connection settings!\n";
    }
}

string connectParam(string serverName)
{
        vector< vector<string> > output;
        string cinfo;
       // const char* c;

    output = parseConfigFile();
    //cout << output[4][0] << ": " << output[4][1] << "\n";
    /*
    cout << "server name: " << output[0][1] << "\n";
    cout << "dbname: " << output[1][1] << "\n";
    cout << "host: " << output[2][1] << "\n";
    cout << "user: " << output[3][1] << "\n";
    cout << "password: " << output[4][1] << "\n";
    cout << "server name: " << output[5][1] << "\n";
    cout << "dbname: " << output[6][1] << "\n";
    cout << "host: " << output[7][1] << "\n";
    cout << "user: " << output[8][1] << "\n";
    cout << "password: " << output[9][1] << "\n";
    */
    //int count = output.size();
    //cout << count << "\n";

    if (serverName == "" || serverName == "Local")
    {
        cinfo = "dbname=";
        cinfo.append(output[1][1]);
        cinfo.append(" host=");
        cinfo.append(output[2][1]);
        cinfo.append(" user=");
        cinfo.append(output[3][1]);
        cinfo.append(" password=");
        cinfo.append(output[4][1]);
        //c = cinfo.c_str ();
    }
    else if (serverName == "Central")
    {
        cinfo = "dbname=";
        cinfo.append(output[6][1]);
        cinfo.append(" host=");
        cinfo.append(output[7][1]);
        cinfo.append(" user=");
        cinfo.append(output[8][1]);
        cinfo.append(" password=");
        cinfo.append(output[9][1]);
       // c = cinfo.c_str ();
    }
    return cinfo;
}

PGconn *connectDB(string param)
{
//This function returns the connection "conn" variable needed to perform all queries
//The string for PQconnectdb must be set to all the correct information or it will fail
//Currently it is setup to the default Archfield database.

    PGconn          *conn;

    const char*    c = param.c_str();
    conn = PQconnectdb(c);

         if (PQstatus(conn) == CONNECTION_BAD) {
                 puts("We were unable to connect to the ddatabase");
                exit(0);
         }
         else
         {
             //cout << "Connection Established";
         }

return conn;
}

PGresult *queryPG(string query, string serverName)
{
//This function can send any query to the database and simplifies everything for us.


    PGconn          *conn;
    PGresult *res;

    string param;
    //string serverName = "";

    param = connectParam(serverName);
    //cout << param << "\n";

    conn = connectDB(param);

    res = PQexec(conn,query.c_str());

    PQfinish(conn);

    return res;

}

vector< vector<string> > queryPostGIS(string query, string list, string type, string srid)
{
//This is the main function used in ArchField to pull out spatial data
//It is designed to streamline the complicated process of extracting the spatial information
//It is designed to handle either latlong or utm and point or polygon
//Note that we are not returning the final array, just yet!

    PGresult        *res;
    string          transform;
    string          qending;
    string          temp;
    int             pos;

    int             count;
    char               delim;

    //..................................................................
    //If the srid (projection--utm or latlong) is set this will handle it

    if (srid != "")
    {
     transform = "ST_Transform(the_geom, ";
     transform.append(srid);
     transform.append(")");
    }
    else
    {
     transform = "the_geom";
    }
//...............................................
//This section sets up the specific query, accounting for the specific geometry syntax

    if (type == "Point")
    {
    //pos = strpos(query,"FROM");
    //temp = query;
    temp = query;
    pos = query.find("FROM", 0);
    pos = pos - 1;
    qending = temp.erase(0,pos);
    //query = query.erase(0, pos);
    query = query.erase(pos);
    query.append(", ST_X("); //Inorder to get x,y,z from a point each must be called seperately
    query.append(transform);
    query.append(") ");
    query.append(", ST_Y(");
    query.append(transform);
    query.append(") ");
    query.append(", ST_Z(");
    query.append(transform);
    query.append(") ");
    query.append(qending);
    //cout << query << "\n";
    //query = substr($query,0,$pos-1) . ', ST_X(' . $transform . ') ' . ', ST_Y(' . $transform . ') ' . ', ST_Z(' . $transform . ') ' . substr($query,$pos);
    }
    else if (type == "Polygon")
    {
    temp = query;
    pos = query.find("FROM", 0);
    pos = pos - 1;
    qending = temp.erase(0,pos);
    //query = query.erase(0, pos);
    query = query.erase(pos);
    query.append(", ST_AsEWKT("); //This pulls out the ascii format and retains the z.
    query.append(transform);
    query.append(") ");
    query.append(qending);
    //cout << query << "\n";

    //$query = substr($query,0,$pos-1) . ', ST_AsText(' . $transform . ') ' . substr($query,$pos);
    }
    else if (type == "KML-P")
    {
    temp = query;
    pos = query.find("FROM", 0);
    pos = pos - 1;
    qending = temp.erase(0,pos);
    //query = query.erase(0, pos);
    query = query.erase(pos);
    query.append(", ST_AsKML(");  //This pulls out the format used for polygon KML files
    query.append(transform);
    query.append(") ");
    query.append(qending);
    //cout << query << "\n";
    //$query = substr($query,0,$pos-1) . ', ST_AsKML(ST_Reverse(' . $transform . ')) ' . substr($query,$pos);
    }

    res = queryPG(query,"");


//...............................................
//Converts list sent by user to array for next for loop

    delim = ',';
    vector<string>data;
    data = splitSTR(list,delim);
//.....................................................
//Now we loop through the results and store them in the array named entry
//Note that the geometry must be handled seperately
    int rows = PQntuples(res);

    //cout << "Rows: " << rows << "\n";
    count = data.size();
    vector<string>entry;
    vector< vector<string> > rowdata;
    int i;
    int m;
    string value;
    //int lens;
    //count = 0;
for(m=0; m<rows; m++)
{
    entry.clear();
    for (i=0; i<count; i++)
	{

	    if(data[i] == "the_geom" && type == "Point")
	    {
	        string coord;
	        string x;
	        string y;
	        string z;

            x = PQgetvalue(res, m, i);
            coord = x;

            y = PQgetvalue(res, m, i+1);
            coord.append(", ");
            coord.append(y);

            z = PQgetvalue(res, m, i+2);
            coord.append(", ");
            coord.append(z);

            entry.push_back(coord); //The final results is KML point format in latlong
            //cout << entry[i] << "\n";
	    }
	    else if (data[i] == "the_geom" && type == "Polygon")
	    {
	        string coord;
	        int len;
            coord = PQgetvalue(res, m, i);
            coord = coord.erase(0,20); //We just want the raw coords, this gets rid of the rest
            len = coord.length() - 2;
            coord = coord.erase(len,2);
            entry.push_back(coord);  //The final results is text format in UTM
             //cout << entry[i] << "\n";

	    }
	    else if (data[i] == "the_geom" && type == "KML-P")
	    {
            string coord;
	        int len;
            coord = PQgetvalue(res, m, i);
            //cout << coord << "\n";
            coord = coord.erase(0,51);  //We just want the raw coords, this gets rid of the rest
            len = coord.length() - 55;
            coord = coord.erase(len,55);
            entry.push_back(coord);  //The final result is a KML polygon coords.
             //cout << entry[i] << "\n";
	    }
	    else
	    {
	        value = PQgetvalue(res, m, i);
	        //cout << value << "\n";
             if (value == "" || value == "\r")
             {
                // cout << value << "Exists" << "\n";
             entry.push_back("NULL");
             }
             else
             {
             entry.push_back(PQgetvalue(res, m, i));  //standard columns can all be stored in the same way
             //cout << data[i] << ": " << entry[i] << "\n";
             }

	    }

    }

    rowdata.push_back(entry);
}

    return rowdata;
}
void newtablePG(string table)
{
//This function creates two tables (points and polygons) for a new project in Archfield.
//This function is now obsolete and will be replaced with newPointTable and newPolygonTable

PGresult        *res;
string query;
string tableS;
string tableL;
int rows;

tableS = table;
tableS.append("sf");
tableL = table;
tableL.append("l");

//......................................................
//Check if table already exists!
query = "SELECT true FROM pg_tables WHERE tablename = '";
query.append(tableS);
query.append("';");

res = queryPG(query,"");
rows = PQntuples(res);


//rows = 0;
if(rows == 0) //If table does not exist create both tables and insert starter data
{
    //Create Special Finds Table

    query = "CREATE TABLE ";
    query.append(tableS);
    query.append(" (");
    query.append("gid       int4 PRIMARY KEY,");
	query.append("basket        varchar(255),");
	query.append("chk       varchar(255),");
	query.append("dccode       varchar(255),");
	query.append("locus       varchar(255),");
	query.append("square       varchar(255),");
	query.append("date       varchar(255),");
	query.append("area       varchar(255),");
	query.append("publication      varchar(255),");
	query.append("heightinstrument      varchar(255),");
	query.append("poleheight       varchar(255),");
	query.append("bone       varchar(255),");
	query.append("flint       varchar(255),");
	query.append("pottery       varchar(255),");
	query.append("rc       varchar(255),");
	query.append("soilsample       varchar(255),");
	query.append("crate       varchar(255),");
	query.append("location       varchar(255)");
    query.append(");");

    res = queryPG(query,"");

    cout << query << "\n";

    //.....................................
    //This next query adds the geometry column (PostGIS will also insert this table as an entry
    //into the "geometry_columns" table.

    query = "SELECT AddGeometryColumn('public', '";
    query.append(tableS);
    query.append("', 'the_geom', 32636, 'POINT', 4)");

    res = queryPG(query,"");

    cout << query << "\n";

//......................................
//These next three queries remove checks on the table just created that would prevent easy data entry

    query = "ALTER TABLE ";
    query.append(tableS);
    query.append(" DROP CONSTRAINT enforce_srid_the_geom");

    res = queryPG(query,"");

    cout << query << "\n";

    query = "ALTER TABLE ";
    query.append(tableS);
    query.append(" DROP CONSTRAINT enforce_dims_the_geom");

    res = queryPG(query,"");

    cout << query << "\n";

    query = "ALTER TABLE ";
    query.append(tableS);
    query.append(" DROP CONSTRAINT enforce_geotype_the_geom");

    res = queryPG(query,"");

    cout << query << "\n";
//...............................................................
//Now insert first row of data for table

    //Initialize string variables for entring in data into row
    string      gid;
    string      basket;
    string      locus;
    string      dccode;
    string      square;
    string      date;
    string      area;
    string      publication;
    string      the_geom;
    string      locusdesc;
    //.....................................

    //Set variables for data entry (This shows how to enter data
    //into a row that includes spatial information).

    gid = "1";
    basket = "50111";
    locus = "4012";
    dccode = "SP-Special Pottery";
    square = "A1";
    date = "072210";
    area = "C";
    publication = "KIS12";
    the_geom = "739942.846 3373218.220 1451.294";

    query = "INSERT INTO ";
    query.append(tableS);
    query.append(" (gid, basket, locus, dccode, square, date, area, publication, the_geom) VALUES (");
    query.append(gid);
    query.append(", '");
    query.append(basket);
    query.append("', '");
    query.append(locus);
    query.append("', '");

    query.append(dccode);
    query.append("', '");

    query.append(square);
    query.append("', '");

    query.append(date);
    query.append("', '");

    query.append(area);
    query.append("', '");

    query.append(publication);

    query.append("', ST_GeomFromText('POINT(");
    query.append(the_geom);
    query.append(")',32636))");

    res = queryPG(query,"");
    cout << res << "\n";
    cout << query << "\n";
    cout << "======================================" << "\n";

    //...............................................

    //Create Polygon Table--Everything is the same as above except that polygons are handled slightly different

    query = "CREATE TABLE ";
    query.append(tableL);
    query.append(" (");
    query.append("gid       int4 PRIMARY KEY,");
	query.append("basket        varchar(255),");
	query.append("chk       varchar(255),");
	query.append("dccode       varchar(255),");
	query.append("locusdesc       varchar(255),");
	query.append("locus       varchar(255),");
	query.append("square       varchar(255),");
	query.append("date       varchar(255),");
	query.append("area       varchar(255),");
	query.append("structure       varchar(255),");
	query.append("publication      varchar(255),");
	query.append("heightinstrument      varchar(255),");
	query.append("poleheight       varchar(255)");
	query.append(");");

    res = queryPG(query,"");

    cout << query << "\n";

    //Inserting the_geom column--note that the main difference is the polygon entry instead of Point
    query = "SELECT AddGeometryColumn('public', '";
    query.append(tableL);
    query.append("', 'the_geom', 32636, 'POLYGON', 4)");

    res = queryPG(query,"");

    cout << query << "\n";

    //.........................................................
    //Remove Checks
    query = "ALTER TABLE ";
    query.append(tableL);
    query.append(" DROP CONSTRAINT enforce_srid_the_geom");

    res = queryPG(query,"");

    cout << query << "\n";

    query = "ALTER TABLE ";
    query.append(tableL);
    query.append(" DROP CONSTRAINT enforce_dims_the_geom");

    res = queryPG(query,"");

    cout << query << "\n";

    query = "ALTER TABLE ";
    query.append(tableL);
    query.append(" DROP CONSTRAINT enforce_geotype_the_geom");

    res = queryPG(query,"");

    cout << query << "\n";

    //.............................................
    //Insert first row--note that locusdesc was initialized earlier

    gid = "1";
    basket = "55111";
    locus = "4012";
    dccode = "PS-Perimeter Start";
    locusdesc = "FL-Floor";
    square = "A1";
    date = "072210";
    area = "C";
    publication = "KIS12";
    the_geom = "739946.31382673 3373205.2601785 1451.3232239906,739946.33765209 3373207.0434307 1451.2699909167,739948.02643762 3373207.0188115 1451.3527477566,739947.86870826 3373205.2422077 1451.3214121321,739946.31382673 3373205.2601785 1451.3232239906";

    query = "INSERT INTO ";
    query.append(tableL);
    query.append(" (gid, basket, locus, dccode, locusdesc, square, date, area, publication, the_geom) VALUES (");
    query.append(gid);
    query.append(", '");
    query.append(basket);
    query.append("', '");
    query.append(locus);
    query.append("', '");

    query.append(dccode);
    query.append("', '");

    query.append(locusdesc);
    query.append("', '");

    query.append(square);
    query.append("', '");

    query.append(date);
    query.append("', '");

    query.append(area);
    query.append("', '");

    query.append(publication);

    query.append("', ST_GeomFromText('POLYGON((");
    query.append(the_geom);
    query.append("))',32636))");

  //  query.append("')");

    res = queryPG(query,"");
    cout << res << "\n";
    cout << query << "\n";


}


}

void dropTable(string table)
{
    PGresult        *res;
    string          query;

    query = "SELECT DropGeometryTable('";
    query.append(table);
    query.append("')");
    res = queryPG(query,"");
    cout << res << "\n";
}

vector<string> splitSTR(string tmp, char lim)
{
//This function was the best solution I found on the internet for exploding a string into an array.
//If you know of a faster or better way, please let me know.

    vector<string>array;
    string token;
    //string tmp = "this@is@a@line";

    istringstream iss(tmp);

    while ( getline(iss, token, lim) )
    {
    array.push_back(token);
    }
return array;
}

vector<string> fileIntoArray(const char* file)
{
    //This function takes a large text file and puts each line into an array


    vector<string>array;
    string token;
    ifstream myfile(file);

    while(!myfile.eof())
    {
        getline(myfile,token,'\n');
        array.push_back(token);

    }

return array;
}
vector< vector<string> > parseConfigFile()
{
    vector<string>array;
    int count;
    char delim;
    //char file[256];
    const char* file;
    file = "config.ini";

    array = fileIntoArray(file);

    count = array.size();
    //cout << count << "\n";
    delim = '=';
    //vector<string>data;
    vector< vector<string> > data;
    //(3, vector<string>(2,0));
    int i;
    //count = 0;
    string test;
    for (i=0; i<count; i++)
	{
	    //cout << array[i] << "\n";
	data.push_back(splitSTR(array[i],delim));
	//test = data[i][1];
	//cout << i << " " << test << "\n";
	}


//cout << data[4][0] << "= " << data[4][1] << "\n";

//data[4][1] = "";
return data;
}
void importCSV(string sfile, string table, string type, string delim)
{
    //This function reads a csv file and puts it into a two-dimensional array.
    //Currently it is also functioning to directly import the table into PostGIS
    //Later we will just have it return the 2D array.

    vector<string>array;
    int count;
    //char delim;
    //char file[256];
    const char* file;
    file = sfile.c_str ();

    //file = "config.ini";

    array = fileIntoArray(file);

    count = array.size();

    //delim = '|';
    //vector<string>data;
    vector< vector<string> > data;
    //(3, vector<string>(2,0));
    int i;
    //count = 0;

    char cdelim;
    if (delim == "|")
    {
        cdelim = '|';
    }
    else if (delim == ",")
    {
        cdelim = ',';
    }
    else if (delim == ";")
    {
        cdelim = ';';
    }
    else if (delim == " ")
    {
        cdelim = ' ';
    }
    else
    {
        cdelim = ',';
    }

    for (i=0; i<count; i++)
	{
	data.push_back(splitSTR(array[i],cdelim));
	}
	string fields = array[0];

    //This section creates new tables and add the data from the csv
    //It also checks to make sure if the table has not already been created
    //If the table exists the option is provided to drop it.
    if (type == "")
    {
    type = table;
    int len = type.length() - 1;
    type = type.erase(0,len);
    cout << type << "\n";
    }

    int exists;
    string eraseTbls;
    if (type == "f")
    {
        exists = newPointTable(table);
        if (exists == 0)
        {
        importPointData(table, data);
        }
        else
        {
            cout << "This table already exists, would you like to drop it? (y,n): ";
            cin >> eraseTbls;
            cin.ignore();

            if (eraseTbls == "y")
            {
                dropTable(table);
            }

        }
    }
    else if (type == "l")
    {
        exists = newPolygonTable(table);
        if (exists == 0)
        {
        importPolygonData(table, data);
        }
        else
        {
            cout << "This table already exists, would you like to drop it? (y,n): ";
            cin >> eraseTbls;
            cin.ignore();

            if (eraseTbls == "y")
            {
                dropTable(table);
            }

        }
    }
    else if (type=="std")
    {
        cout << "Standard Table Creation\n";
        //string fields = "id|basket";
        exists = newStdTable(table, fields);
        if (exists == 0)
        {
        importTableData(table, data, fields);
        }
        else
        {
            cout << "This table already exists, would you like to drop it? (y,n): ";
            cin >> eraseTbls;
            cin.ignore();

            if (eraseTbls == "y")
            {
                dropTable(table);
                cout << "Table Dropped! Rerun to import new table\n";
            }

        }
    }

}
void importTableData(string table, vector< vector<string> > data, string fields )
{
    //This function will only work for the provided csv, I will make it universal later

    PGresult        *res;
    //Initialize string variables for entring in data into row
    /*
    string      gid;
    string      basket;
    string      chk;
    string      locus;
    string      dccode;
    string      square;
    string      date;
    string      area;
    string      publication;
    string      heightinstrument;
    string      poleheight;
    string      bone;
    string      flint;
    string      pottery;
    string      rc;
    string      soilsample;
    string      crate;
    string      location;
    string      the_geom;
    */

    int i;
    //i = 0;
    string query;
    int count = data.size();
    cout << count << "\n";
    //count = 0;
    cout << "This may take a while, have patience! \n";

    int m;
//    int count;

    vector<string>array;
    char delim = ',';
    array = splitSTR(fields,delim);

    int count2 = array.size();
    //count = count -1;
    for (i=1; i<count; i++)
    {
        /*
    basket = data[i][0];
    chk = data[i][1];
    dccode = data[i][2];
    locus = data[i][3];
    square = data[i][4];
    date = data[i][5];
    area = data[i][6];
    publication = data[i][7];
    heightinstrument = data[i][8];
    poleheight = data[i][9];
    bone = data[i][10];
    flint = data[i][11];
    pottery = data[i][12];
    the_geom = data[i][13];
    gid = data[i][14];
    rc = data[i][15];
    soilsample = data[i][16];
    crate = data[i][17];

    int last = data[i].size();
    if (last == 19)
    {
    location = data[i][18];
    }
    else
    {
        location = "";
    }

    */
    query = "INSERT INTO ";
    query.append(table);
    //query.append(" (gid, basket, chk, locus, dccode, square, date, area, publication, heightinstrument, poleheight, bone, flint, pottery, rc, soilsample, crate, location, the_geom) VALUES (");
    query.append(" (");
    query.append(fields);
    query.append(") VALUES (");

    int check = data[i].size();
    //cout << check << "\n";
if (check != 0)
{

    if (check < count2)
    {
        data[i].push_back("");
    }

    for (m=0; m<count2; m++)
    {
        if (m == 0)
        {
            query.append(data[i][m]);
            query.append(", '");
        }
        else if (m == (count2-1))
        {
            query.append(data[i][m]);
            //query.append("', '");
        }
        else
        {
            query.append(data[i][m]);
            query.append("', '");
        }

    }

    query.append("')");

    res = queryPG(query,"");

    //cout << query << "\n";
}
    }




}
void importPointData(string table, vector< vector<string> > data )
{
    //This function will only work for the provided csv, I will make it universal later

    PGresult        *res;
    //Initialize string variables for entring in data into row
    string      gid;
    string      basket;
    string      chk;
    string      locus;
    string      dccode;
    string      square;
    string      date;
    string      area;
    string      publication;
    string      heightinstrument;
    string      poleheight;
    string      bone;
    string      flint;
    string      pottery;
    string      rc;
    string      soilsample;
    string      crate;
    string      location;
    string      the_geom;


    int i;
    //i = 0;
    string query;
    int count = data.size();
    cout << count << "\n";
    //count = 0;
    cout << "This may take a while, have patience! \n";

    for (i=0; i<count; i++)
    {
    basket = data[i][0];
    chk = data[i][1];
    dccode = data[i][2];
    locus = data[i][3];
    square = data[i][4];
    date = data[i][5];
    area = data[i][6];
    publication = data[i][7];
    heightinstrument = data[i][8];
    poleheight = data[i][9];
    bone = data[i][10];
    flint = data[i][11];
    pottery = data[i][12];
    the_geom = data[i][13];
    gid = data[i][14];
    rc = data[i][15];
    soilsample = data[i][16];
    crate = data[i][17];

    int last = data[i].size();
    if (last == 19)
    {
    location = data[i][18];
    }
    else
    {
        location = "";
    }


    query = "INSERT INTO ";
    query.append(table);
    query.append(" (gid, basket, chk, locus, dccode, square, date, area, publication, heightinstrument, poleheight, bone, flint, pottery, rc, soilsample, crate, location, the_geom) VALUES (");
    query.append(gid);
    query.append(", '");
    query.append(basket);
    query.append("', '");
    query.append(chk);
    query.append("', '");
    query.append(locus);
    query.append("', '");
    query.append(dccode);
    query.append("', '");
    query.append(square);
    query.append("', '");
    query.append(date);
    query.append("', '");
    query.append(area);
    query.append("', '");
    query.append(publication);
    query.append("', '");
    query.append(heightinstrument);
    query.append("', '");
    query.append(poleheight);
    query.append("', '");
    query.append(bone);
    query.append("', '");
    query.append(flint);
    query.append("', '");
    query.append(pottery);
    query.append("', '");
    query.append(rc);
    query.append("', '");
    query.append(soilsample);
    query.append("', '");
    query.append(crate);
    query.append("', '");
    query.append(location);
    query.append("', '");
    query.append(the_geom);
    query.append("')");
/*
    query.append("', ST_GeomFromText('POINT(");
    query.append(the_geom);
    query.append(")',32636))");
*/
    res = queryPG(query,"");
    //cout << res << "\n";
    //cout << the_geom << "\n";
    //cout << query << "\n";
    }

/*
    cout << "\n" << "============Points===============" << "\n";

//table = "con01sf";
query = "SELECT Basket,Chk,DCCODE,Locus,Square,Date,Area,Publication,bone,flint,pottery,rc,soilsample,location,crate FROM ";
query.append(table);
query.append(" WHERE basket = '50737'");
string list = "Basket,Chk,DCCODE,Locus,Square,Date,Area,Publication,bone,flint,pottery,rc,soilsample,location,crate,the_geom";
string type = "Point";
string srid = "32636";

queryPostGIS(query, list, type, srid);
*/

}
void importPolygonData(string table, vector< vector<string> > data )
{
    //This function will only work for the provided csv, I will make it universal later
    PGresult        *res;
    //Initialize string variables for entring in data into row
    string      gid;
    string      basket;
    string      chk;
    string      locus;
    string      dccode;
    string      locusdesc;
    string      square;
    string      date;
    string      area;
    string      structure;
    string      publication;
    string      heightinstrument;
    string      poleheight;
    string      the_geom;


    int i;
    //i = 0;
    string query;

    int count = data.size();
    cout << count << "\n";
    //count = 1;
    cout << "This may take a while, have patience! \n";

    for (i=0; i<count; i++)
    {
    basket = data[i][0];
    chk = data[i][1];
    dccode = data[i][2];
    locusdesc = data[i][3];
    locus = data[i][4];
    square = data[i][5];
    date = data[i][6];
    area = data[i][7];
    structure = data[i][8];
    publication = data[i][9];
    heightinstrument = data[i][10];
    poleheight = data[i][11];
    the_geom = data[i][12];
    gid = data[i][13];
/*
    int last = data[i].size();
    if (last == 19)
    {
    location = data[i][18];
    }
    else
    {
        location = "";
    }
*/

    query = "INSERT INTO ";
    query.append(table);
    query.append(" (gid, basket, chk, dccode, locusdesc, locus, square, date, area, structure, publication, heightinstrument, poleheight, the_geom) VALUES (");
    query.append(gid);
    query.append(", '");
    query.append(basket);
    query.append("', '");
    query.append(chk);
    query.append("', '");
    query.append(dccode);
    query.append("', '");
    query.append(locusdesc);
    query.append("', '");
    query.append(locus);
    query.append("', '");
    query.append(square);
    query.append("', '");
    query.append(date);
    query.append("', '");
    query.append(area);
    query.append("', '");
    query.append(structure);
    query.append("', '");
    query.append(publication);
    query.append("', '");
    query.append(heightinstrument);
    query.append("', '");
    query.append(poleheight);
    query.append("', '");
    query.append(the_geom);
    query.append("')");
/*
    query.append("', ST_GeomFromText('POINT(");
    query.append(the_geom);
    query.append(")',32636))");
*/
    res = queryPG(query,"");
    //cout << res << "\n";
    //cout << the_geom << "\n";
    //cout << query << "\n";
    }
/*
//Test Polygons
cout << "\n" << "============Polygons===============" << "\n";
//table = "con01l";
query = "SELECT Basket,dccode,chk,date FROM ";
query.append(table);
query.append(" WHERE Basket = '55093'");
string list = "Basket,dccode,chk,date,the_geom";
string type = "Polygon";
string srid = "32636";


queryPostGIS(query, list, type, srid);
*/




}

int newStdTable(string tableS, string fields)
{
//Creates a new point table for ArchField

PGresult        *res;
string query;
//string tableS;
//string tableL;
int rows;


//......................................................
//Check if table already exists!
query = "SELECT true FROM pg_tables WHERE tablename = '";
query.append(tableS);
query.append("';");

res = queryPG(query,"");
rows = PQntuples(res);


//rows = 0;
    if(rows == 0) //If table does not exist create both tables and insert starter data
    {
    //Create Special Finds Table

    query = "CREATE TABLE ";
    query.append(tableS);
    query.append(" (");

    vector<string>array;
    char delim = ',';
    array = splitSTR(fields,delim);
    int count = array.size();
    int i;
    for (i=0; i<count; i++)
    {

        if (array[i] == "id" || array[i] == "gid" )
        {
            query.append(array[i]);
            query.append("       int4 PRIMARY KEY,");
        }
        else if (i == (count-1))
        {
            query.append(array[i]);
            query.append("        varchar(255)");
        }
        else
        {
            query.append(array[i]);
            query.append("        varchar(255),");
        }

    }
    //cout << fields << "\n";
    /*
    query.append("gid       int4 PRIMARY KEY,");
	query.append("basket        varchar(255),");
	query.append("chk       varchar(255),");
	query.append("dccode       varchar(255),");
	query.append("locus       varchar(255),");
	query.append("square       varchar(255),");
	query.append("date       varchar(255),");
	query.append("area       varchar(255),");
	query.append("publication      varchar(255),");
	query.append("heightinstrument      varchar(255),");
	query.append("poleheight       varchar(255),");
	query.append("bone       varchar(255),");
	query.append("flint       varchar(255),");
	query.append("pottery       varchar(255),");
	query.append("rc       varchar(255),");
	query.append("soilsample       varchar(255),");
	query.append("crate       varchar(255),");
	query.append("location       varchar(255)");
	*/
    query.append(");");

    res = queryPG(query,"");

    //cout << query << "\n";

    //.....................................
    //This next query adds the geometry column (PostGIS will also insert this table as an entry
    //into the "geometry_columns" table.
    /*
    query = "SELECT AddGeometryColumn('public', '";
    query.append(tableS);
    query.append("', 'the_geom', 32636, 'POINT', 4)");

    res = queryPG(query,"");

    cout << query << "\n";
    */
//......................................
//These next three queries remove checks on the table just created that would prevent easy data entry
    /*
    query = "ALTER TABLE ";
    query.append(tableS);
    query.append(" DROP CONSTRAINT enforce_srid_the_geom");

    res = queryPG(query,"");

    cout << query << "\n";

    query = "ALTER TABLE ";
    query.append(tableS);
    query.append(" DROP CONSTRAINT enforce_dims_the_geom");

    res = queryPG(query,"");

    cout << query << "\n";

    query = "ALTER TABLE ";
    query.append(tableS);
    query.append(" DROP CONSTRAINT enforce_geotype_the_geom");

    res = queryPG(query,"");

    cout << query << "\n";
    */


    }
    return rows;
}
int newPointTable(string tableS)
{
//Creates a new point table for ArchField

PGresult        *res;
string query;
//string tableS;
//string tableL;
int rows;


//......................................................
//Check if table already exists!
query = "SELECT true FROM pg_tables WHERE tablename = '";
query.append(tableS);
query.append("';");

res = queryPG(query,"");
rows = PQntuples(res);


//rows = 0;
    if(rows == 0) //If table does not exist create both tables and insert starter data
    {
    //Create Special Finds Table

    query = "CREATE TABLE ";
    query.append(tableS);
    query.append(" (");
    query.append("gid       int4 PRIMARY KEY,");
	query.append("basket        varchar(255),");
	query.append("chk       varchar(255),");
	query.append("dccode       varchar(255),");
	query.append("locus       varchar(255),");
	query.append("square       varchar(255),");
	query.append("date       varchar(255),");
	query.append("area       varchar(255),");
	query.append("publication      varchar(255),");
	query.append("heightinstrument      varchar(255),");
	query.append("poleheight       varchar(255),");
	query.append("bone       varchar(255),");
	query.append("flint       varchar(255),");
	query.append("pottery       varchar(255),");
	query.append("rc       varchar(255),");
	query.append("soilsample       varchar(255),");
	query.append("crate       varchar(255),");
	query.append("location       varchar(255)");
    query.append(");");

    res = queryPG(query,"");

    cout << query << "\n";

    //.....................................
    //This next query adds the geometry column (PostGIS will also insert this table as an entry
    //into the "geometry_columns" table.

    query = "SELECT AddGeometryColumn('public', '";
    query.append(tableS);
    query.append("', 'the_geom', 32636, 'POINT', 4)");

    res = queryPG(query,"");

    cout << query << "\n";

//......................................
//These next three queries remove checks on the table just created that would prevent easy data entry

    query = "ALTER TABLE ";
    query.append(tableS);
    query.append(" DROP CONSTRAINT enforce_srid_the_geom");

    res = queryPG(query,"");

    cout << query << "\n";

    query = "ALTER TABLE ";
    query.append(tableS);
    query.append(" DROP CONSTRAINT enforce_dims_the_geom");

    res = queryPG(query,"");

    cout << query << "\n";

    query = "ALTER TABLE ";
    query.append(tableS);
    query.append(" DROP CONSTRAINT enforce_geotype_the_geom");

    res = queryPG(query,"");

    cout << query << "\n";



    }
    return rows;
}
int newPolygonTable(string tableL)
{
//Creates a new polygon table for ArchField

PGresult        *res;
string query;
//string tableS;
//string tableL;
int rows;


//......................................................
//Check if table already exists!
query = "SELECT true FROM pg_tables WHERE tablename = '";
query.append(tableL);
query.append("';");

res = queryPG(query,"");
rows = PQntuples(res);


//rows = 0;
    if(rows == 0) //If table does not exist create both tables and insert starter data
    {
        //Create Polygon Table--Everything is the same as above except that polygons are handled slightly different

    query = "CREATE TABLE ";
    query.append(tableL);
    query.append(" (");
    query.append("gid       int4 PRIMARY KEY,");
	query.append("basket        varchar(255),");
	query.append("chk       varchar(255),");
	query.append("dccode       varchar(255),");
	query.append("locusdesc       varchar(255),");
	query.append("locus       varchar(255),");
	query.append("square       varchar(255),");
	query.append("date       varchar(255),");
	query.append("area       varchar(255),");
	query.append("structure       varchar(255),");
	query.append("publication      varchar(255),");
	query.append("heightinstrument      varchar(255),");
	query.append("poleheight       varchar(255)");
	query.append(");");

    res = queryPG(query,"");

    cout << query << "\n";

    //Inserting the_geom column--note that the main difference is the polygon entry instead of Point
    query = "SELECT AddGeometryColumn('public', '";
    query.append(tableL);
    query.append("', 'the_geom', 32636, 'POLYGON', 4)");

    res = queryPG(query,"");

    cout << query << "\n";

    //.........................................................
    //Remove Checks
    query = "ALTER TABLE ";
    query.append(tableL);
    query.append(" DROP CONSTRAINT enforce_srid_the_geom");

    res = queryPG(query,"");

    cout << query << "\n";

    query = "ALTER TABLE ";
    query.append(tableL);
    query.append(" DROP CONSTRAINT enforce_dims_the_geom");

    res = queryPG(query,"");

    cout << query << "\n";

    query = "ALTER TABLE ";
    query.append(tableL);
    query.append(" DROP CONSTRAINT enforce_geotype_the_geom");

    res = queryPG(query,"");

    cout << query << "\n";


    }
return rows;
}
void runDemo()
{
    //This function runs a series of queries to demonstrate how to interact with PostGIS in C++
    //It can be used to make sure everything is working with your install of PostGIS

    PGresult        *res;
 int             rec_count;
 int             row;
 int             col;
 string          query;
 string            list;
 string             type;
 string             srid;
 string             table;




//.............................................
//Create New Table and insert new data
table = "con01";
newtablePG(table);


//...............................................
//A Basic query using queryPG()
table = "con01sf";
query = ("SELECT Basket,Chk,DCCODE,Locus,Square,Date,Area,Publication,bone,flint,pottery,rc,soilsample,location,crate FROM ");
//query.append(", ST_AsKML(");
query.append(table);
query.append(" WHERE Date = '072210'");


res = queryPG(query,"");

         if (PQresultStatus(res) != PGRES_TUPLES_OK) {
                 puts("We did not get any data!");
                exit(0);
         }

         rec_count = PQntuples(res);

         printf("We received %d records.\n", rec_count);
         puts("==========================");

        for (row=0; row<rec_count; row++) {
                 for (col=0; col<3; col++) {
                         printf("%s\t", PQgetvalue(res, row, col));
                 }
                 puts("");
        }

         puts("==========================");

         PQclear(res);


//...............................................................................
//Test Points

cout << "\n" << "============Points===============" << "\n";

table = "con01sf";
query = "SELECT Basket,Chk,DCCODE,Locus,Square,Date,Area,Publication,bone,flint,pottery,rc,soilsample,location,crate FROM ";
query.append(table);
query.append(" WHERE basket = '50111'");
list = "Basket,Chk,DCCODE,Locus,Square,Date,Area,Publication,bone,flint,pottery,rc,soilsample,location,crate,the_geom";
type = "Point";
srid = "32636";

queryPostGIS(query, list, type, srid);

//..................................................................
//Test Polygons
cout << "\n" << "============Polygons===============" << "\n";
table = "con01l";
query = "SELECT Basket,dccode,chk,date FROM ";
query.append(table);
query.append(" WHERE Basket = '55111'");
list = "Basket,dccode,chk,date,the_geom";
type = "Polygon";
srid = "32636";


queryPostGIS(query, list, type, srid);

//..............................................................................
//Test KML-P
cout << "\n" << "============KML Polygons===============" << "\n";
table = "con01l";
query = "SELECT Basket,dccode,chk,date FROM ";
query.append(table);
query.append(" WHERE Basket = '55111'");
list = "Basket,dccode,chk,date,the_geom";
type = "KML-P";
srid = "4326";


queryPostGIS(query, list, type, srid);
cout << "===========================" << "\n";

string eraseTbls;

cout << "Would you like to drop the newly created tables? (y,n)";
cin >> eraseTbls;
cin.ignore();

if (eraseTbls == "y")
{
    table = "con01sf";
    dropTable(table);

    table = "con01l";
    dropTable(table);

}
cout << "===========================" << "\n";


         //cin.get();

}
void importDemos()
{
    //................................................................
//This section will allow you to import a point table in the database. It is set to kis10sf.csv


string  check;

cout << "Do you want to import a point csv into the database? (y,n): ";
cin >> check;
cin.ignore();

if (check == "y")
{
    string sfile;
    string table;
    sfile = "kis10sf.csv";

    cout << "Please enter the name of the table ending with 'sf' (e.g. kis10sf): ";
    cin >> table;
    cin.ignore();

    if (table == "")
    {
    table = "kis1001sf";
    }

    //importCSV(sfile,table);
    importCSV(sfile, table, "", "|");
}

//................................................................
//This section will allow you to import a polygon table in the database. It is set to kis10l.csv

cout << "Do you want to import a polygon csv into the database? (y,n): ";
cin >> check;
cin.ignore();

if (check == "y")
{
    string sfile;
    string table;
    sfile = "kis10l.csv";

    cout << "Please enter the name of the table ending with 'l' (e.g. kis10l): ";
    cin >> table;
    cin.ignore();

    if (table == "")
    {
    table = "kis1001l";
    }

    //importCSV(sfile,table);
    importCSV(sfile, table, "", "|");
}



}
