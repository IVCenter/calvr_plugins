#include "vtk2bin.h"

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <climits>

#ifdef WITH_FX_LIB
#define UNDERSCORE
#include <FX.h>
#endif

#ifdef WIN32
#define snprintf _snprintf_s
#endif

#define BINARY_FILE_VERSION 1

void printUsage(std::string name)
{
    std::cerr << "Usage: " << name << " [options] input" << std::endl;
    std::cerr << "Input is a format string, i.e. state.%03d.vtk" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -s <int>" << std::endl;
    std::cerr << "               Set the starting frame number for the input set, default: 0" << std::endl;
    std::cerr << "  -f <int>" << std::endl;
    std::cerr << "               Set the number of frames to process, default: 0 (all, step until file not found)" << std::endl;
    std::cerr << "  -o <string>" << std::endl;
    std::cerr << "               Base name for output files" << std::endl;
}

bool fileExists(std::string file)
{
    std::ifstream testFile(file.c_str());
    if(testFile.good())
    {
	testFile.close();
	return true;
    }
    testFile.close();
    return false;
}

int main(int argc, char ** argv)
{
    std::string outputFile;
    std::string inputFile;
    int start = 0, frames = 0;

    for(int i = 1; i < argc; ++i)
    {
	if(std::string(argv[i]) == "-s" && (i + 1) < argc)
	{
	    i++;
	    start = atoi(argv[i]);
	}
	else if(std::string(argv[i]) == "-f" && (i + 1) < argc)
	{
	    i++;
	    frames = atoi(argv[i]);
	}
	else if(std::string(argv[i]) == "-o" && (i + 1) < argc)
	{
	    i++;
	    outputFile = argv[i];
	}
	else if(std::string(argv[i]) == "-h" || std::string(argv[i]) == "--help")
	{
	    printUsage(argv[0]);
	    return 1;
	}
	else
	{
	    inputFile = argv[i];
	}
    }

    std::cerr << "Input: " << inputFile << std::endl;
    std::cerr << "Output Root: " << outputFile << std::endl;
    std::cerr << "Start: " << start << std::endl;
    std::cerr << "Frames: " << frames;
    if(frames == 0)
    {
	std::cerr << " (all)";
    }
    std::cerr << std::endl;

    if(inputFile.empty() || outputFile.empty() || frames < 0)
    {
	printUsage(argv[0]);
	return 1;
    }

    char currentFile[1024];
    int currentFrame = start;
    snprintf(currentFile,1023,inputFile.c_str(),currentFrame);

    FlowDataSet * set = new FlowDataSet;

    while(fileExists(currentFile))
    {
	if(frames && (currentFrame - start) >= frames)
	{
	    break;
	}

	std::cerr << "Loading file: " << currentFile << std::endl;

	if(!parseVTK(currentFile,set))
	{
	    std::cerr << "Error parsing file." << std::endl;
	}
	else
	{
	    if(set->frameList.size())
	    {
		processFrameWithFX(set,set->frameList.size()-1);
		writeBinaryFrameFile(set,outputFile,set->frameList.size()-1);
	    }
	}

	currentFrame++;
	snprintf(currentFile,1023,inputFile.c_str(),currentFrame);
    }

    if(set->frameList.size())
    {
	//processWithFX(set);

	for(int i = 0; i < set->frameList.size(); ++i)
	{
	    for(int j = 0; j < set->frameList[i]->pointData.size(); ++j)
	    {
		if(set->frameList[i]->pointData[j]->dataType == VDT_DOUBLE)
		{
		    std::map<std::string,std::pair<float,float> >::iterator it = set->attribRanges.find(set->frameList[i]->pointData[j]->name);
		    if(it == set->attribRanges.end())
		    {
			set->attribRanges[set->frameList[i]->pointData[j]->name] = std::pair<float,float>(FLT_MAX,FLT_MIN);
			it = set->attribRanges.find(set->frameList[i]->pointData[j]->name);
		    }

		    if(set->frameList[i]->pointData[j]->floatMin < it->second.first)
		    {
			it->second.first = set->frameList[i]->pointData[j]->floatMin;
		    }
		    if(set->frameList[i]->pointData[j]->floatMax > it->second.second)
		    {
			it->second.second = set->frameList[i]->pointData[j]->floatMax;
		    }
		}
	    }

	    set->bb.expandBy(set->frameList[i]->bb);
	}

	//writeBinaryFiles(set,outputFile);
	writeBinaryMetaFile(set,outputFile);
    }

    return 1;
}

bool parseVTK(std::string file, FlowDataSet * set)
{
    bool status = true;
    char buffer[1024];
    FILE * fileptr = fopen(file.c_str(),"r");

    int points;
    int numIndices;

    VTKDataFrame * frame = new VTKDataFrame;

    if(fileptr)
    {
	while(!feof(fileptr))
	{
	    int count = 0;
	    int ret = fscanf(fileptr,"#%n",&count);

	    while(!feof(fileptr) && count)
	    {
		//std::cerr << "got comment" << std::endl;
		char c = '\0';
		while(!feof(fileptr) && c != '\n')
		{
		    fread(&c,sizeof(char),1,fileptr);
		}
		count = 0;
		ret = fscanf(fileptr,"#%n",&count);
	    }

	    int read = fscanf(fileptr,"%s",buffer);
	    if(read <= 0)
	    {
		break;
	    }

	    if(!strcmp(buffer,"LS-DYNA"))
	    {
		//std::cerr << "LS-DYNA" << std::endl;
		fscanf(fileptr,"%*s %*s");
	    }
	    else if(!strcmp(buffer,"ASCII"))
	    {
		//std::cerr << "ASCII" << std::endl;
	    }
	    else if(!strcmp(buffer,"DATASET"))
	    {
		char type[1024];
		fscanf(fileptr,"%s",type);
		if(strcmp(type,"UNSTRUCTURED_GRID"))
		{
		    std::cerr << "Unknown dataset type: " << type << std::endl;
		    status = false;
		    deleteVTKFrame(frame);
		    break;
		}
	    }
	    else if(!strcmp(buffer,"POINTS"))
	    {
		fscanf(fileptr,"%d %*s",&points);
		//std::cerr << "Points: " << points << std::endl;
		frame->verts = new osg::Vec3Array(points+1);
		for(int i = 1; i <= points; ++i)
		{
		    fscanf(fileptr,"%f %f %f",&frame->verts->at(i).x(),&frame->verts->at(i).y(),&frame->verts->at(i).z());
		    frame->bb.expandBy(frame->verts->at(i));
		}
		std::cerr << "Bounds: min x: " << frame->bb.xMin() << " y: " << frame->bb.yMin() << " z: " << frame->bb.zMin() << std::endl;
		std::cerr << "Bounds: max x: " << frame->bb.xMax() << " y: " << frame->bb.yMax() << " z: " << frame->bb.zMax() << std::endl;
	    }
	    else if(!strcmp(buffer,"CELLS"))
	    {
		int dataCount = 0;
		fscanf(fileptr,"%d %d",&numIndices,&dataCount);
		frame->indices = new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS,numIndices*4);
		frame->indices->resize(numIndices*4);

		for(int i = 0; i < numIndices; ++i)
		{
		    int icount = 0;
		    fscanf(fileptr,"%d",&icount);
		    if(icount != 4)
		    {
			std::cerr << "Non quad found." << std::endl;
			break;
		    }
		    int i1,i2,i3,i4;
		    fscanf(fileptr,"%d %d %d %d",&i1, &i2, &i3, &i4);
		    frame->indices->at((i*4)+0) = i1+1;
		    frame->indices->at((i*4)+1) = i2+1;
		    frame->indices->at((i*4)+2) = i3+1;
		    frame->indices->at((i*4)+3) = i4+1;
		}

		numIndices *= 4;
	    }
	    else if(!strncmp(buffer,"CELL_TYPES",10))
	    {
		int dataCount = 0;

		// fix for turbine files
		if(strlen(buffer) > 10)
		{
		    dataCount = atoi(buffer+10);
		}
		else
		{
		    fscanf(fileptr,"%d",&dataCount);
		}

		frame->cellTypes = new osg::IntArray(dataCount);
		for(int i = 0; i < dataCount; ++i)
		{
		    fscanf(fileptr,"%d",&frame->cellTypes->at(i));
		}
	    }
	    else if(!strncmp(buffer,"CELL_DATA",9))
	    {
		int dataCount = 0;

		// fix for turbine files
		if(strlen(buffer) > 9)
		{
		    dataCount = atoi(buffer+9);
		}
		else
		{
		    fscanf(fileptr,"%d",&dataCount);
		}
		while(!feof(fileptr))
		{
		    long int fpos = ftell(fileptr);
		    char dataType[1024];
		    fscanf(fileptr,"%s",dataType);
		    if(!strcmp(dataType,"SCALARS") || !strcmp(dataType,"VECTORS"))
		    {
			VTKDataAttrib * attrib = parseVTKAttrib(fileptr,dataType,dataCount);
			if(attrib)
			{
			    frame->cellData.push_back(attrib);
			}
			else
			{
			    status = false;
			    break;
			}
		    }
		    else
		    {
			fseek(fileptr,fpos,SEEK_SET);
			break;
		    }
		}
		if(!status)
		{
		    deleteVTKFrame(frame);
		    break;
		}
	    }
	    else if(!strcmp(buffer,"POINT_DATA"))
	    {
		int dataCount = 0;
		fscanf(fileptr,"%d",&dataCount);
		while(!feof(fileptr))
		{
		    long int fpos = ftell(fileptr);
		    char dataType[1024];
		    fscanf(fileptr,"%s",dataType);
		    if(feof(fileptr))
		    {
			break;
		    }
		    if(!strcmp(dataType,"SCALARS") || !strcmp(dataType,"VECTORS"))
		    {
			VTKDataAttrib * attrib = parseVTKAttrib(fileptr,dataType,dataCount);
			if(attrib)
			{
			    frame->pointData.push_back(attrib);
			}
			else
			{
			    status = false;
			    break;
			}
		    }
		    else
		    {
			fseek(fileptr,fpos,SEEK_SET);
			break;
		    }
		}
		if(!status)
		{
		    deleteVTKFrame(frame);
		    break;
		}
	    }
	    else
	    {
		std::cerr << "Unknown section: " << buffer << std::endl;
		status = false;
		deleteVTKFrame(frame);
		break;
	    }
	}
	fclose(fileptr);

	extractSurfaceVTK(frame);
	set->frameList.push_back(frame);
    }
    else
    {
	std::cerr << "Unable to open file." << std::endl;
	status = false;
	deleteVTKFrame(frame);
    }

    return status;
}

VTKDataAttrib * parseVTKAttrib(FILE * file, std::string type, int count)
{
    VTKDataAttrib * attrib = new VTKDataAttrib;

    if(type == "SCALARS")
    {
	attrib->attribType = VAT_SCALARS;
	char name[1024];
	char valType[1024];
	fscanf(file,"%s %s",name,valType);
	attrib->name = name;
	
	std::string svalType(valType);

	// clear lookup table entry
	fscanf(file,"%*s %*s");

	if(svalType == "int")
	{
	    attrib->dataType = VDT_INT;
	    attrib->intData = new osg::IntArray(count+1);
	    attrib->intMin = INT_MAX;
	    attrib->intMax = INT_MIN;
	    for(int i = 1; i <= count; ++i)
	    {
		fscanf(file,"%d",&attrib->intData->at(i));
		if(attrib->intData->at(i) < attrib->intMin)
		{
		    attrib->intMin = attrib->intData->at(i);
		}
		if(attrib->intData->at(i) > attrib->intMax)
		{
		    attrib->intMax = attrib->intData->at(i);
		}
		//std::cerr << "attrib: " << attrib->intData->at(i) << std::endl;
		/*if(attrib->intData->at(i) != -1)
		{
		    attrib->intData->at(i) = 8;
		}*/
	    }
	    //std::cerr << "Int name: " << attrib->name <<  " min: " << attrib->intMin << " max: " << attrib->intMax << std::endl;
	}
	else if(svalType == "double")
	{
	    attrib->dataType = VDT_DOUBLE;
	    attrib->floatData = new osg::FloatArray(count+1);
	    attrib->floatMin = FLT_MAX;
	    attrib->floatMax = FLT_MIN;
	    for(int i = 1; i <= count; ++i)
	    {
		fscanf(file,"%f",&attrib->floatData->at(i));
		if(attrib->floatData->at(i) < attrib->floatMin)
		{
		    attrib->floatMin = attrib->floatData->at(i);
		}
		if(attrib->floatData->at(i) > attrib->floatMax)
		{
		    attrib->floatMax = attrib->floatData->at(i);
		}
	    }
	}
	else
	{
	    std::cerr << "Unknown attribute value type: " << svalType << std::endl;
	    delete attrib;
	    return NULL;
	}
    }
    else if(type == "VECTORS")
    {
	attrib->attribType = VAT_VECTORS;
	char name[1024];
	char valType[1024];
	fscanf(file,"%s %s",name,valType);
	attrib->name = name;
	
	std::string svalType(valType);
	if(svalType == "double")
	{
	    attrib->dataType = VDT_DOUBLE;
	    attrib->floatMin = FLT_MAX;
	    attrib->floatMax = FLT_MIN;
	    attrib->vecData = new osg::Vec3Array(count+1);
	    for(int i = 1; i <= count; ++i)
	    {
		fscanf(file,"%f %f %f",&attrib->vecData->at(i).x(),&attrib->vecData->at(i).y(),&attrib->vecData->at(i).z());
		float mag = attrib->vecData->at(i).length();
		if(mag > attrib->floatMax)
		{
		    attrib->floatMax = mag;
		}
		if(mag < attrib->floatMin)
		{
		    attrib->floatMin = mag;
		}
	    }
	}
	else
	{
	    std::cerr << "Unknown vector attribute value type: " << svalType << std::endl;
	    delete attrib;
	    return NULL;
	}
    }
    else
    {
	std::cerr << "Error: Unknown Attribute type: " << type << std::endl;
	delete attrib;
	return NULL;
    }

    return attrib;
}

struct face
{
    face()
    {
	indices[0] = indices[1] = indices[2] = 0.0;
    }
    face(unsigned int a, unsigned int b, unsigned int c)
    {
	indices[0] = a;
	indices[1] = b;
	indices[2] = c;
    }
    unsigned int indices[3];
    int cell;
};

void extractSurfaceVTK(VTKDataFrame * frame)
{
    std::cerr << "extracting surface" << std::endl;
    frame->surfaceInd = new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES,0);

    std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, std::pair<face,int> > > > faces;
    for(int i = 0; i < frame->indices->size(); i += 4)
    {
	std::vector<face> faceList;
	faceList.push_back(face(frame->indices->at(i+0),frame->indices->at(i+1),frame->indices->at(i+3)));
	faceList.push_back(face(frame->indices->at(i+0),frame->indices->at(i+2),frame->indices->at(i+1)));
	faceList.push_back(face(frame->indices->at(i+0),frame->indices->at(i+3),frame->indices->at(i+2)));
	faceList.push_back(face(frame->indices->at(i+1),frame->indices->at(i+2),frame->indices->at(i+3)));

	for(int j = 0; j < faceList.size(); ++j)
	{
	    faceList[j].cell = (i / 4) + 1;
	    unsigned int first,second,third;
	    if(faceList[j].indices[0] >= faceList[j].indices[1])
	    {
		if(faceList[j].indices[0] >= faceList[j].indices[2])
		{
		    first = faceList[j].indices[0];
		    if(faceList[j].indices[1] >= faceList[j].indices[2])
		    {
			second = faceList[j].indices[1];
			third = faceList[j].indices[2];
		    }
		    else
		    {
			second = faceList[j].indices[2];
			third = faceList[j].indices[1];
		    }
		}
		else
		{
		    first = faceList[j].indices[2];
		    second = faceList[j].indices[0];
		    third = faceList[j].indices[1];
		}
	    }
	    else if(faceList[j].indices[0] >= faceList[j].indices[2])
	    {
		first = faceList[j].indices[1];
		second = faceList[j].indices[0];
		third = faceList[j].indices[2];
	    }
	    else
	    {
		third = faceList[j].indices[0];
		if(faceList[j].indices[1] >= faceList[j].indices[2])
		{
		    first = faceList[j].indices[1];
		    second = faceList[j].indices[2];
		}
		else
		{
		    first = faceList[j].indices[2];
		    second = faceList[j].indices[1];
		}
	    }

	    faces[first][second][third].first = faceList[j];
	    faces[first][second][third].second++;
	}
    }

    frame->surfaceFacets = new osg::Vec4iArray();
    frame->surfaceCells = new osg::IntArray();

    for(std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, std::pair<face,int> > > >::iterator it = faces.begin(); it != faces.end(); ++it)
    {
	for(std::map<unsigned int, std::map<unsigned int, std::pair<face,int> > >::iterator itt = it->second.begin(); itt != it->second.end(); ++itt)
	{
	    for(std::map<unsigned int, std::pair<face,int> >::iterator ittt = itt->second.begin(); ittt != itt->second.end(); ++ittt)
	    {
		if(ittt->second.second == 1)
		{
		    frame->surfaceInd->push_back(ittt->second.first.indices[0]);
		    frame->surfaceInd->push_back(ittt->second.first.indices[1]);
		    frame->surfaceInd->push_back(ittt->second.first.indices[2]);
		    frame->surfaceFacets->push_back(osg::Vec4i(ittt->second.first.indices[0],ittt->second.first.indices[1],ittt->second.first.indices[2],0));
		    frame->surfaceCells->push_back(ittt->second.first.cell);
		}
	    }
	}
    }

    std::cerr << "done: got " << frame->surfaceInd->size() << " indices." << std::endl;
}

void deleteVTKFrame(VTKDataFrame * frame)
{
    if(!frame)
    {
	return;
    }

    for(int j = 0; j < frame->cellData.size(); ++j)
    {
	delete frame->cellData[j];
    }

    for(int j = 0; j < frame->pointData.size(); ++j)
    {
	delete frame->pointData[j];
    }

    delete frame;
}

FlowDataSet * fxSet = NULL;
int fxFrame = 0;

void processWithFX(FlowDataSet * set)
{
#ifdef WITH_FX_LIB
    std::cerr << "Processing set with FX library." << std::endl;

    if(set)
    {
	for(int i = 0; i < set->frameList.size(); ++i)
	{
	    // init fx library

	    // assuming dry air at 20c for the moment
	    float gamma = 1.4;
	    int iopt = 0;
	    int knode = set->frameList[i]->verts->size()-1;
	    int nhalo = 0, npyra = 0, nprism = 0, nhexa = 0, nblock = 0;
	    int ntets = set->frameList[i]->indices->size() / 4;
	    int * blocks = NULL;
	    int nhcell = 0;
	    int nfacet = set->frameList[i]->surfaceCells->size();
	    int nbc = 1;
	    int flag = 1 + 2 + 4 + 8;

	    fxSet = set;
	    fxFrame = i;

	    FX_INIT(&gamma,&iopt,&knode,&nhalo,&ntets,&npyra,&nprism,&nhexa,&nblock,blocks,&nhcell,&nfacet,&nbc,&flag);
	    std::cerr << "FX_Init return: " << flag << std::endl;

	    int type = 0;
	    int numSeg;
	    int * segEnds;
	    float * segPoints;
	    float * coreStrength;

	    std::cerr << "Finding vortex cores..." << std::endl;
	    FX_VORTEXCORE(&type,&numSeg,&segEnds,&segPoints,&coreStrength);
	    std::cerr << "Got " << numSeg << " segments" << std::endl;
	    if(numSeg)
	    {
		VortexCoreData * vcore = new VortexCoreData;
		set->frameList[i]->vcoreData = vcore;
		
		vcore->min = FLT_MAX;
		vcore->max = FLT_MIN;

		vcore->verts = new osg::Vec3Array(segEnds[numSeg-1]);
		vcore->coreStr = new osg::FloatArray(segEnds[numSeg-1]);
		for(int j = 0; j < segEnds[numSeg-1]; ++j)
		{
		    vcore->verts->at(j) = osg::Vec3(segPoints[(j*3)+0],segPoints[(j*3)+1],segPoints[(j*3)+2]);
		    vcore->coreStr->at(j) = coreStrength[j];
		    if(coreStrength[j] < vcore->min)
		    {
			vcore->min = coreStrength[j];
		    }
		    if(coreStrength[j] > vcore->max)
		    {
			vcore->max = coreStrength[j];
		    }
		}

		int start = 0;
		for(int j = 0; j < numSeg; ++j)
		{
		    vcore->coreSegments.push_back(new osg::DrawArrays(GL_LINE_STRIP,start,segEnds[j]-start));
		    start = segEnds[j];
		}

		free(segEnds);
		free(segPoints);
		free(coreStrength);
	    }
	    else
	    {
		set->frameList[i]->vcoreData = NULL;
	    }

	    if(nbc)
	    {
		int sEnds[nbc];
		float * sPoints;
		int aEnds[nbc];
		float * aPoints;

		std::cerr << "Finding Sep/Att lines..." << std::endl;
		FX_SEPNLINE(sEnds,&sPoints,aEnds,&aPoints);

		if(sEnds[nbc-1] && aEnds[nbc-1])
		{
		    SepAttLineData * saData = new SepAttLineData;
		    set->frameList[i]->sepAttData = saData;
		    //std::cerr << "sEnds: " << sEnds[nbc-1] << " aEnds: " << aEnds[nbc-1] << std::endl;
		    saData->sverts = new osg::Vec3Array(sEnds[nbc-1]);
		    saData->averts = new osg::Vec3Array(aEnds[nbc-1]);

		    memcpy(&saData->sverts->at(0),sPoints,sEnds[nbc-1]*3*sizeof(float));
		    memcpy(&saData->averts->at(0),aPoints,aEnds[nbc-1]*3*sizeof(float));

		    int sStart = 0;
		    int aStart = 0;
		    for(int j = 0; j < nbc; ++j)
		    {
			saData->sSegments.push_back(new osg::DrawArrays(GL_LINES,sStart,sEnds[j]-sStart));
			saData->aSegments.push_back(new osg::DrawArrays(GL_LINES,aStart,aEnds[j]-aStart));
			sStart = sEnds[j];
			aStart = aEnds[j];
		    }

		    free(sPoints);
		    free(aPoints);
		}
		else
		{
		    set->frameList[i]->sepAttData = NULL;
		}
	    }
	    else
	    {
		set->frameList[i]->sepAttData = NULL;
	    }

	    std::cerr << "Getting shock info..." << std::endl;
	    float * shock;
	    FX_SHOCKFIND(&shock);

	    if(shock)
	    {
		VTKDataAttrib * attr = new VTKDataAttrib;
		attr->name = "Shock";
		attr->attribType = VAT_SCALARS;
		attr->dataType = VDT_DOUBLE;
		attr->floatMin = FLT_MAX;
		attr->floatMax = FLT_MIN;

		attr->floatData = new osg::FloatArray(set->frameList[i]->verts->size());
		for(int j = 1; j < attr->floatData->size(); ++j)
		{
		    attr->floatData->at(j) = shock[j-1];
		    if(shock[j-1] < attr->floatMin)
		    {
			attr->floatMin = shock[j-1];
		    }
		    if(shock[j-1] > attr->floatMax)
		    {
			attr->floatMax = shock[j-1];
		    }
		}

		set->frameList[i]->pointData.push_back(attr);

		free(shock);
	    }

	    FX_CLOSE();
	}

	float vcoreMin = FLT_MAX;
	float vcoreMax = FLT_MIN;
	for(int i = 0; i < set->frameList.size(); ++i)
	{
	    if(!set->frameList[i]->vcoreData)
	    {
		continue;
	    }
	    if(set->frameList[i]->vcoreData->min < vcoreMin)
	    {
		vcoreMin = set->frameList[i]->vcoreData->min;
	    }
	    if(set->frameList[i]->vcoreData->max > vcoreMax)
	    {
		vcoreMax = set->frameList[i]->vcoreData->max;
	    }
	}
	set->attribRanges["Vortex Cores"] = std::pair<float,float>(vcoreMin,vcoreMax);

	float min = FLT_MAX;
	float max = FLT_MIN;
	for(int i = 0; i < set->frameList.size(); ++i)
	{
	    for(int j = 0; j < set->frameList[i]->pointData.size(); ++j)
	    {
		if(set->frameList[i]->pointData[j]->name != "Shock")
		{
		    continue;
		}
		if(set->frameList[i]->pointData[j]->floatMin < min)
		{
		    min = set->frameList[i]->pointData[j]->floatMin;
		}
		if(set->frameList[i]->pointData[j]->floatMax < max)
		{
		    max = set->frameList[i]->pointData[j]->floatMax;
		}
	    }
	}
	set->attribRanges["Shock"] = std::pair<float,float>(min,max);
    }
    else
    {
	fxSet = NULL;
    }
    
#else
    if(set)
    {
	for(int i = 0; i < set->frameList.size(); ++i)
	{
	    set->frameList[i]->vcoreData = NULL;
	    set->frameList[i]->sepAttData = NULL;
	}
    }
    std::cerr << "Not built with FX library." << std::endl;
#endif
}

void processFrameWithFX(FlowDataSet * set, int frame)
{
#ifdef WITH_FX_LIB
    std::cerr << "Processing set with FX library." << std::endl;

    if(set)
    {
	// init fx library

	// assuming dry air at 20c for the moment
	float gamma = 1.4;
	int iopt = 0;
	int knode = set->frameList[frame]->verts->size()-1;
	int nhalo = 0, npyra = 0, nprism = 0, nhexa = 0, nblock = 0;
	int ntets = set->frameList[frame]->indices->size() / 4;
	int * blocks = NULL;
	int nhcell = 0;
	int nfacet = set->frameList[frame]->surfaceCells->size();
	int nbc = 1;
	int flag = 1 + 2 + 4 + 8;

	fxSet = set;
	fxFrame = frame;

	FX_INIT(&gamma,&iopt,&knode,&nhalo,&ntets,&npyra,&nprism,&nhexa,&nblock,blocks,&nhcell,&nfacet,&nbc,&flag);
	std::cerr << "FX_Init return: " << flag << std::endl;

	int type = 0;
	int numSeg;
	int * segEnds;
	float * segPoints;
	float * coreStrength;

	std::cerr << "Finding vortex cores..." << std::endl;
	FX_VORTEXCORE(&type,&numSeg,&segEnds,&segPoints,&coreStrength);
	std::cerr << "Got " << numSeg << " segments" << std::endl;
	if(numSeg)
	{
	    VortexCoreData * vcore = new VortexCoreData;
	    set->frameList[frame]->vcoreData = vcore;

	    vcore->min = FLT_MAX;
	    vcore->max = FLT_MIN;

	    vcore->verts = new osg::Vec3Array(segEnds[numSeg-1]);
	    vcore->coreStr = new osg::FloatArray(segEnds[numSeg-1]);
	    for(int j = 0; j < segEnds[numSeg-1]; ++j)
	    {
		vcore->verts->at(j) = osg::Vec3(segPoints[(j*3)+0],segPoints[(j*3)+1],segPoints[(j*3)+2]);
		vcore->coreStr->at(j) = coreStrength[j];
		if(coreStrength[j] < vcore->min)
		{
		    vcore->min = coreStrength[j];
		}
		if(coreStrength[j] > vcore->max)
		{
		    vcore->max = coreStrength[j];
		}
	    }

	    int start = 0;
	    for(int j = 0; j < numSeg; ++j)
	    {
		vcore->coreSegments.push_back(new osg::DrawArrays(GL_LINE_STRIP,start,segEnds[j]-start));
		start = segEnds[j];
	    }

	    free(segEnds);
	    free(segPoints);
	    free(coreStrength);
	}
	else
	{
	    set->frameList[frame]->vcoreData = NULL;
	}

	if(nbc)
	{
	    int sEnds[nbc];
	    float * sPoints;
	    int aEnds[nbc];
	    float * aPoints;

	    std::cerr << "Finding Sep/Att lines..." << std::endl;
	    FX_SEPNLINE(sEnds,&sPoints,aEnds,&aPoints);

	    if(sEnds[nbc-1] && aEnds[nbc-1])
	    {
		SepAttLineData * saData = new SepAttLineData;
		set->frameList[frame]->sepAttData = saData;
		//std::cerr << "sEnds: " << sEnds[nbc-1] << " aEnds: " << aEnds[nbc-1] << std::endl;
		saData->sverts = new osg::Vec3Array(sEnds[nbc-1]);
		saData->averts = new osg::Vec3Array(aEnds[nbc-1]);

		memcpy(&saData->sverts->at(0),sPoints,sEnds[nbc-1]*3*sizeof(float));
		memcpy(&saData->averts->at(0),aPoints,aEnds[nbc-1]*3*sizeof(float));

		int sStart = 0;
		int aStart = 0;
		for(int j = 0; j < nbc; ++j)
		{
		    saData->sSegments.push_back(new osg::DrawArrays(GL_LINES,sStart,sEnds[j]-sStart));
		    saData->aSegments.push_back(new osg::DrawArrays(GL_LINES,aStart,aEnds[j]-aStart));
		    sStart = sEnds[j];
		    aStart = aEnds[j];
		}

		free(sPoints);
		free(aPoints);
	    }
	    else
	    {
		set->frameList[frame]->sepAttData = NULL;
	    }
	}
	else
	{
	    set->frameList[frame]->sepAttData = NULL;
	}

	std::cerr << "Getting shock info..." << std::endl;
	float * shock;
	FX_SHOCKFIND(&shock);

	if(shock)
	{
	    VTKDataAttrib * attr = new VTKDataAttrib;
	    attr->name = "Shock";
	    attr->attribType = VAT_SCALARS;
	    attr->dataType = VDT_DOUBLE;
	    attr->floatMin = FLT_MAX;
	    attr->floatMax = FLT_MIN;

	    attr->floatData = new osg::FloatArray(set->frameList[frame]->verts->size());
	    for(int j = 1; j < attr->floatData->size(); ++j)
	    {
		attr->floatData->at(j) = shock[j-1];
		if(shock[j-1] < attr->floatMin)
		{
		    attr->floatMin = shock[j-1];
		}
		if(shock[j-1] > attr->floatMax)
		{
		    attr->floatMax = shock[j-1];
		}
	    }

	    set->frameList[frame]->pointData.push_back(attr);

	    free(shock);
	}

	FX_CLOSE();

	if(set->frameList[frame]->vcoreData)
	{
	    if(set->attribRanges.find("Vortex Cores") == set->attribRanges.end())
	    {
		set->attribRanges["Vortex Cores"] = std::pair<float,float>(FLT_MAX,FLT_MIN);
	    }

	    if(set->frameList[frame]->vcoreData->min < set->attribRanges["Vortex Cores"].first)
	    {
		set->attribRanges["Vortex Cores"].first = set->frameList[frame]->vcoreData->min;
	    }
	    if(set->frameList[frame]->vcoreData->max > set->attribRanges["Vortex Cores"].second)
	    {
		set->attribRanges["Vortex Cores"].second = set->frameList[frame]->vcoreData->max;
	    }
	}

	/*float vcoreMin = FLT_MAX;
	float vcoreMax = FLT_MIN;
	for(int i = 0; i < set->frameList.size(); ++i)
	{
	    if(!set->frameList[i]->vcoreData)
	    {
		continue;
	    }
	    if(set->frameList[i]->vcoreData->min < vcoreMin)
	    {
		vcoreMin = set->frameList[i]->vcoreData->min;
	    }
	    if(set->frameList[i]->vcoreData->max > vcoreMax)
	    {
		vcoreMax = set->frameList[i]->vcoreData->max;
	    }
	}
	set->attribRanges["Vortex Cores"] = std::pair<float,float>(vcoreMin,vcoreMax);*/

	/*float min = FLT_MAX;
	float max = FLT_MIN;
	for(int i = 0; i < set->frameList.size(); ++i)
	{
	    for(int j = 0; j < set->frameList[i]->pointData.size(); ++j)
	    {
		if(set->frameList[i]->pointData[j]->name != "Shock")
		{
		    continue;
		}
		if(set->frameList[i]->pointData[j]->floatMin < min)
		{
		    min = set->frameList[i]->pointData[j]->floatMin;
		}
		if(set->frameList[i]->pointData[j]->floatMax < max)
		{
		    max = set->frameList[i]->pointData[j]->floatMax;
		}
	    }
	}
	set->attribRanges["Shock"] = std::pair<float,float>(min,max);*/
    }
    else
    {
	fxSet = NULL;
    }
    
#else
    if(set)
    {
	set->frameList[frame]->vcoreData = NULL;
	set->frameList[frame]->sepAttData = NULL;
    }
    std::cerr << "Not built with FX library." << std::endl;
#endif
}

#ifdef WITH_FX_LIB

void FXCELLPTR(int **tets, int **pyras, int **prisms, int **hexas, int *halo)
{
    std::cerr << "cell ptr" << std::endl;
    *pyras = NULL;
    *prisms = NULL;
    *hexas = NULL;
    
     if(fxSet)
     {
	 *tets = (int*)&fxSet->frameList[fxFrame]->indices->at(0);
     }
}

void FXGRIDPTR(float **xyz, float *hxyz)
{
    std::cerr << "grid ptr" << std::endl;
    if(fxSet)
    {
	// one biased arrays
	*xyz = (float*)&fxSet->frameList[fxFrame]->verts->at(1);
    }
}

void FXSURFACEPTR(int *nsurf, int **cell, int **facet)
{
    std::cerr << "surface ptr" << std::endl;
  if(!fxSet)
  {
      return;
  }
  // hmm, is this one biased too? maybe
  nsurf[0] = fxSet->frameList[fxFrame]->surfaceCells->size();
  nsurf[1] = 6;

  *cell = &fxSet->frameList[fxFrame]->surfaceCells->at(0);
  *facet = (int*)&fxSet->frameList[fxFrame]->surfaceFacets->at(0);
}

void FXSCAL(int *type, float *s, float *hs)
{
    std::cerr << "scal type: " << *type << std::endl;

    if(*type == 2)
    {
	// pressure
	for(int i = 0; i < fxSet->frameList[fxFrame]->pointData.size(); ++i)
	{
	    if(fxSet->frameList[fxFrame]->pointData[i]->name == "Pressure")
	    {
		memcpy(s,&fxSet->frameList[fxFrame]->pointData[i]->floatData->at(1),(fxSet->frameList[fxFrame]->pointData[i]->floatData->size()-1)*sizeof(float));
		break;
	    }
	}
    }
    else if(*type == 3)
    {
	// mach number
	for(int i = 0; i < fxSet->frameList[fxFrame]->pointData.size(); ++i)
	{
	    if(fxSet->frameList[fxFrame]->pointData[i]->name == "Velocity")
	    {
		for(int j = 1; j < fxSet->frameList[fxFrame]->pointData[i]->vecData->size(); ++j)
		{
		    s[j-1] = fxSet->frameList[fxFrame]->pointData[i]->vecData->at(j).length() / 343.6;
		}
		break;
	    }
	}
    }
}

void FXVELPTR(float **vel, float *hvel)
{
    std::cerr << "vel ptr" << std::endl;
    if(!fxSet)
    {
	return;
    }

    for(int i = 0; i < fxSet->frameList[fxFrame]->pointData.size(); ++i)
    {
	if(fxSet->frameList[fxFrame]->pointData[i]->name == "Velocity" && fxSet->frameList[fxFrame]->pointData[i]->attribType == VAT_VECTORS && fxSet->frameList[fxFrame]->pointData[i]->dataType == VDT_DOUBLE)

	{
	    //std::cerr << "Found vel" << std::endl;
	    // one biased
	    *vel = (float*)&fxSet->frameList[fxFrame]->pointData[i]->vecData->at(1);
	    break;
	}
    }
}

#endif


void writeBinaryFiles(FlowDataSet * set, std::string name)
{
    std::string metaName = name + ".meta";
    FILE * metaFile;
    metaFile = fopen(metaName.c_str(),"wb");

    if(!metaFile)
    {
	std::cerr << "Unable to open file: " << metaName << " for writing." << std::endl;
	return;
    }

    int version = BINARY_FILE_VERSION;
    fwrite(&version,sizeof(int),1,metaFile);

    int temp = set->frameList.size();
    fwrite(&temp,sizeof(int),1,metaFile);
    fwrite(&set->bb.xMin(),sizeof(float),1,metaFile);
    fwrite(&set->bb.xMax(),sizeof(float),1,metaFile);
    fwrite(&set->bb.yMin(),sizeof(float),1,metaFile);
    fwrite(&set->bb.yMax(),sizeof(float),1,metaFile);
    fwrite(&set->bb.zMin(),sizeof(float),1,metaFile);
    fwrite(&set->bb.zMax(),sizeof(float),1,metaFile);

    temp = set->attribRanges.size();
    fwrite(&temp,sizeof(int),1,metaFile);

    char buffer[1024];
    buffer[1023] = '\0';

    for(std::map<std::string,std::pair<float,float> >::iterator it = set->attribRanges.begin(); it != set->attribRanges.end(); ++it)
    {
	strncpy(buffer,it->first.c_str(),1023);
	fwrite(buffer,sizeof(char),1024,metaFile);
	fwrite(&it->second.first,sizeof(float),1,metaFile);
	fwrite(&it->second.second,sizeof(float),1,metaFile);
    }

    int maxInd = 0;
    int maxVert = 0;
    int maxSurf = 0;

    for(int i = 0; i < set->frameList.size(); ++i)
    {
	if(set->frameList[i]->verts->size() > maxVert)
	{
	    maxVert = set->frameList[i]->verts->size();
	}
	if(set->frameList[i]->indices->size() > maxInd)
	{
	    maxInd = set->frameList[i]->indices->size();
	}
	if(set->frameList[i]->surfaceInd->size() > maxSurf)
	{
	    maxSurf = set->frameList[i]->surfaceInd->size();
	}
    }

    fwrite(&maxInd,sizeof(int),1,metaFile);
    fwrite(&maxVert,sizeof(int),1,metaFile);
    fwrite(&maxSurf,sizeof(int),1,metaFile);

    
    for(int i = 0; i < set->frameList.size(); ++i)
    {
	std::stringstream framess;
	framess << name << i << ".bin";

	FILE * frameFile = fopen(framess.str().c_str(),"wb");
	if(!frameFile)
	{
	    std::cerr << "Unable to open file: " << framess.str() << " for writing." << std::endl;
	    continue;
	}

	int offset = 0;

	temp = set->frameList[i]->indices->size();
	fwrite(&temp,sizeof(int),1,metaFile);
	fwrite(&offset,sizeof(int),1,metaFile);
	fwrite(&set->frameList[i]->indices->at(0),sizeof(unsigned int),temp,frameFile);
	offset += temp * sizeof(unsigned int);

	temp = set->frameList[i]->verts->size();
	fwrite(&temp,sizeof(int),1,metaFile);
	fwrite(&offset,sizeof(int),1,metaFile);
	fwrite(&set->frameList[i]->verts->at(0),sizeof(float)*3,temp,frameFile);
	offset += temp * 3 * sizeof(float);

	temp = set->frameList[i]->surfaceInd->size();
	fwrite(&temp,sizeof(int),1,metaFile);
	fwrite(&offset,sizeof(int),1,metaFile);
	fwrite(&set->frameList[i]->surfaceInd->at(0),sizeof(unsigned int),temp,frameFile);
	offset += temp * sizeof(unsigned int);

	temp = set->frameList[i]->pointData.size();
	fwrite(&temp,sizeof(int),1,metaFile);

	for(int j = 0; j < set->frameList[i]->pointData.size(); ++j)
	{
	    strncpy(buffer,set->frameList[i]->pointData[j]->name.c_str(),1023);
	    fwrite(buffer,sizeof(char),1024,metaFile);

	    temp = (int)set->frameList[i]->pointData[j]->attribType;
	    fwrite(&temp,sizeof(int),1,metaFile);
	    temp = (int)set->frameList[i]->pointData[j]->dataType;
	    fwrite(&temp,sizeof(int),1,metaFile);

	    if(set->frameList[i]->pointData[j]->dataType == VDT_INT)
	    {
		fwrite(&set->frameList[i]->pointData[j]->intMin,sizeof(int),1,metaFile);
		fwrite(&set->frameList[i]->pointData[j]->intMax,sizeof(int),1,metaFile);
	    }
	    else
	    {
		fwrite(&set->frameList[i]->pointData[j]->floatMin,sizeof(float),1,metaFile);
		fwrite(&set->frameList[i]->pointData[j]->floatMax,sizeof(float),1,metaFile);
	    }

	    if(set->frameList[i]->pointData[j]->attribType == VAT_VECTORS)
	    {
		fwrite(&set->frameList[i]->pointData[j]->vecData->at(0),3*sizeof(float),set->frameList[i]->verts->size(),frameFile);
		fwrite(&offset,sizeof(int),1,metaFile);
		offset += set->frameList[i]->verts->size()*3*sizeof(float); 
	    }
	    else
	    {
		int elementSize;
		void * addr;
		if(set->frameList[i]->pointData[j]->dataType == VDT_INT)
		{
		    elementSize = sizeof(int);
		    addr = &set->frameList[i]->pointData[j]->intData->at(0);
		}
		else
		{
		    elementSize= sizeof(float);
		    addr = &set->frameList[i]->pointData[j]->floatData->at(0);
		}

		fwrite(addr,elementSize,set->frameList[i]->verts->size(),frameFile);
		fwrite(&offset,sizeof(int),1,metaFile);
		offset += set->frameList[i]->verts->size()*elementSize;
	    }
	}

	if(set->frameList[i]->vcoreData)
	{
	    temp = set->frameList[i]->vcoreData->verts->size();
	    fwrite(&temp,sizeof(int),1,metaFile);
	    fwrite(&offset,sizeof(int),1,metaFile);
	    fwrite(&set->frameList[i]->vcoreData->verts->at(0),3*sizeof(float),temp,frameFile);
	    fwrite(&set->frameList[i]->vcoreData->coreStr->at(0),sizeof(float),temp,frameFile);
	    offset += temp * (3*sizeof(float) + sizeof(float));

	    temp = set->frameList[i]->vcoreData->coreSegments.size();
	    fwrite(&temp,sizeof(int),1,metaFile);
	    for(int j = 0; j < set->frameList[i]->vcoreData->coreSegments.size(); ++j)
	    {
		int first, count;
		first = set->frameList[i]->vcoreData->coreSegments[j]->getFirst();
		count = set->frameList[i]->vcoreData->coreSegments[j]->getCount();
		fwrite(&first,sizeof(int),1,metaFile);
		fwrite(&count,sizeof(int),1,metaFile);
	    }
	}
	else
	{
	    temp = 0;
	    fwrite(&temp,sizeof(int),1,metaFile);
	    fwrite(&offset,sizeof(int),1,metaFile);
	    fwrite(&temp,sizeof(int),1,metaFile);
	}

	if(set->frameList[i]->sepAttData)
	{
	    temp = set->frameList[i]->sepAttData->sverts->size();
	    fwrite(&temp,sizeof(int),1,metaFile);
	    fwrite(&offset,sizeof(int),1,metaFile);
	    fwrite(&set->frameList[i]->sepAttData->sverts->at(0),3*sizeof(float),temp,frameFile);
	    offset += temp*3*sizeof(float);

	    temp = set->frameList[i]->sepAttData->sSegments.size();
	    fwrite(&temp,sizeof(int),1,metaFile);
	    for(int j = 0; j < set->frameList[i]->sepAttData->sSegments.size(); ++j)
	    {
		int first, count;
		first = set->frameList[i]->sepAttData->sSegments[j]->getFirst();
		count = set->frameList[i]->sepAttData->sSegments[j]->getCount();
		fwrite(&first,sizeof(int),1,metaFile);
		fwrite(&count,sizeof(int),1,metaFile);
	    }

	    temp = set->frameList[i]->sepAttData->averts->size();
	    fwrite(&temp,sizeof(int),1,metaFile);
	    fwrite(&offset,sizeof(int),1,metaFile);
	    fwrite(&set->frameList[i]->sepAttData->averts->at(0),3*sizeof(float),temp,frameFile);
	    offset += temp*3*sizeof(float);

	    temp = set->frameList[i]->sepAttData->aSegments.size();
	    fwrite(&temp,sizeof(int),1,metaFile);
	    for(int j = 0; j < set->frameList[i]->sepAttData->aSegments.size(); ++j)
	    {
		int first, count;
		first = set->frameList[i]->sepAttData->aSegments[j]->getFirst();
		count = set->frameList[i]->sepAttData->aSegments[j]->getCount();
		fwrite(&first,sizeof(int),1,metaFile);
		fwrite(&count,sizeof(int),1,metaFile);
	    }
	}
	else
	{
	    temp = 0;
	    fwrite(&temp,sizeof(int),1,metaFile);
	    fwrite(&offset,sizeof(int),1,metaFile);
	    fwrite(&temp,sizeof(int),1,metaFile);

	    fwrite(&temp,sizeof(int),1,metaFile);
	    fwrite(&offset,sizeof(int),1,metaFile);
	    fwrite(&temp,sizeof(int),1,metaFile);
	}

	fclose(frameFile);
    }
    fclose(metaFile);
}

void writeBinaryFrameFile(FlowDataSet * set, std::string name, int frame)
{
    std::stringstream framess;
    framess << name << frame << ".bin";

    FILE * frameFile = fopen(framess.str().c_str(),"wb");
    if(!frameFile)
    {
	std::cerr << "Unable to open file: " << framess.str() << " for writing." << std::endl;
	return;;
    }

    int offset = 0;
    int temp;

    temp = set->frameList[frame]->indices->size();
    //fwrite(&temp,sizeof(int),1,metaFile);
    //fwrite(&offset,sizeof(int),1,metaFile);
    fwrite(&set->frameList[frame]->indices->at(0),sizeof(unsigned int),temp,frameFile);
    set->frameList[frame]->indicesDataOffset = offset;
    set->frameList[frame]->indicesDataSize = temp;
    offset += temp * sizeof(unsigned int);

    temp = set->frameList[frame]->verts->size();
    //fwrite(&temp,sizeof(int),1,metaFile);
    //fwrite(&offset,sizeof(int),1,metaFile);
    fwrite(&set->frameList[frame]->verts->at(0),sizeof(float)*3,temp,frameFile);
    set->frameList[frame]->vertsDataOffset = offset;
    set->frameList[frame]->vertsDataSize = temp;
    offset += temp * 3 * sizeof(float);

    temp = set->frameList[frame]->surfaceInd->size();
    //fwrite(&temp,sizeof(int),1,metaFile);
    //fwrite(&offset,sizeof(int),1,metaFile);
    fwrite(&set->frameList[frame]->surfaceInd->at(0),sizeof(unsigned int),temp,frameFile);
    set->frameList[frame]->surfaceDataOffset = offset;
    set->frameList[frame]->surfaceDataSize = temp;
    offset += temp * sizeof(unsigned int);

    //temp = set->frameList[i]->pointData.size();
    //fwrite(&temp,sizeof(int),1,metaFile);

    for(int j = 0; j < set->frameList[frame]->pointData.size(); ++j)
    {
	//strncpy(buffer,set->frameList[i]->pointData[j]->name.c_str(),1023);
	//fwrite(buffer,sizeof(char),1024,metaFile);

	//temp = (int)set->frameList[i]->pointData[j]->attribType;
	//fwrite(&temp,sizeof(int),1,metaFile);
	//temp = (int)set->frameList[i]->pointData[j]->dataType;
	//fwrite(&temp,sizeof(int),1,metaFile);

	//if(set->frameList[i]->pointData[j]->dataType == VDT_INT)
	//{
	//    fwrite(&set->frameList[i]->pointData[j]->intMin,sizeof(int),1,metaFile);
	//    fwrite(&set->frameList[i]->pointData[j]->intMax,sizeof(int),1,metaFile);
	//}
	//else
	//{
	//    fwrite(&set->frameList[i]->pointData[j]->floatMin,sizeof(float),1,metaFile);
	//    fwrite(&set->frameList[i]->pointData[j]->floatMax,sizeof(float),1,metaFile);
	//}

	if(set->frameList[frame]->pointData[j]->attribType == VAT_VECTORS)
	{
	    fwrite(&set->frameList[frame]->pointData[j]->vecData->at(0),3*sizeof(float),set->frameList[frame]->verts->size(),frameFile);
	    //fwrite(&offset,sizeof(int),1,metaFile);
	    set->frameList[frame]->pointData[j]->dataOffset = offset;
	    offset += set->frameList[frame]->verts->size()*3*sizeof(float); 
	}
	else
	{
	    int elementSize;
	    void * addr;
	    if(set->frameList[frame]->pointData[j]->dataType == VDT_INT)
	    {
		elementSize = sizeof(int);
		addr = &set->frameList[frame]->pointData[j]->intData->at(0);
	    }
	    else
	    {
		elementSize= sizeof(float);
		addr = &set->frameList[frame]->pointData[j]->floatData->at(0);
	    }

	    fwrite(addr,elementSize,set->frameList[frame]->verts->size(),frameFile);
	    set->frameList[frame]->pointData[j]->dataOffset = offset;
	    //fwrite(&offset,sizeof(int),1,metaFile);
	    offset += set->frameList[frame]->verts->size()*elementSize;
	}
    }

    if(set->frameList[frame]->vcoreData)
    {
	temp = set->frameList[frame]->vcoreData->verts->size();
	//fwrite(&temp,sizeof(int),1,metaFile);
	//fwrite(&offset,sizeof(int),1,metaFile);
	fwrite(&set->frameList[frame]->vcoreData->verts->at(0),3*sizeof(float),temp,frameFile);
	fwrite(&set->frameList[frame]->vcoreData->coreStr->at(0),sizeof(float),temp,frameFile);
	set->frameList[frame]->vcoreData->coreDataOffset = offset;
	set->frameList[frame]->vcoreData->coreDataSize = temp;
	offset += temp * (3*sizeof(float) + sizeof(float));

	//temp = set->frameList[i]->vcoreData->coreSegments.size();
	//fwrite(&temp,sizeof(int),1,metaFile);
	//for(int j = 0; j < set->frameList[i]->vcoreData->coreSegments.size(); ++j)
	//{
	//    int first, count;
	//    first = set->frameList[i]->vcoreData->coreSegments[j]->getFirst();
	//    count = set->frameList[i]->vcoreData->coreSegments[j]->getCount();
	//    fwrite(&first,sizeof(int),1,metaFile);
	//    fwrite(&count,sizeof(int),1,metaFile);
	//}
    }
    else
    {
	//temp = 0;
	//fwrite(&temp,sizeof(int),1,metaFile);
	//fwrite(&offset,sizeof(int),1,metaFile);
	//fwrite(&temp,sizeof(int),1,metaFile);
    }

    if(set->frameList[frame]->sepAttData)
    {
	temp = set->frameList[frame]->sepAttData->sverts->size();
	//fwrite(&temp,sizeof(int),1,metaFile);
	//fwrite(&offset,sizeof(int),1,metaFile);
	fwrite(&set->frameList[frame]->sepAttData->sverts->at(0),3*sizeof(float),temp,frameFile);
	set->frameList[frame]->sepAttData->sDataOffset = offset;
	set->frameList[frame]->sepAttData->sDataSize = temp;
	offset += temp*3*sizeof(float);

	//temp = set->frameList[i]->sepAttData->sSegments.size();
	//fwrite(&temp,sizeof(int),1,metaFile);
	//for(int j = 0; j < set->frameList[i]->sepAttData->sSegments.size(); ++j)
	//{
	//    int first, count;
	//    first = set->frameList[i]->sepAttData->sSegments[j]->getFirst();
	//    count = set->frameList[i]->sepAttData->sSegments[j]->getCount();
	//    fwrite(&first,sizeof(int),1,metaFile);
	//    fwrite(&count,sizeof(int),1,metaFile);
	//}

	temp = set->frameList[frame]->sepAttData->averts->size();
	//fwrite(&temp,sizeof(int),1,metaFile);
	//fwrite(&offset,sizeof(int),1,metaFile);
	fwrite(&set->frameList[frame]->sepAttData->averts->at(0),3*sizeof(float),temp,frameFile);
	set->frameList[frame]->sepAttData->aDataOffset = offset;
	set->frameList[frame]->sepAttData->aDataSize = temp;
	offset += temp*3*sizeof(float);

	//temp = set->frameList[i]->sepAttData->aSegments.size();
	//fwrite(&temp,sizeof(int),1,metaFile);
	//for(int j = 0; j < set->frameList[i]->sepAttData->aSegments.size(); ++j)
	//{
	//    int first, count;
	//    first = set->frameList[i]->sepAttData->aSegments[j]->getFirst();
	//    count = set->frameList[i]->sepAttData->aSegments[j]->getCount();
	//    fwrite(&first,sizeof(int),1,metaFile);
	//    fwrite(&count,sizeof(int),1,metaFile);
	//}
    }
    else
    {
	//temp = 0;
	//fwrite(&temp,sizeof(int),1,metaFile);
	//fwrite(&offset,sizeof(int),1,metaFile);
	//fwrite(&temp,sizeof(int),1,metaFile);

	//fwrite(&temp,sizeof(int),1,metaFile);
	//fwrite(&offset,sizeof(int),1,metaFile);
	//fwrite(&temp,sizeof(int),1,metaFile);
    }

    fclose(frameFile);

    // clear the raw data in the frame
    set->frameList[frame]->verts = new osg::Vec3Array();
    set->frameList[frame]->indices = new osg::DrawElementsUInt();
    set->frameList[frame]->surfaceInd = new osg::DrawElementsUInt();
    if(set->frameList[frame]->surfaceFacets)
    {
	set->frameList[frame]->surfaceFacets = new osg::Vec4iArray();
	set->frameList[frame]->surfaceCells = new osg::IntArray();
	set->frameList[frame]->cellTypes = new osg::IntArray();
    }

    for(int i = 0; i < set->frameList[frame]->cellData.size(); ++i)
    {
	if(set->frameList[frame]->cellData[i]->attribType == VAT_VECTORS)
	{
	    set->frameList[frame]->cellData[i]->vecData = new osg::Vec3Array();
	}
	else if(set->frameList[frame]->cellData[i]->dataType == VDT_INT)
	{
	    set->frameList[frame]->cellData[i]->intData = new osg::IntArray();
	}
	else if(set->frameList[frame]->cellData[i]->dataType == VDT_DOUBLE)
	{
	    set->frameList[frame]->cellData[i]->floatData = new osg::FloatArray();
	}
    }

    for(int i = 0; i < set->frameList[frame]->pointData.size(); ++i)
    {
	if(set->frameList[frame]->pointData[i]->attribType == VAT_VECTORS)
	{
	    set->frameList[frame]->pointData[i]->vecData = new osg::Vec3Array();
	}
	else if(set->frameList[frame]->pointData[i]->dataType == VDT_INT)
	{
	    set->frameList[frame]->pointData[i]->intData = new osg::IntArray();
	}
	else if(set->frameList[frame]->pointData[i]->dataType == VDT_DOUBLE)
	{
	    set->frameList[frame]->pointData[i]->floatData = new osg::FloatArray();
	}
    }

    if(set->frameList[frame]->vcoreData)
    {
	set->frameList[frame]->vcoreData->verts = new osg::Vec3Array();
	set->frameList[frame]->vcoreData->coreStr = new osg::FloatArray();
    }

    if(set->frameList[frame]->sepAttData)
    {
	set->frameList[frame]->sepAttData->sverts = new osg::Vec3Array();
	set->frameList[frame]->sepAttData->averts = new osg::Vec3Array();
    }
}

void writeBinaryMetaFile(FlowDataSet * set, std::string name)
{
    std::string metaName = name + ".meta";
    FILE * metaFile;
    metaFile = fopen(metaName.c_str(),"wb");

    if(!metaFile)
    {
	std::cerr << "Unable to open file: " << metaName << " for writing." << std::endl;
	return;
    }

    int version = BINARY_FILE_VERSION;
    fwrite(&version,sizeof(int),1,metaFile);

    int temp = set->frameList.size();
    fwrite(&temp,sizeof(int),1,metaFile);
    fwrite(&set->bb.xMin(),sizeof(float),1,metaFile);
    fwrite(&set->bb.xMax(),sizeof(float),1,metaFile);
    fwrite(&set->bb.yMin(),sizeof(float),1,metaFile);
    fwrite(&set->bb.yMax(),sizeof(float),1,metaFile);
    fwrite(&set->bb.zMin(),sizeof(float),1,metaFile);
    fwrite(&set->bb.zMax(),sizeof(float),1,metaFile);

    temp = set->attribRanges.size();
    fwrite(&temp,sizeof(int),1,metaFile);

    char buffer[1024];
    buffer[1023] = '\0';

    for(std::map<std::string,std::pair<float,float> >::iterator it = set->attribRanges.begin(); it != set->attribRanges.end(); ++it)
    {
	strncpy(buffer,it->first.c_str(),1023);
	fwrite(buffer,sizeof(char),1024,metaFile);
	fwrite(&it->second.first,sizeof(float),1,metaFile);
	fwrite(&it->second.second,sizeof(float),1,metaFile);
    }

    int maxInd = 0;
    int maxVert = 0;
    int maxSurf = 0;

    for(int i = 0; i < set->frameList.size(); ++i)
    {
	if(set->frameList[i]->vertsDataSize > maxVert)
	{
	    maxVert = set->frameList[i]->vertsDataSize;
	}
	if(set->frameList[i]->indicesDataSize > maxInd)
	{
	    maxInd = set->frameList[i]->indicesDataSize;
	}
	if(set->frameList[i]->surfaceDataSize > maxSurf)
	{
	    maxSurf = set->frameList[i]->surfaceDataSize;
	}
    }

    fwrite(&maxInd,sizeof(int),1,metaFile);
    fwrite(&maxVert,sizeof(int),1,metaFile);
    fwrite(&maxSurf,sizeof(int),1,metaFile);

    
    for(int i = 0; i < set->frameList.size(); ++i)
    {
	int offset = 0;

	temp = set->frameList[i]->indicesDataSize;
	offset = set->frameList[i]->indicesDataOffset;
	fwrite(&temp,sizeof(int),1,metaFile);
	fwrite(&offset,sizeof(int),1,metaFile);
	//fwrite(&set->frameList[i]->indices->at(0),sizeof(unsigned int),temp,frameFile);
	//offset += temp * sizeof(unsigned int);

	temp = set->frameList[i]->vertsDataSize;
	offset = set->frameList[i]->vertsDataOffset;
	fwrite(&temp,sizeof(int),1,metaFile);
	fwrite(&offset,sizeof(int),1,metaFile);
	//fwrite(&set->frameList[i]->verts->at(0),sizeof(float)*3,temp,frameFile);
	//offset += temp * 3 * sizeof(float);

	temp = set->frameList[i]->surfaceDataSize;
	offset = set->frameList[i]->surfaceDataOffset;
	fwrite(&temp,sizeof(int),1,metaFile);
	fwrite(&offset,sizeof(int),1,metaFile);
	//fwrite(&set->frameList[i]->surfaceInd->at(0),sizeof(unsigned int),temp,frameFile);
	//offset += temp * sizeof(unsigned int);

	temp = set->frameList[i]->pointData.size();
	fwrite(&temp,sizeof(int),1,metaFile);

	for(int j = 0; j < set->frameList[i]->pointData.size(); ++j)
	{
	    strncpy(buffer,set->frameList[i]->pointData[j]->name.c_str(),1023);
	    fwrite(buffer,sizeof(char),1024,metaFile);

	    temp = (int)set->frameList[i]->pointData[j]->attribType;
	    fwrite(&temp,sizeof(int),1,metaFile);
	    temp = (int)set->frameList[i]->pointData[j]->dataType;
	    fwrite(&temp,sizeof(int),1,metaFile);

	    if(set->frameList[i]->pointData[j]->dataType == VDT_INT)
	    {
		fwrite(&set->frameList[i]->pointData[j]->intMin,sizeof(int),1,metaFile);
		fwrite(&set->frameList[i]->pointData[j]->intMax,sizeof(int),1,metaFile);
	    }
	    else
	    {
		fwrite(&set->frameList[i]->pointData[j]->floatMin,sizeof(float),1,metaFile);
		fwrite(&set->frameList[i]->pointData[j]->floatMax,sizeof(float),1,metaFile);
	    }

	    if(set->frameList[i]->pointData[j]->attribType == VAT_VECTORS)
	    {
		//fwrite(&set->frameList[i]->pointData[j]->vecData->at(0),3*sizeof(float),set->frameList[i]->verts->size(),frameFile);
		offset = set->frameList[i]->pointData[j]->dataOffset;
		fwrite(&offset,sizeof(int),1,metaFile);
		//offset += set->frameList[i]->verts->size()*3*sizeof(float); 
	    }
	    else
	    {
		//int elementSize;
		//void * addr;
		//if(set->frameList[i]->pointData[j]->dataType == VDT_INT)
		//{
		    //elementSize = sizeof(int);
		    //addr = &set->frameList[i]->pointData[j]->intData->at(0);
		//}
		//else
		//{
		    //elementSize= sizeof(float);
		    //addr = &set->frameList[i]->pointData[j]->floatData->at(0);
		//}

		//fwrite(addr,elementSize,set->frameList[i]->verts->size(),frameFile);
		offset = set->frameList[i]->pointData[j]->dataOffset;
		fwrite(&offset,sizeof(int),1,metaFile);
		//offset += set->frameList[i]->verts->size()*elementSize;
	    }
	}

	if(set->frameList[i]->vcoreData)
	{
	    temp = set->frameList[i]->vcoreData->coreDataSize;
	    offset = set->frameList[i]->vcoreData->coreDataOffset;
	    fwrite(&temp,sizeof(int),1,metaFile);
	    fwrite(&offset,sizeof(int),1,metaFile);
	    //fwrite(&set->frameList[i]->vcoreData->verts->at(0),3*sizeof(float),temp,frameFile);
	    //fwrite(&set->frameList[i]->vcoreData->coreStr->at(0),sizeof(float),temp,frameFile);
	    //offset += temp * (3*sizeof(float) + sizeof(float));

	    temp = set->frameList[i]->vcoreData->coreSegments.size();
	    fwrite(&temp,sizeof(int),1,metaFile);
	    for(int j = 0; j < set->frameList[i]->vcoreData->coreSegments.size(); ++j)
	    {
		int first, count;
		first = set->frameList[i]->vcoreData->coreSegments[j]->getFirst();
		count = set->frameList[i]->vcoreData->coreSegments[j]->getCount();
		fwrite(&first,sizeof(int),1,metaFile);
		fwrite(&count,sizeof(int),1,metaFile);
	    }
	}
	else
	{
	    temp = 0;
	    fwrite(&temp,sizeof(int),1,metaFile);
	    fwrite(&offset,sizeof(int),1,metaFile);
	    fwrite(&temp,sizeof(int),1,metaFile);
	}

	if(set->frameList[i]->sepAttData)
	{
	    temp = set->frameList[i]->sepAttData->sDataSize;
	    offset = set->frameList[i]->sepAttData->sDataOffset;
	    fwrite(&temp,sizeof(int),1,metaFile);
	    fwrite(&offset,sizeof(int),1,metaFile);
	    //fwrite(&set->frameList[i]->sepAttData->sverts->at(0),3*sizeof(float),temp,frameFile);
	    //offset += temp*3*sizeof(float);

	    temp = set->frameList[i]->sepAttData->sSegments.size();
	    fwrite(&temp,sizeof(int),1,metaFile);
	    for(int j = 0; j < set->frameList[i]->sepAttData->sSegments.size(); ++j)
	    {
		int first, count;
		first = set->frameList[i]->sepAttData->sSegments[j]->getFirst();
		count = set->frameList[i]->sepAttData->sSegments[j]->getCount();
		fwrite(&first,sizeof(int),1,metaFile);
		fwrite(&count,sizeof(int),1,metaFile);
	    }

	    temp = set->frameList[i]->sepAttData->aDataSize;
	    offset = set->frameList[i]->sepAttData->aDataOffset;
	    fwrite(&temp,sizeof(int),1,metaFile);
	    fwrite(&offset,sizeof(int),1,metaFile);
	    //fwrite(&set->frameList[i]->sepAttData->averts->at(0),3*sizeof(float),temp,frameFile);
	    //offset += temp*3*sizeof(float);

	    temp = set->frameList[i]->sepAttData->aSegments.size();
	    fwrite(&temp,sizeof(int),1,metaFile);
	    for(int j = 0; j < set->frameList[i]->sepAttData->aSegments.size(); ++j)
	    {
		int first, count;
		first = set->frameList[i]->sepAttData->aSegments[j]->getFirst();
		count = set->frameList[i]->sepAttData->aSegments[j]->getCount();
		fwrite(&first,sizeof(int),1,metaFile);
		fwrite(&count,sizeof(int),1,metaFile);
	    }
	}
	else
	{
	    temp = 0;
	    fwrite(&temp,sizeof(int),1,metaFile);
	    fwrite(&offset,sizeof(int),1,metaFile);
	    fwrite(&temp,sizeof(int),1,metaFile);

	    fwrite(&temp,sizeof(int),1,metaFile);
	    fwrite(&offset,sizeof(int),1,metaFile);
	    fwrite(&temp,sizeof(int),1,metaFile);
	}
    }
    fclose(metaFile);
}
