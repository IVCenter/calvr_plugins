#pragma warning(disable: 26451)

#include "ImageLoader.hpp"

#ifdef WIN32
#include <Shlwapi.h>
#else
#include <stdlib.h> 
#include <sys/types.h>
#include <dirent.h>
#include <linux/limits.h>
#endif

#ifdef LoadImage
#undef LoadImage
#endif

#include <iostream>

#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h>

#include <algorithm>

using namespace std;

struct Slice {
	DicomImage* image;
	double location;

	Slice() {};
	Slice(const Slice& s) : image(s.image), location(s.location) {};
	Slice(DicomImage* img, double loc) : image(img), location(loc) {};
	~Slice() {};

	inline friend bool operator < (Slice const& a, Slice const& b) {
		return a.location < b.location;
	}
};

osg::Image* CreateTexture(GLenum internalFormat, GLenum type, unsigned int width, unsigned int height, unsigned int depth)
{
	osg::Image* img = new osg::Image();
	img->allocateImage(width, height, depth, internalFormat, type);
	return img;
}



string GetExt(const string& path) {
	size_t k = path.rfind('.');
	if (k == string::npos) return "";
	return path.substr((int)k + 1);
}
string GetName(const string& path) {
	char const* str = path.c_str();

	int f = 0;
	int l = 0;
	for (int i = 0; i < path.length(); i++) {
		if (str[i] == '\\' || str[i] == '/')
			f = i + 1;
		else if (str[i] == '.')
			l = i;
	}

	return path.substr(f, l - f);
}
string GetFullPath(const string& str) {
#ifdef WIN32
	char buf[MAX_PATH];
	if (GetFullPathName(str.c_str(), 256, buf, nullptr) == 0) {
		printf("Failed to get full file path of %s (%d)", str.c_str(), GetLastError());
		return str;
	}
	return string(buf);
#else
	char buf[PATH_MAX];
	realpath(str.c_str(), buf);
	return string(buf);
#endif
}

osg::Image* LoadDicomImage(const string& path, osg::Vec3& size) {
	DicomImage* image = new DicomImage(path.c_str());
	assert(image != NULL);
	assert(image->getStatus() == EIS_Normal);

	// Get information
	DcmFileFormat fileFormat;
	assert(fileFormat.loadFile(path.c_str()).good());
	DcmDataset* dataset = fileFormat.getDataset();
	double spacingX = 0.0;
	double spacingY = 0.0;
	double thickness = 0.0;
	OFCondition cnd;
	cnd = dataset->findAndGetFloat64(DCM_PixelSpacing, spacingX, 0);
	cnd = dataset->findAndGetFloat64(DCM_PixelSpacing, spacingY, 1);
	cnd = dataset->findAndGetFloat64(DCM_SliceThickness, thickness, 0);

	unsigned int w = image->getWidth();
	unsigned int h = image->getHeight();
	unsigned int d = 1;

	// volume size in meters
	size.x() = .001f * (float)spacingX * w;
	size.y() = .001f * (float)spacingY * h;
	size.z() = .001f * (float)thickness;

	osg::Image* img = CreateTexture(GL_RG, GL_UNSIGNED_SHORT, w, h, d);
	uint16_t* data = (uint16_t*)img->data();
	

	memset(data, 0, w * h * d * sizeof(uint16_t) * 2);

	image->setMinMaxWindow();
	uint16_t* pixelData = (uint16_t*)image->getOutputData(16);
	unsigned int j = 0;
	for (unsigned int x = 0; x < w; x++)
		for (unsigned int y = 0; y < h; y++) {
			j = 2 * (x + y * w);
			data[j] = pixelData[x + y * w];
			data[j + 1] = 0xFFFF;
		}

	return img;
}
osg::Image* LoadDicomVolume(const vector<string>& files, osg::Matrix& transform) {

	vector<Slice> images;

	// Get information
	double spacingX = 0.0;
	double spacingY = 0.0;
	double thickness = 0.0;
	double positionX = 0.0;
	double positionY = 0.0;
	double positionZ = 0.0;
	double orientation1X = 0.0;
	double orientation1Y = 0.0;
	double orientation1Z = 0.0;
	double orientation2X = 0.0;
	double orientation2Y = 0.0;
	double orientation2Z = 0.0;
	double minPos = 1e10;
	double maxPos = -1e10;

	OFCondition cnd;


	for (unsigned int i = 0; i < files.size(); i++) {

		//std::cout << files[i].c_str() << std::endl;

		DcmFileFormat* fileFormat = new DcmFileFormat();
		cnd = fileFormat->loadFile(files[i].c_str());
		assert(cnd.good());
		DcmDataset* dataset = fileFormat->getDataset();

		if (i == 0) {
			cnd = dataset->findAndGetFloat64(DCM_PixelSpacing, spacingX, 0);
			cnd = dataset->findAndGetFloat64(DCM_PixelSpacing, spacingY, 1);
			cnd = dataset->findAndGetFloat64(DCM_SliceThickness, thickness, 0);
			cnd = dataset->findAndGetFloat64(DCM_ImagePositionPatient, positionX, 0);
			cnd = dataset->findAndGetFloat64(DCM_ImagePositionPatient, positionY, 1);
			cnd = dataset->findAndGetFloat64(DCM_ImagePositionPatient, positionZ, 2);
			cnd = dataset->findAndGetFloat64(DCM_ImageOrientationPatient, orientation1X, 0);
			cnd = dataset->findAndGetFloat64(DCM_ImageOrientationPatient, orientation1Y, 1);
			cnd = dataset->findAndGetFloat64(DCM_ImageOrientationPatient, orientation1Z, 2);
			cnd = dataset->findAndGetFloat64(DCM_ImageOrientationPatient, orientation2X, 3);
			cnd = dataset->findAndGetFloat64(DCM_ImageOrientationPatient, orientation2Y, 4);
			cnd = dataset->findAndGetFloat64(DCM_ImageOrientationPatient, orientation2Z, 5);
		}


		double x;
		cnd = dataset->findAndGetFloat64(DCM_SliceLocation, x, 0);
		minPos = minPos < x ? minPos : x;
		maxPos = maxPos > x ? maxPos : x;

		DicomImage* img = new DicomImage(files[i].c_str());
		assert(img != NULL);
		assert(img->getStatus() == EIS_Normal);
		images.push_back(Slice(img, x));
		
		delete fileFormat;
	}

	std::sort(images.data(), images.data() + images.size());

	double prevX = images[0].location;
	for (unsigned int i = 1; i < images.size(); ++i)
	{
 		prevX = images[i].location;
	}
	unsigned int w = images[0].image->getWidth();
	unsigned int h = images[0].image->getHeight();
	unsigned int d = (unsigned int)images.size();
	std::cout << "w " << w << "h " << h << "d " << d << std::endl;
	osg::Vec3 size = osg::Vec3(0, 0, 0);
	// volume size in millimeters
	size.x() = (float)spacingX * (float)w;
	size.y() = (float)spacingY * (float)h;
	size.z() = maxPos - minPos + thickness;//(float)thickness * (float)images.size();

	/*printf("%fm x %fm x %fm\n", (float)w, (float)h, (float)images.size());

	printf("%fm x %fm x %fm\n", (float)spacingX, (float)spacingY, (float)thickness);

	printf("%fm x %fm x %fm\n", size.x(), size.y(), size.z());

	printf("min: %f    max: %f", (float)minPos, (float)maxPos);*/

	osg::Vec3 rowdir = osg::Vec3(orientation1X, orientation1Y, orientation1Z);
	osg::Vec3 coldir = osg::Vec3(orientation2X, orientation2Y, orientation2Z);
	osg::Vec3 depthdir = coldir ^ rowdir;

	//transform

	transform.set(
		rowdir.x(), rowdir.y(), rowdir.z(), 0,
		coldir.x(), coldir.y(), coldir.z(), 0,
		depthdir.x(), depthdir.y(), depthdir.z(), 0,
		0, 0, 0, 1
	);

	transform.preMultScale(size);
	std::cout << "size: " << "x: " << size.x() << "y: " << size.y() << "z: " << size.z() << std::endl;
	

	osg::Image* img = CreateTexture(GL_RG, GL_UNSIGNED_SHORT, w, h, d);
	uint16_t* data = (uint16_t*)img->data();
	memset(data, 0, w * h * d * sizeof(uint16_t) * 2);

	for (unsigned int i = 0; i < images.size(); i++) {
		images[i].image->setMinMaxWindow();
		uint16_t* pixelData = (uint16_t*)images[i].image->getOutputData(16);
		uint16_t* slice = data + 2 * i * w * h;
		unsigned int j = 0;
		for (unsigned int y = 0; y < h; y++) {
			for (unsigned int x = 0; x < w; x++) {
				j = 2 * (x + y * w);
				slice[j] = pixelData[x + y * w];
				//slice[j + 1] = 0xFFFF;
			}
		}

		//free memory
		delete images[i].image;
	}

	
	return img;
}

// An unsigned char can store 1 Bytes (8bits) of data (0-255)
typedef unsigned char BYTE;

long getFileSize(FILE* file)
{
	long lCurPos, lEndPos;
	lCurPos = ftell(file);
	fseek(file, 0, 2);
	lEndPos = ftell(file);
	fseek(file, lCurPos, 0);
	return lEndPos;
}

osg::Image* LoadRAWVolumeImage(const string& file, osg::Matrix& transform) {
	std::string whdString = file.substr(file.find_first_of("_")+1, file.find_last_of("_") - file.find_first_of("_")-1);
	std::string wString = whdString.substr(0, whdString.find_first_of("x"));
	std::string hString = whdString.substr(whdString.find_first_of("x")+1, whdString.find_last_of("x") - whdString.find_first_of("x")-1);
	std::string dString = whdString.substr(whdString.find_last_of("x")+1, whdString.length() - whdString.find_last_of("x"));
 	


	unsigned int w = std::stoi(wString);
	unsigned int h = std::stoi(hString);
	unsigned int d = std::stoi(dString);
	//std::cout << "w " << w << "h " << h << "d " << d << std::endl;
	osg::Vec3 size = osg::Vec3(0, 0, 0);
	// volume size in millimeters
	size.x() = (float)w;
	size.y() = (float)h;
	size.z() = (float)d;

	

	transform.preMultScale(size);
	

	//GETDATA
 	FILE* filepath = NULL;		// File pointer
	BYTE* fileBuf; // Pointer to our buffered data
	// Open the file in binary mode using the "rb" format string
	// This also checks if the file exists and/or can be opened for reading correctly
	using namespace std;
	if ((filepath = fopen(file.c_str(), "rb")) == NULL)
		cout << "Could not open specified file" << endl;
	else
		cout << "File opened successfully" << endl;

	// Get the size of the file in bytes
	long fileSize = getFileSize(filepath);

	// Allocate space in the buffer for the whole file
	fileBuf = new BYTE[fileSize];

	// Read the file in to the buffer
	fread(fileBuf, fileSize, 1, filepath);

	// Now that we have the entire file buffered, we can take a look at some binary infomation
	// Lets take a look in hexadecimal
	//for (int i = 0; i < fileSize; i++)
	//	if(i%2048)
	//		printf("%X ", fileBuf[i]);
 
	osg::Image* img = CreateTexture(GL_RG, GL_UNSIGNED_SHORT, w, h, d);
	uint16_t* data = (uint16_t*)img->data();
	memset(data, 0, w * h * d * sizeof(uint16_t) * 2);
	uint16_t* pixelData = (uint16_t*)fileBuf;
	

	for (unsigned int i = 0; i < d; i++) {
		
		uint16_t* slice = data + 2 * i * w * h;

		unsigned int j = 0;

		for (unsigned int y = 0; y < h; y++) {
			for (unsigned int x = 0; x < w; x++) {
				j = 2 * (x + y * w);
				slice[j] = fileBuf[x + y * w + i*h*w] * 255;
				//slice[j + 1] = 0xFFFF;
			}
		}

		//free memory
 	}


	delete[]fileBuf;
	fclose(filepath);   // Almost forgot this 


	return img;
}

osg::Image* ImageLoader::LoadImage(const string& path, osg::Vec3& size) {
	string ext = GetExt(path.c_str());
	if (ext == "dcm")
		return LoadDicomImage(path, size);
	else
		return 0;
}

void GetFiles(const string& path, vector<string>& files) {
#ifdef WIN32
	string d = path + "\\*";

	WIN32_FIND_DATAA ffd;
	HANDLE hFind = FindFirstFile(d.c_str(), &ffd);
	if (hFind == INVALID_HANDLE_VALUE) {
		assert(false);
		return;
	}

	do {
		if (ffd.cFileName[0] == L'.') continue;

		string c = path + "\\" + ffd.cFileName;

		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
			// file is a directory
		}
		else {
			string ext = GetExt(c);
			if (ext == "dcm" || ext == "raw" || ext == "png")
				files.push_back(GetFullPath(c));
		}
	} while (FindNextFileA(hFind, &ffd) != 0);

	FindClose(hFind);
#else    

	DIR* dirp = opendir(path.c_str());
	struct dirent* dp;
	while ((dp = readdir(dirp)) != NULL) {
		const char* ext = strrchr(dp->d_name, '.');
		if (!ext) continue;
		if (strcmp(ext + 1, "dcm") != 0) continue;
		// printf("Helmsley: Found dcm %s\n", dp->d_name);
		files.push_back(path + "/" + dp->d_name);
	}
	std::sort(files.begin(), files.end());
	closedir(dirp);

#endif
}


std::string ImageLoader::GetRawFile(const string& path) {
#ifdef WIN32
	string d = path + "\\*";

	WIN32_FIND_DATAA ffd;
	HANDLE hFind = FindFirstFile(d.c_str(), &ffd);
	if (hFind == INVALID_HANDLE_VALUE) {
		assert(false);
		return "";
	}

	do {
		if (ffd.cFileName[0] == L'.') continue;

		string c = path + "\\" + ffd.cFileName;

		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
			// file is a directory
		}
		else {
			string ext = GetExt(c);
			if (ext == "dcm" || ext == "raw" || ext == "png")
				return GetFullPath(c);
		}
	} while (FindNextFileA(hFind, &ffd) != 0);

	FindClose(hFind);
#else    

	DIR* dirp = opendir(path.c_str());
	struct dirent* dp;
	while ((dp = readdir(dirp)) != NULL) {
		const char* ext = strrchr(dp->d_name, '.');
		if (!ext) continue;
		if (strcmp(ext + 1, "raw") != 0) continue;
		// printf("Helmsley: Found dcm %s\n", dp->d_name);
		return path + "/" + dp->d_name);
	}
	
	closedir(dirp);

#endif
}

osg::Image* ImageLoader::LoadVolume(const string& path, osg::Matrix& transform) {

	vector<string> files;
	GetFiles(path, files);
	OSG_NOTICE << "Number of files: " << files.size() << std::endl;

	if (files.size() == 0) return 0;

	string ext = GetExt(files[0]);
	if (ext == "dcm")
		return LoadDicomVolume(files, transform);
	else
		return 0;
}
osg::Image* ImageLoader::LoadRawVolume(const string& path, osg::Matrix& transform) {

	string file = GetRawFile(path);
 
	if (file.empty())
		return 0;

	string ext = GetExt(file);
	if (ext == "raw")
		return LoadRAWVolumeImage(file, transform);
	else
		return 0;
}