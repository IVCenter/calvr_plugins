#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

int main(int argc, char** argv)
{
	if(argc <= 1)
	{
		cerr << "Need to xyz file as argument\n";
		return 1;
	}

	// read in base file name
	string filename(argv[1]);
	string newFile(argv[1]);

	size_t found = filename.find(".xyz");
	if(found == string::npos)
	{
		cerr << "No xyz file found\n";
		return 1;
	}

	newFile.replace(found, 4, ".xyb");

	// create a stream to read in file
        ifstream ifs( filename.c_str() );
	ofstream ofs( newFile.c_str() , ios::out|ios::binary);

        string value, values;
        stringstream ss;
        stringstream ssdouble;
	float point[3];
	float color[3];

        while( getline( ifs, values ) )
        {
                ss << values;

                int index = 0;
                while(ss >> value)
                {
                        ssdouble << value;

                        if( index < 3 )
                        {
                                ssdouble >> point[index];
                        }
                        else
                        {
                                ssdouble >> color[index - 3];
                                color[index - 3]/=255.0;
                        }

                        ssdouble.clear();
                        index++;

                }

		// need to write binary point out
		ofs.write((char*)point, sizeof(float) * 3);
		ofs.write((char*)color, sizeof(float) * 3);

                ss.clear();
        }
        ifs.close();
	ofs.close();

	return 0;
}
