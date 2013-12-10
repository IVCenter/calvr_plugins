#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstring>
#include <cstdlib>

using namespace std;

int main(int argc, char** argv)
{
	if(argc <= 1 || !strcmp(argv[1],"--help") || !strcmp(argv[1],"-h"))
	{
		cerr << "Usage: " << argv[0] << " [-r reductionFactor] [-o outfile] input.xyz" << std::endl;
		return 1;
	}

	string filename;
	string newFile;
	string tempFile;
        bool isPts = false;
	int reductionFactor = 1;

	int argindex = 1;
	while(argindex < argc)
	{
	    if(!strcmp(argv[argindex],"-r"))
	    {
		argindex++;
		if(argindex < argc)
		{
		    reductionFactor = atoi(argv[argindex]);
		    if(reductionFactor <= 0)
		    {
			reductionFactor = 1;
		    }
		}
	    }
	    else if(!strcmp(argv[argindex],"-o"))
	    {
		argindex++;
		if(argindex < argc)
		{
		    newFile = argv[argindex];
		}
	    }
	    else
	    {
		filename = argv[argindex];
		break;
	    }
	    argindex++;
	}

	if(filename.empty())
	{
	    std::cerr << "Error: no input file." << std::endl;
	    return 1;
	}

	if(newFile.empty())
	{
	    newFile = filename;
	    size_t found = newFile.find(".xyz");
	    if(found == string::npos)
	    {
		cerr << "No xyz file found\n";
		return 1;
	    }

	    newFile.replace(found, 4, ".xyb");
	}

       tempFile = filename;
       size_t found = tempFile.find(".xyz");
       found = tempFile.find(".pts");
       if(found != string::npos)
       {
         isPts = true;
         std::cerr << "isPts = true\n";
       }
	std::cerr << "Input file: " << filename << std::endl;
	std::cerr << "Output file: " << newFile << std::endl;

	// create a stream to read in file
        ifstream ifs( filename.c_str() );
	ofstream ofs( newFile.c_str() , ios::out|ios::binary);

	if(ifs.fail())
	{
	    std::cerr << "Error opening input file." << std::endl;
	    return 1;
	}

	if(ofs.fail())
	{
	    std::cerr << "Error opening output file." << std::endl;
	    return 1;
	}

        string value, values;
        stringstream ss;
        stringstream ssdouble;
        ssdouble.precision(19);
	float point[3];
	float color[3];
        float skip;
	unsigned int linecount = 0;

        while( getline( ifs, values ) )
        {
		if(linecount % reductionFactor)
		{
		    linecount++;
		    continue;
		}

                ss << values;

                int index = 0;
                while(ss >> value)
                {
                        ssdouble << value;

                        if( index < 3 )
                        {
                                ssdouble >> point[index];
                        }
                        else if(index < 4 && isPts)
                        {
                           //Skip PTS Intensity value;
                           ssdouble >> skip;
                        } 
                        else if(isPts)
                        {
                                ssdouble >> color[index - 4];
                                color[index - 4]/=255.0;
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
		linecount++;
        }
        ifs.close();
	ofs.close();

	return 0;
}
