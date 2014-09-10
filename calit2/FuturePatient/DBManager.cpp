#include "DBManager.h"
#include "md5.h"

#include <iostream>
#include <fstream>
#include <vector>

DBManager::DBManager(std::string database, std::string server, std::string user, std::string password, std::string cacheDir)
{
    _conn = new mysqlpp::Connection(false);
    if(!_conn->connect(database.c_str(),server.c_str(),user.c_str(),password.c_str()))
    {
	std::cerr << "DBManager: Unable to connect to database." << std::endl;
	delete _conn;
	_conn = NULL;
    }

    _useCache = true;

    if(!cacheDir.empty())
    {
	_withCache = true;
	_cacheDir = cacheDir;
    }
    else
    {
	_withCache = false;
    }
}

DBManager::~DBManager()
{
}

bool DBManager::isConnected()
{
    return _conn;
}

bool DBManager::runQuery(std::string query, DBMQueryResult & result)
{
    if(!_conn)
    {
	result = DBMQueryResult(0,0,std::map<std::string,int>());
	return false;
    }

    if(_withCache && _useCache)
    {
	bool status;
	status = loadFromCache(query,result);
	if(!status)
	{
	    status = loadFromDB(query,result);
	    if(status)
	    {
		writeToCache(query,result);
	    }
	}
	return status;
    }
    else
    {
	bool status = loadFromDB(query,result);
	if(status && _withCache)
	{
	    writeToCache(query,result);
	}
	return status;
    }
}

bool DBManager::loadFromCache(const std::string & query, DBMQueryResult & result)
{
    std::string fileHash = md5(query);
    std::string file = _cacheDir + "/" + fileHash + ".dbc";
    //std::cerr << "Query: " << query << std::endl;
    //std::cerr << "CacheFile: " << file << std::endl;

    result = DBMQueryResult(0,0,std::map<std::string,int>());

    std::string line;
    std::ifstream infile(file.c_str());
    if(infile.is_open())
    {
	std::map<std::string,int> colNames;
	int numCols = 0;
	int numRows = 0;

	std::string cquery;
	if(getline(infile,cquery))
	{
	    //std::cerr << "Cached query: " << cquery << std::endl;
	}
	else
	{
	    infile.close();
	    return false;
	}

	if(getline(infile,line))
	{
	    numCols = atoi(line.c_str());
	}
	else
	{
	    infile.close();
	    return false;
	}

	if(getline(infile,line))
	{
	    numRows = atoi(line.c_str());
	}
	else
	{
	    infile.close();
	    return false;
	}

	for(int i = 0; i < numCols; ++i)
	{ 
	    if(getline(infile,line))
	    {
		colNames[line] = i;
	    }
	    else
	    {
		infile.close();
		return false;
	    }
	}

	result = DBMQueryResult(numCols,numRows,colNames);

	for(int i = 0; i < numRows; ++i)
	{
	    for(int j = 0; j < numCols; ++j)
	    {
		if(getline(infile,line))
		{
		    result(i,j) = line;
		}
		else
		{
		    infile.close();
		    return false;
		}
	    }
	}
	infile.close();
        return true;
    }
    else
    {
	//std::cerr << "DBManager: unable to open cache file: " << file << std::endl;
	return false;
    }
}

bool DBManager::loadFromDB(const std::string & query, DBMQueryResult & result)
{
    //std::cerr << "Reading from DB, query: " << query << std::endl;

    if(!isConnected())
    {
	result = DBMQueryResult();
	return false;
    }

    mysqlpp::Query q = _conn->query(query.c_str());
    mysqlpp::StoreQueryResult res = q.store();

    if(res.num_rows() == 0)
    {
	result = DBMQueryResult();
	return true;
    }

    int numRows, numCols;
    std::map<std::string,int> indexMap;

    numRows = res.num_rows();
    numCols = res.num_fields();
    for(int i = 0; i < numCols; ++i)
    {
	indexMap[res.field_name(i)] = i;
    }
    
    result = DBMQueryResult(numCols,numRows,indexMap);

    for(int i = 0; i < numRows; ++i)
    {
	for(int j = 0; j < numCols; ++j)
	{
	    result(i,j) = res[i][j].c_str();
	}
    }
    return true;
}

void DBManager::writeToCache(const std::string & query, DBMQueryResult & result)
{
    std::string fileHash = md5(query);
    std::string file = _cacheDir + "/" + fileHash + ".dbc";
    //std::cerr << "Writing CacheFile: " << file << std::endl;

    std::ofstream outfile(file.c_str());
    if(outfile.is_open())
    {
	outfile << query << "\n";
	outfile << result.numCols() << "\n";
	outfile << result.numRows() << "\n";

	for(int i = 0; i < result.numCols(); ++i)
	{
	    outfile << result.getColName(i) << "\n";
	}

	for(int i = 0; i < result.numRows(); ++i)
	{
	    for(int j = 0; j < result.numCols(); ++j)
	    {
		outfile << result(i,j) << "\n";
	    }
	}

	outfile.close();
    }
    else
    {
	std::cerr << "DBManager: Unable to open for writing file: " << file << std::endl;
    }
}

DBMQueryResult::DBMQueryResult()
{
}

DBMQueryResult::DBMQueryResult(int cols, int rows, std::map<std::string,int> indexMap)
{
    _data = std::vector<std::vector<std::string> >(rows,std::vector<std::string>(cols,""));
    _indexMap = indexMap;
}

int DBMQueryResult::numRows()
{
    return _data.size();
}

int DBMQueryResult::numCols()
{
    if(_data.size())
    {
	return _data[0].size();
    }

    return 0;
}

const std::string & DBMQueryResult::getColName(int i)
{
    static std::string def;
    for(std::map<std::string,int>::iterator it = _indexMap.begin(); it != _indexMap.end(); ++it)
    {
	if(it->second == i)
	{
	    return it->first;
	}
    }
    return def;
}

int DBMQueryResult::getColIndex(std::string name)
{
    std::map<std::string,int>::iterator it;
    if((it = _indexMap.find(name)) != _indexMap.end())
    {
	return it->second;
    }
    return -1;
}

std::string & DBMQueryResult::operator()(int row, int col)
{
    static std::string def;

    if(row >= 0 && row < _data.size() && col >= 0 && col < _data[row].size())
    {
	return _data[row][col];
    }

    return def;
}

std::string & DBMQueryResult::operator()(int row, std::string colName)
{
    static std::string def;
    int colIndex;

    std::map<std::string,int>::iterator it;
    if((it = _indexMap.find(colName)) != _indexMap.end())
    {
	colIndex = it->second;
    }
    else
    {
	return def;
    }

    if(row >= 0 && row < _data.size() && colIndex >= 0 && colIndex < _data[row].size())
    {
	return _data[row][colIndex];
    }

    return def;
}
