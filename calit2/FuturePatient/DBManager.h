#ifndef FP_DB_MANAGER_H
#define FP_DB_MANAGER_H

#include <mysql++/mysql++.h>

#include <string>
#include <list>
#include <map>
#include <vector>

//typedef std::list<std::map<std::string,std::string> > DBMQueryResult;

class DBMQueryResult
{
    public:
        DBMQueryResult();
        DBMQueryResult(int cols, int rows, std::map<std::string,int> indexMap);

        int numRows();
        int numCols();

        const std::string & getColName(int i);
        int getColIndex(std::string name);

        std::string & operator()(int row, int col);
        std::string & operator()(int row, std::string colName);

    protected:
        std::vector<std::vector<std::string> > _data;
        std::map<std::string,int> _indexMap; 
};

class DBManager
{
    public:
        DBManager(std::string database, std::string server, std::string user, std::string password, std::string cacheDir="");
        ~DBManager();

        bool isConnected();

        void setUseCache(bool b)
        {
            _useCache = b;
        }
        bool getUseCache()
        {
            return _useCache;
        }

        bool runQuery(std::string query, DBMQueryResult & result);

    protected:
        bool loadFromCache(const std::string & query, DBMQueryResult & result);
        bool loadFromDB(const std::string & query, DBMQueryResult & result);

        void writeToCache(const std::string & query, DBMQueryResult & result);

        mysqlpp::Connection * _conn;

        bool _useCache;
        bool _withCache;
        std::string _cacheDir;
};

#endif
