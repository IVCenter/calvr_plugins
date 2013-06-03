#ifndef STRINGREPLACE_H_
#define STRINGREPLACE_H_

class stringreplace
{
public:
	static int sprintfiv(char* buffer, const char* pattern, const int* args, unsigned int numargs);
	static int replaceTokens(const char* pattern, const char* tokenlist, const int* tokenvalues, int sizetokenlist, char** replacedString);
	static char* cloneString(const char* str);
	static char* cloneString(const char* str, int strlength);
	static int grabNumberList(const char* text, int** list);
};

#endif /*STRINGREPLACE_H_*/
