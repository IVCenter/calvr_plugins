#include "stringreplace.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <list>

int stringreplace::sprintfiv(char* buffer, const char* pattern, const int* args, unsigned int numargs)
{
	switch (numargs)
	{
		case 0:
			return sprintf(buffer, pattern);
			break;
		case 1:
			return sprintf(buffer, pattern, args[0]);
			break;
		case 2:
			return sprintf(buffer, pattern, args[0], args[1]);
			break;
		case 3:
			return sprintf(buffer, pattern, args[0], args[1], args[2]);
			break;
		case 4:
			return sprintf(buffer, pattern, args[0], args[1], args[2], args[3]);
			break;
		default:
			return 0;
			break;
	}
	return 0;
}


int stringreplace::replaceTokens(const char* pattern, const char* tokenlist, const int* tokenvalues, int sizetokenlist, char** replacedString)
{
	int numreplaceitems = 0;
	int* _activeReplaceList;
	// loop through the image file and find the replacement tags
	size_t ilen = strlen(pattern);
	if (ilen == 0)
	{
		*replacedString = 0;
		return 0;
	}
	for (unsigned int i = 0; i < ilen-2; i++) // ilen -2 because you need at least %xd where x is a valid token
	{
		if (pattern[i] == '%') // start of replacement tag
		{
			bool replacementfound = false;
			for (int t = 0; t < sizetokenlist; t++)
			if (pattern[i+1] == tokenlist[t])
			{
				replacementfound = true;
				break;
			}

			if (!replacementfound)
			{
				fprintf(stderr, "(stringreplace::replaceTokens) Malformed replacement tab. Found \'%c\'. Valid options are:",
					pattern[i+1]);
				for (int t = 0; t < sizetokenlist; t++)
					fprintf(stderr, "\'%c\'=%d.", tokenlist[t], tokenvalues[t]);
				fprintf(stderr, "\n");
				return -1;
			}
			else
			{
				 numreplaceitems++;
				 i++;
			}
		}
	}
	
	if (numreplaceitems > 0)
	{
		_activeReplaceList = new int[numreplaceitems];
	}
	else
	{
		*replacedString = new char[ilen+1];
		memcpy(*replacedString, pattern, ilen+1);
		return 1;
	}

	*replacedString = new char[ilen+1 - numreplaceitems];
	
	for (unsigned int i = 0, n = 0; i < ilen + 1; i++)
	{
		if (pattern[i] == '%') // start of replacement tag
		{
			for (int t = 0; t < sizetokenlist; t++)
			if (pattern[i+1] == tokenlist[t])
				_activeReplaceList[n] = tokenvalues[t];
			(*replacedString)[i - n] = pattern[i];
			n++;
			i++;
		}
		else
		{
			(*replacedString)[i - n] = pattern[i];	
		}
	}
	
	// should make replaceString the buffer and have the allocation above be into something temp
	//
	// also I assume that no entry in the sprintf will insert 12 new characters
	// check return code for sprintf and verify that there was enough space in buffer
	char* buffer = new char[ilen + numreplaceitems * 12];

	stringreplace::sprintfiv(buffer, *replacedString, _activeReplaceList, numreplaceitems);
	delete[] *replacedString;
	delete[] _activeReplaceList;
	*replacedString = buffer;
	return 0;
}

char* stringreplace::cloneString(const char* str, int strlength)
{
	char* tmp = new char[strlength+1];
	strncpy(tmp, str, strlength);
	return tmp;
}

char* stringreplace::cloneString(const char* str)
{
	char* tmp = new char[strlen(str)+1];
	strcpy(tmp, str);
	return tmp;
}

int stringreplace::grabNumberList(const char* numlist, int** list)
{
	
	size_t numtimesteps = 0;
	int start = 0;
	int i;
	char* tlist = new char[strlen(numlist)+1];
	memcpy(tlist, numlist, strlen(numlist)+1);
	for (i = 0; ; i++)
	{
		if (numlist[i] == ' ')
		{
			if (start != i)
			{
				numtimesteps++;
			}
			start = i+1;
			tlist[i] = '\0';
			
		}
		else if (numlist[i] == '\0')
		{
			if (start != i)
			{
				numtimesteps++;
			}
			break;
		}
	}

	std::list<int> timelist;
	if (numtimesteps == 0)
	{
		delete[] tlist;
		return(0);
	}
	start = 0;
	unsigned int n;
	
	for (n = 0, i = 0; n < numtimesteps; i++)
	{
		if (tlist[i] == '\0')
		{
			if (start != i)
			{
				// check to see if there's one timestep or if there's
				// a sequence of timesteps such as 6-8 (6,7,8)
				int s = start;
				int e = start + 1;
				int col = start;
				while (tlist[e] != '\0')
				{
					if (tlist[e] == '-')
					{
						s = e;
						tlist[e] = '\0';
					}
					else if (tlist[e] == ':')
					{
						// first or 2nd colon?
						if (s == start) // first colon
						{
							s = e;
							tlist[e] = '\0';
						}
						else
						{
							col = s;
							s = e;
							tlist[e] = '\0';
						}
					}
					e++;
				}
				int first = strtol(&tlist[start], NULL, 10);
				if (s != start)
				{
					int second = strtol(&tlist[s+1], NULL, 10);
					int increment = 1;
					if (col != start)
					{
						increment = strtol(&tlist[col+1], NULL, 10);
					}
					while (first <= second)
					{
						timelist.push_back(first);
						first += increment;
					}
				}
				else
					timelist.push_back(first);
				n++;
			}
			start = i+1;
		}
	}
	numtimesteps = timelist.size();
	*list = new int[numtimesteps];
	std::list<int>::iterator iter;
	i = 0;
	for (iter = timelist.begin(); iter != timelist.end(); iter++)
	{
		(*list)[i++] = *iter;
	}
	delete[] tlist;
	return (int)numtimesteps;
}
