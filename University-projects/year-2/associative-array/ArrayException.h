#ifndef JAKUBOVSKY_PROJEKT_ARRAYEXCEPTION_H
#define JAKUBOVSKY_PROJEKT_ARRAYEXCEPTION_H

#include <string>

using namespace std;

class Exception
{
    string message;
public:
    Exception(const string &text) : message(text) {};
    const string &getMessage() const { return message; };
};

#endif //JAKUBOVSKY_PROJEKT_ARRAYEXCEPTION_H
