#include "StringOper.h"

uint16 StringLen(const string * cInputPtr)
{
    uint16 u16StrLen;
    if (cInputPtr == nullptr)
    {
        u16StrLen = 0;
    }
    else
    {
        u16StrLen = (uint16)(cInputPtr->length());
    }
    return u16StrLen;
}