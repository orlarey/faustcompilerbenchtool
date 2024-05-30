
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>

#define NBSAMPLES 44100
#define NBITERATIONS 1000

#ifndef FAUSTFLOAT
#define FAUSTFLOAT float
#endif

#ifndef FAUSTCLASS
#define FAUSTCLASS mydsp
#endif

#ifdef __APPLE__
#define exp10f __exp10f
#define exp10 __exp10
#endif

#if defined(_WIN32)
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

class UI {
   public:
    virtual ~UI() {}
    virtual void openTabBox(const char* label)                                         = 0;
    virtual void openHorizontalBox(const char* label)                                  = 0;
    virtual void openVerticalBox(const char* label)                                    = 0;
    virtual void closeBox()                                                            = 0;
    virtual void addButton(const char* label, FAUSTFLOAT* zone)                        = 0;
    virtual void addCheckButton(const char* label, FAUSTFLOAT* zone)                   = 0;
    virtual void addVerticalSlider(const char* label, FAUSTFLOAT* zone, FAUSTFLOAT min,
                                   FAUSTFLOAT max, FAUSTFLOAT step, FAUSTFLOAT init)   = 0;
    virtual void addHorizontalSlider(const char* label, FAUSTFLOAT* zone, FAUSTFLOAT min,
                                     FAUSTFLOAT max, FAUSTFLOAT step, FAUSTFLOAT init) = 0;
    virtual void addNumEntry(const char* label, FAUSTFLOAT* zone, FAUSTFLOAT min, FAUSTFLOAT max,
                             FAUSTFLOAT step, FAUSTFLOAT init)                         = 0;
    virtual void addHorizontalBargraph(const char* label, FAUSTFLOAT* zone, FAUSTFLOAT min,
                                       FAUSTFLOAT max)                                 = 0;
    virtual void addVerticalBargraph(const char* label, FAUSTFLOAT* zone, FAUSTFLOAT min,
                                     FAUSTFLOAT max)                                   = 0;
    virtual void addText(const char* text)                                             = 0;
    virtual void declare(float* zone, const char* key, const char* value)              = 0;
    virtual void declare(const char* key, const char* value)                           = 0;
};

class Meta {
   public:
    virtual ~Meta() {}
    virtual void declare(const char* key, const char* value) = 0;
};

class dsp {
   private:
   public:
    virtual ~dsp() {}
    virtual void buildUserInterface(UI* ui_interface)                          = 0;
    virtual void compute(int count, FAUSTFLOAT** inputs, FAUSTFLOAT** outputs) = 0;
    virtual void init(int samplingFreq)                                        = 0;
    virtual void instanceClear()                                               = 0;
    virtual void instanceConstants(int samplingFreq)                           = 0;
    virtual void instanceInit(int samplingFreq)                                = 0;
    virtual void instanceResetUserInterface()                                  = 0;
    virtual int  getNumInputs()                                                = 0;
    virtual int  getNumOutputs()                                               = 0;
    virtual int  getSampleRate()                                               = 0;
    virtual dsp* clone()                                                       = 0;
};

#ifndef FAUSTFLOAT
#define FAUSTFLOAT float
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>

#ifndef FAUSTCLASS
#define FAUSTCLASS mydsp
#endif

#ifdef __APPLE__
#define exp10f __exp10f
#define exp10 __exp10
#endif

#if defined(_WIN32)
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif
