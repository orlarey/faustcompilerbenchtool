
// flasharch_footer.cpp -- the flash judge, appended after bencharch_header.cpp
// and a faust-generated class. Two modes :
//   default        : flash bench. LCG noise input (silence would let a reverb
//                    settle to exact zeros and measure nothing), 512-frame
//                    blocks, warmup then repetitions, prints the BEST ns/frame
//                    (the min is the machine's clean answer, the rest is
//                    scheduling noise).
//   FLASH_IR=<n>   : correctness probe. Impulse on every input at t=0, renders
//                    n samples, prints them at full precision (17 significant
//                    digits) -- one line per sample, one column per output --
//                    plus, after each block, one line with every bargraph zone
//                    (a dropped display is a wrong program too).
// Knobs (environment) : FLASH_COUNT (512), FLASH_WARM (400), FLASH_REPS (30),
// FLASH_BLOCKS (200), FLASH_SPIN_MS (200), FLASH_SCRUB (0 KB).

#include <cstdio>
#include <cstring>
#include <vector>
#include <utility>

static int envInt(const char* name, int dflt)
{
    const char* v = getenv(name);
    return v ? atoi(v) : dflt;
}

// Writes every widget's declared default into its zone. Without this, the
// control values are whatever the heap held (the old C++ backend assigns
// defaults only through buildUserInterface) : on programs whose per-sample
// work depends on a control, the measured cost is the cost of garbage.
// The old backend passes the initial value as the THIRD argument.
struct SetDefaultUI : public UI {
    std::vector<std::pair<const char*, FAUSTFLOAT*>> bargraphs;  // the display zones, for the correctness probe
    void openTabBox(const char*) override {}
    void openHorizontalBox(const char*) override {}
    void openVerticalBox(const char*) override {}
    void closeBox() override {}
    void addButton(const char*, FAUSTFLOAT* z) override { *z = 0; }
    void addCheckButton(const char*, FAUSTFLOAT* z) override { *z = 0; }
    void addVerticalSlider(const char*, FAUSTFLOAT* z, FAUSTFLOAT init, FAUSTFLOAT, FAUSTFLOAT,
                           FAUSTFLOAT) override
    {
        *z = init;
    }
    void addHorizontalSlider(const char*, FAUSTFLOAT* z, FAUSTFLOAT init, FAUSTFLOAT, FAUSTFLOAT,
                             FAUSTFLOAT) override
    {
        *z = init;
    }
    void addNumEntry(const char*, FAUSTFLOAT* z, FAUSTFLOAT init, FAUSTFLOAT, FAUSTFLOAT,
                     FAUSTFLOAT) override
    {
        *z = init;
    }
    void addHorizontalBargraph(const char* l, FAUSTFLOAT* z, FAUSTFLOAT, FAUSTFLOAT) override { bargraphs.push_back({l, z}); }
    void addVerticalBargraph(const char* l, FAUSTFLOAT* z, FAUSTFLOAT, FAUSTFLOAT) override { bargraphs.push_back({l, z}); }
    void addText(const char*) override {}
    void declare(FAUSTFLOAT*, const char*, const char*) override {}
    void declare(const char*, const char*) override {}
};

int main()
{
    const int count  = envInt("FLASH_COUNT", 512);
    const int warm   = envInt("FLASH_WARM", 400);
    const int reps   = envInt("FLASH_REPS", 30);
    const int blocks = envInt("FLASH_BLOCKS", 200);
    const int irlen  = envInt("FLASH_IR", 0);
    const long scrubKB = envInt("FLASH_SCRUB", 0);

    FAUSTCLASS* d = new FAUSTCLASS();
    d->init(44100);
    SetDefaultUI ui;
    d->buildUserInterface(&ui);
    int nins  = d->getNumInputs();
    int nouts = d->getNumOutputs();

    FAUSTFLOAT** in  = new FAUSTFLOAT*[nins ? nins : 1];
    FAUSTFLOAT** out = new FAUSTFLOAT*[nouts ? nouts : 1];
    for (int i = 0; i < nins; i++) {
        in[i] = new FAUSTFLOAT[count];
        memset(in[i], 0, count * sizeof(FAUSTFLOAT));
    }
    for (int i = 0; i < nouts; i++) {
        out[i] = new FAUSTFLOAT[count];
    }

    if (irlen > 0) {
        // correctness probe : impulse in, full-precision samples out
        for (int i = 0; i < nins; i++) {
            in[i][0] = FAUSTFLOAT(1);
        }
        int done = 0;
        while (done < irlen) {
            int n = (irlen - done < count) ? irlen - done : count;
            d->compute(n, in, out);
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < nouts; i++) {
                    printf("%.17g ", double(out[i][j]));
                }
                printf("\n");
            }
            // the displays are part of the program's observable behaviour :
            // one line per block with every bargraph zone, so a candidate
            // that silently dropped its displays cannot pass the gate
            printf("bargraphs");
            for (auto& b : ui.bargraphs) {
                printf(" %.17g", double(*b.second));
            }
            printf("\n");
            for (int i = 0; i < nins; i++) {
                in[i][0] = FAUSTFLOAT(0);  // the impulse lives in the first block only
            }
            done += n;
        }
        return 0;
    }

    unsigned lcg  = 123456789u;
    auto     fill = [&]() {
        for (int i = 0; i < nins; i++) {
            for (int j = 0; j < count; j++) {
                lcg      = lcg * 1664525u + 1013904223u;
                in[i][j] = FAUSTFLOAT(int(lcg >> 9) % 65536 - 32768) / FAUSTFLOAT(32768);
            }
        }
    };

    char*         scrub  = scrubKB ? new char[scrubKB * 1024] : nullptr;
    unsigned char scrubv = 1;
    auto          doScrub = [&]() {
        for (long i = 0; i < scrubKB * 1024; i += 64) {
            scrub[i] = char(i + scrubv);
        }
        scrubv++;
    };

    fill();
    // Core promotion : asymmetric schedulers start short processes on
    // efficiency cores and only promote SUSTAINED work ; a flash run computes
    // mere hundreds of microseconds and would be timed on a lottery-drawn
    // DVFS step. ~200 ms of insistence buys the fast core before anything is
    // timed. FLASH_SPIN_MS=0 disables (forensics).
    {
        double          spinMs = envInt("FLASH_SPIN_MS", 200);
        volatile double spin   = 1.0;
        auto            t0     = std::chrono::steady_clock::now();
        while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                   .count() < spinMs) {
            for (int i = 0; i < 20000; i++) {
                spin = spin * 1.0000001 + 1e-9;
            }
        }
        if (spin < 0) printf("%f", spin);  // keep the loop observable
    }
    for (int w = 0; w < warm; w++) {
        d->compute(count, in, out);
    }

    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        fill();
        double ns;
        if (scrub) {
            double sum = 0;
            for (int b = 0; b < blocks; b++) {
                doScrub();
                auto t0 = std::chrono::steady_clock::now();
                d->compute(count, in, out);
                auto t1 = std::chrono::steady_clock::now();
                sum += std::chrono::duration<double, std::nano>(t1 - t0).count();
            }
            ns = sum / (double(blocks) * double(count));
        } else {
            auto t0 = std::chrono::steady_clock::now();
            for (int b = 0; b < blocks; b++) {
                d->compute(count, in, out);
            }
            auto t1 = std::chrono::steady_clock::now();
            ns = std::chrono::duration<double, std::nano>(t1 - t0).count() /
                 (double(blocks) * double(count));
        }
        if (ns < best) {
            best = ns;
        }
    }
    // the output buffers must stay observable, or the whole loop is dead code
    double sink = 0;
    for (int i = 0; i < nouts; i++) {
        sink += double(out[i][count - 1]);
    }
    printf("%.3f ns/frame (sink %g)\n", best, sink);
    return 0;
}
