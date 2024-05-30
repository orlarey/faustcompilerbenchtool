

int main(int argc, char* argv[])
{
    mydsp* d = new mydsp();

    if (d == nullptr) {
        std::cerr << "Failed to create DSP object\n";
        return 1;
    }
    d->init(44100);

    // Create the input buffers
    FAUSTFLOAT* inputs[256];
    for (int i = 0; i < d->getNumInputs(); i++) {
        inputs[i] = new FAUSTFLOAT[NBSAMPLES];
        for (int j = 0; j < NBSAMPLES; j++) {
            inputs[i][j] = 0.0;
        }
        inputs[i][0] = 1.0;
    }

    // Create the output buffers
    FAUSTFLOAT* outputs[256];
    for (int i = 0; i < d->getNumOutputs(); i++) {
        outputs[i] = new FAUSTFLOAT[NBSAMPLES];
        for (int j = 0; j < NBSAMPLES; j++) {
            outputs[i][j] = 0.0;
        }
    }
    // warmup
    d->compute(NBSAMPLES, inputs, outputs);

    // benchmark
    auto start = std::chrono::high_resolution_clock::now();

    d->compute(NBSAMPLES, inputs, outputs);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> mindur = end - start;

    int N = NBITERATIONS;  // How long the minimum should be stable to be the result
    while (N > 0) {
        start = std::chrono::high_resolution_clock::now();
        d->compute(NBSAMPLES, inputs, outputs);
        end                                    = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        if (duration < mindur) {
            // we have a new minimun
            mindur = duration;
            N      = NBITERATIONS;
        } else {
            // minimun stable so far
            N--;
        }
    }

    // Print the duration in seconds
    std::cout << argv[0] << " " << mindur.count() * 1000 << " ms" << std::endl;

    return 0;
}
